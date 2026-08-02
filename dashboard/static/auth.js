/* Dashboard credentials for the browser UI — Wave 1 Track A (KAI-DASH-D01).
 *
 * Track A closed the gateway to anonymous callers, which also closed it to
 * its only real client: the UI makes 121 fetch() calls that carried no
 * credential. This attaches one, in one place, rather than editing every
 * call site — and keeps the credential out of URLs and out of cookies.
 *
 * Why sessionStorage and not localStorage: the token is the operator's
 * full authority over the stack. sessionStorage drops it when the tab
 * closes, which costs one re-entry per session and removes a persistent
 * theft target. There is no Content-Security-Policy on these pages yet
 * (KAI-DASH-088, Track I), so script injection would reach either store —
 * the shorter lifetime is the only real mitigation available today.
 *
 * Why the header and not a cookie: a cookie would be attached
 * automatically by the browser on cross-site requests, which is exactly
 * the CSRF exposure the Authorization header avoids. See DECISIONS D142.
 */
(function () {
  'use strict';

  const STORAGE_KEY = 'kai.dashboard.token';
  const nativeFetch = window.fetch.bind(window);

  function getToken() {
    try {
      return sessionStorage.getItem(STORAGE_KEY) || '';
    } catch (e) {
      return '';
    }
  }

  function setToken(token) {
    try {
      if (token) sessionStorage.setItem(STORAGE_KEY, token);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) { /* private browsing; the session simply will not persist */ }
  }

  /* Same-origin only. A blanket wrapper that attached the token to every
   * request would hand the operator's authority to any third-party URL
   * the page happens to fetch. */
  function isSameOrigin(input) {
    try {
      const url = new URL(
        typeof input === 'string' ? input : (input && input.url) || '',
        window.location.href
      );
      return url.origin === window.location.origin;
    } catch (e) {
      return false;
    }
  }

  // ── Sign-in prompt ────────────────────────────────────────────────

  let promptOpen = null;

  function promptForToken(message) {
    if (promptOpen) return promptOpen;

    promptOpen = new Promise(function (resolve) {
      const overlay = document.createElement('div');
      overlay.setAttribute('role', 'dialog');
      overlay.setAttribute('aria-modal', 'true');
      overlay.setAttribute('aria-label', 'Dashboard sign in');
      overlay.style.cssText =
        'position:fixed;inset:0;z-index:99999;display:flex;align-items:center;' +
        'justify-content:center;background:rgba(8,8,10,0.92);' +
        'font-family:system-ui,-apple-system,sans-serif;';

      const card = document.createElement('form');
      card.style.cssText =
        'background:#141418;border:1px solid #2a2a32;border-radius:12px;' +
        'padding:28px;width:min(420px,90vw);color:#e8e8ec;' +
        'box-shadow:0 20px 60px rgba(0,0,0,0.5);';

      const title = document.createElement('h2');
      title.textContent = 'Dashboard sign in';
      title.style.cssText = 'margin:0 0 8px;font-size:1.15rem;font-weight:600;';

      const hint = document.createElement('p');
      hint.textContent = message ||
        'This dashboard controls the whole stack, so it requires a token.';
      hint.style.cssText =
        'margin:0 0 18px;font-size:0.85rem;line-height:1.5;color:#a0a0ac;';

      const input = document.createElement('input');
      input.type = 'password';
      input.autocomplete = 'current-password';
      input.required = true;
      input.setAttribute('aria-label', 'Dashboard token');
      input.placeholder = 'KAI_DASHBOARD_TOKEN';
      input.style.cssText =
        'width:100%;box-sizing:border-box;padding:11px 13px;border-radius:8px;' +
        'border:1px solid #33333d;background:#0d0d10;color:#e8e8ec;' +
        'font-family:ui-monospace,SFMono-Regular,monospace;font-size:0.9rem;';

      const button = document.createElement('button');
      button.type = 'submit';
      button.textContent = 'Continue';
      button.style.cssText =
        'margin-top:16px;width:100%;padding:11px;border-radius:8px;border:0;' +
        'background:#4a6cf7;color:#fff;font-size:0.92rem;font-weight:600;' +
        'cursor:pointer;';

      const note = document.createElement('p');
      note.textContent =
        'Held for this tab only, and cleared when it closes.';
      note.style.cssText =
        'margin:14px 0 0;font-size:0.75rem;color:#70707c;text-align:center;';

      card.append(title, hint, input, button, note);
      overlay.appendChild(card);

      card.addEventListener('submit', function (event) {
        event.preventDefault();
        const value = input.value.trim();
        if (!value) return;
        setToken(value);
        overlay.remove();
        promptOpen = null;
        resolve(value);
      });

      document.body.appendChild(overlay);
      input.focus();
    });

    return promptOpen;
  }

  // ── Authenticated fetch ───────────────────────────────────────────

  function withToken(init, token) {
    const next = Object.assign({}, init || {});
    const headers = new Headers((init && init.headers) || {});
    headers.set('Authorization', 'Bearer ' + token);
    next.headers = headers;
    return next;
  }

  window.fetch = async function (input, init) {
    if (!isSameOrigin(input)) return nativeFetch(input, init);

    let token = getToken();
    if (!token) token = await promptForToken();

    let response = await nativeFetch(input, withToken(init, token));

    /* 401 means the token is wrong or absent; ask again and retry once.
     * 403 means the token is valid but the role is too narrow — re-prompting
     * would be a lie, so it is surfaced to the caller as-is. */
    if (response.status === 401) {
      setToken('');
      const retry = await promptForToken(
        'That token was not accepted. Check KAI_DASHBOARD_TOKEN.'
      );
      response = await nativeFetch(input, withToken(init, retry));
    }
    return response;
  };

  // ── Authenticated event stream ────────────────────────────────────

  /* EventSource cannot send headers at all, so /api/events would be
   * unreachable however well the fetch path is wired. This reads the same
   * stream over fetch and emits the same `message` / `error` events, so
   * callers keep the EventSource shape they already use. */
  function AuthenticatedEventStream(url) {
    const listeners = { message: [], error: [], open: [] };
    let closed = false;
    const controller = new AbortController();

    const api = {
      addEventListener: function (type, fn) {
        (listeners[type] || (listeners[type] = [])).push(fn);
      },
      close: function () {
        closed = true;
        controller.abort();
      },
    };
    ['onmessage', 'onerror', 'onopen'].forEach(function (prop) {
      Object.defineProperty(api, prop, {
        set: function (fn) { listeners[prop.slice(2)].push(fn); },
        configurable: true,
      });
    });

    function emit(type, event) {
      (listeners[type] || []).forEach(function (fn) {
        try { fn(event); } catch (e) { /* one bad listener must not stop the stream */ }
      });
    }

    (async function pump() {
      try {
        const response = await window.fetch(url, {
          headers: { Accept: 'text/event-stream' },
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          emit('error', new Event('error'));
          return;
        }
        emit('open', new Event('open'));

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (!closed) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });

          let split;
          while ((split = buffer.indexOf('\n\n')) !== -1) {
            const frame = buffer.slice(0, split);
            buffer = buffer.slice(split + 2);
            const data = frame
              .split('\n')
              .filter(function (line) { return line.indexOf('data:') === 0; })
              .map(function (line) { return line.slice(5).trim(); })
              .join('\n');
            if (data) emit('message', { data: data });
          }
        }
      } catch (e) {
        if (!closed) emit('error', new Event('error'));
      }
    })();

    return api;
  }

  window.KaiAuth = {
    getToken: getToken,
    setToken: setToken,
    signOut: function () { setToken(''); window.location.reload(); },
    eventStream: AuthenticatedEventStream,
  };
})();
