/* Tests for dashboard/static/auth.js — Wave 1, KAI-DASH-D01.
 *
 * Track A closed the gateway and, in doing so, closed it to the UI. The
 * shim that fixes that sits in front of all 121 fetch() call sites, so a
 * mistake in it is a mistake in every one of them. The cases that matter
 * are the ones where it must NOT act: third-party URLs must never receive
 * the operator's token, and a 403 must not be mistaken for a bad password.
 *
 * Run: node scripts/test_dashboard_ui_auth.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const AUTH_JS = path.join(__dirname, '..', 'dashboard', 'static', 'auth.js');

let passed = 0;
let failed = 0;

function check(name, condition, detail) {
  if (condition) {
    passed += 1;
  } else {
    failed += 1;
    console.log(`  FAIL: ${name}${detail ? ' — ' + detail : ''}`);
  }
}

/* A DOM stub just wide enough for the shim: it builds a sign-in overlay
 * from real elements and appends it to the body. */
function makeElement(tag) {
  return {
    tagName: tag,
    style: { cssText: '' },
    children: [],
    value: '',
    _listeners: {},
    setAttribute() {},
    append(...kids) { this.children.push(...kids); },
    appendChild(kid) { this.children.push(kid); return kid; },
    remove() { this.removed = true; },
    focus() {},
    addEventListener(type, fn) { this._listeners[type] = fn; },
    submit() {
      this._listeners.submit({ preventDefault() {} });
    },
  };
}

function makeContext(opts) {
  const store = new Map();
  const context = {
    console,
    Headers: global.Headers,
    URL: global.URL,
    TextDecoder: global.TextDecoder,
    AbortController: global.AbortController,
    Event: function (type) { this.type = type; },
    sessionStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, v),
      removeItem: (k) => store.delete(k),
    },
    document: {
      body: makeElement('body'),
      createElement: makeElement,
    },
    _store: store,
  };
  context.window = context;
  context.location = { href: 'http://localhost:8080/app', origin: 'http://localhost:8080' };
  context.window.location = context.location;
  context.fetch = opts.fetch;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(AUTH_JS, 'utf8'), context);
  return context;
}

/* The overlay is appended to body; submitting it resolves the prompt. */
function answerPrompt(ctx, token) {
  const overlay = ctx.document.body.children[ctx.document.body.children.length - 1];
  if (!overlay) return false;
  const form = overlay.children[0];
  const input = form.children[2];
  input.value = token;
  form.submit();
  return true;
}

async function settle() {
  for (let i = 0; i < 20; i += 1) await Promise.resolve();
}

// ── Same-origin scoping ──────────────────────────────────────────────

async function testThirdPartyNeverReceivesTheToken() {
  const seen = [];
  const ctx = makeContext({
    fetch: async (input, init) => {
      seen.push({ input, init });
      return { ok: true, status: 200 };
    },
  });
  ctx.window.KaiAuth.setToken('secret-token');

  await ctx.window.fetch('https://cdn.jsdelivr.net/npm/marked/marked.min.js');
  const headers = seen[0].init && seen[0].init.headers;
  check('third-party request carries no Authorization header',
    !headers || !headers.get || !headers.get('Authorization'),
    'the operator token would have leaked to a CDN');
}

async function testSameOriginReceivesTheToken() {
  const seen = [];
  const ctx = makeContext({
    fetch: async (input, init) => {
      seen.push({ input, init });
      return { ok: true, status: 200 };
    },
  });
  ctx.window.KaiAuth.setToken('secret-token');

  await ctx.window.fetch('/api/memories');
  check('same-origin request carries the bearer token',
    seen[0].init.headers.get('Authorization') === 'Bearer secret-token',
    String(seen[0].init.headers.get('Authorization')));
}

async function testRelativeAndAbsoluteSameOriginBothMatch() {
  const seen = [];
  const ctx = makeContext({
    fetch: async (input, init) => { seen.push(init); return { ok: true, status: 200 }; },
  });
  ctx.window.KaiAuth.setToken('t');
  await ctx.window.fetch('/api/a');
  await ctx.window.fetch('http://localhost:8080/api/b');
  check('relative URL is treated as same-origin',
    seen[0].headers.get('Authorization') === 'Bearer t');
  check('absolute same-origin URL is treated as same-origin',
    seen[1].headers.get('Authorization') === 'Bearer t');
}

// ── Prompting ────────────────────────────────────────────────────────

async function testPromptsWhenNoTokenHeld() {
  const seen = [];
  const ctx = makeContext({
    fetch: async (input, init) => { seen.push(init); return { ok: true, status: 200 }; },
  });

  const pending = ctx.window.fetch('/api/memories');
  await settle();
  check('a request with no token opens the sign-in prompt',
    ctx.document.body.children.length === 1);
  answerPrompt(ctx, 'entered-token');
  await pending;
  check('the entered token is used for the request',
    seen[0].headers.get('Authorization') === 'Bearer entered-token');
  check('the entered token is retained for the session',
    ctx.window.KaiAuth.getToken() === 'entered-token');
}

async function test401ClearsTokenAndRetriesOnce() {
  const attempts = [];
  const ctx = makeContext({
    fetch: async (input, init) => {
      attempts.push(init.headers.get('Authorization'));
      return { ok: attempts.length > 1, status: attempts.length === 1 ? 401 : 200 };
    },
  });
  ctx.window.KaiAuth.setToken('stale-token');

  const pending = ctx.window.fetch('/api/memories');
  await settle();
  check('a 401 re-opens the prompt', ctx.document.body.children.length === 1);
  answerPrompt(ctx, 'fresh-token');
  const response = await pending;

  check('the stale token was tried first', attempts[0] === 'Bearer stale-token');
  check('the fresh token was retried', attempts[1] === 'Bearer fresh-token');
  check('exactly one retry is made', attempts.length === 2, String(attempts.length));
  check('the retried response is returned', response.status === 200);
}

async function test403DoesNotReprompt() {
  /* 403 means the token is valid but the role is too narrow. Asking for
   * the password again would be a lie. */
  const attempts = [];
  const ctx = makeContext({
    fetch: async (input, init) => {
      attempts.push(init.headers.get('Authorization'));
      return { ok: false, status: 403 };
    },
  });
  ctx.window.KaiAuth.setToken('valid-but-narrow');

  const response = await ctx.window.fetch('/api/browser/navigate');
  await settle();
  check('a 403 does not re-open the prompt',
    ctx.document.body.children.length === 0);
  check('a 403 is not retried', attempts.length === 1, String(attempts.length));
  check('the 403 is surfaced to the caller', response.status === 403);
  check('a 403 does not discard a valid token',
    ctx.window.KaiAuth.getToken() === 'valid-but-narrow');
}

async function testExistingHeadersArePreserved() {
  const seen = [];
  const ctx = makeContext({
    fetch: async (input, init) => { seen.push(init); return { ok: true, status: 200 }; },
  });
  ctx.window.KaiAuth.setToken('t');
  await ctx.window.fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  check('Content-Type survives the wrapper',
    seen[0].headers.get('Content-Type') === 'application/json');
  check('method survives the wrapper', seen[0].method === 'POST');
  check('body survives the wrapper', seen[0].body === '{}');
}

// ── Session lifetime ─────────────────────────────────────────────────

function testSignOutClearsTheToken() {
  const ctx = makeContext({ fetch: async () => ({ ok: true, status: 200 }) });
  ctx.window.KaiAuth.setToken('t');
  ctx.window.KaiAuth.setToken('');
  check('clearing the token removes it', !ctx.window.KaiAuth.getToken());
}

function testTokenIsNotInLocalStorage() {
  /* The context provides no localStorage at all: if auth.js reached for
   * one it would throw, so this proves the token is session-scoped. */
  const ctx = makeContext({ fetch: async () => ({ ok: true, status: 200 }) });
  ctx.window.KaiAuth.setToken('t');
  check('token is held in sessionStorage', ctx._store.get('kai.dashboard.token') === 't');
  check('no localStorage is used', typeof ctx.localStorage === 'undefined');
}

// ── Event stream ─────────────────────────────────────────────────────

async function testEventStreamSendsCredentialsAndParsesFrames() {
  const seen = [];
  const body = 'data: {"a":1}\n\ndata: {"b":2}\n\n';
  const chunks = [Buffer.from(body, 'utf8')];
  let i = 0;

  const ctx = makeContext({
    fetch: async (input, init) => {
      seen.push(init);
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => (i < chunks.length
              ? { done: false, value: chunks[i++] }
              : { done: true }),
          }),
        },
      };
    },
  });
  ctx.window.KaiAuth.setToken('stream-token');

  const messages = [];
  const stream = ctx.window.KaiAuth.eventStream('/api/events');
  stream.addEventListener('message', (e) => messages.push(e.data));
  await settle();

  check('the event stream authenticates',
    seen[0].headers.get('Authorization') === 'Bearer stream-token',
    String(seen[0].headers.get('Authorization')));
  check('the event stream parses both frames',
    messages.length === 2, JSON.stringify(messages));
  check('frame payloads are extracted',
    messages[0] === '{"a":1}' && messages[1] === '{"b":2}',
    JSON.stringify(messages));
}

async function testEventStreamSurvivesABadListener() {
  const body = 'data: one\n\ndata: two\n\n';
  const chunks = [Buffer.from(body, 'utf8')];
  let i = 0;
  const ctx = makeContext({
    fetch: async () => ({
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => (i < chunks.length
            ? { done: false, value: chunks[i++] }
            : { done: true }),
        }),
      },
    }),
  });
  ctx.window.KaiAuth.setToken('t');

  const good = [];
  const stream = ctx.window.KaiAuth.eventStream('/api/events');
  stream.addEventListener('message', () => { throw new Error('bad listener'); });
  stream.addEventListener('message', (e) => good.push(e.data));
  await settle();
  check('one throwing listener does not stop the stream',
    good.length === 2, JSON.stringify(good));
}

async function testEventStreamReportsFailure() {
  const ctx = makeContext({
    fetch: async () => ({ ok: false, status: 503, body: null }),
  });
  ctx.window.KaiAuth.setToken('t');
  let errored = false;
  const stream = ctx.window.KaiAuth.eventStream('/api/events');
  stream.addEventListener('error', () => { errored = true; });
  await settle();
  check('a failed stream emits an error rather than going quiet', errored);
}

async function testEventStreamKeepsTheEventSourceShape() {
  const ctx = makeContext({
    fetch: async () => ({ ok: false, status: 503, body: null }),
  });
  ctx.window.KaiAuth.setToken('t');
  const stream = ctx.window.KaiAuth.eventStream('/api/events');
  check('exposes addEventListener', typeof stream.addEventListener === 'function');
  check('exposes close', typeof stream.close === 'function');
  let viaProperty = false;
  stream.onerror = () => { viaProperty = true; };
  await settle();
  check('supports the onerror property form', viaProperty);
}

// ── Wiring ───────────────────────────────────────────────────────────

function testEveryApiPageLoadsTheShimFirst() {
  const dir = path.join(__dirname, '..', 'dashboard', 'static');
  for (const name of fs.readdirSync(dir).filter((f) => f.endsWith('.html'))) {
    const html = fs.readFileSync(path.join(dir, name), 'utf8');
    if (!html.includes('fetch(') && !html.includes('EventSource')) continue;
    const shim = html.indexOf('/static/auth.js');
    check(`${name} loads auth.js`, shim !== -1);
    if (shim === -1) continue;
    const firstOther = html.search(/<script src="(?!\/static\/auth\.js)/);
    check(`${name} loads auth.js before any other script`,
      firstOther === -1 || shim < firstOther,
      `auth.js at ${shim}, other script at ${firstOther}`);
    check(`${name} uses no raw EventSource`,
      !html.includes('new EventSource('));
  }
}

async function run() {
  await testThirdPartyNeverReceivesTheToken();
  await testSameOriginReceivesTheToken();
  await testRelativeAndAbsoluteSameOriginBothMatch();
  await testPromptsWhenNoTokenHeld();
  await test401ClearsTokenAndRetriesOnce();
  await test403DoesNotReprompt();
  await testExistingHeadersArePreserved();
  testSignOutClearsTheToken();
  testTokenIsNotInLocalStorage();
  await testEventStreamSendsCredentialsAndParsesFrames();
  await testEventStreamSurvivesABadListener();
  await testEventStreamReportsFailure();
  await testEventStreamKeepsTheEventSourceShape();
  testEveryApiPageLoadsTheShimFirst();

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Dashboard UI Auth Tests: ${passed} passed, ${failed} failed`);
  if (failed) {
    console.log('EXIT GATE: FAIL');
    process.exit(1);
  }
  console.log('EXIT GATE: PASS');
}

/* A hang must fail loudly, not quietly.
 *
 * These tests await promises that the code under test is responsible for
 * resolving. If a change makes one of them hang — say, prompting for a
 * token on a 403, where nothing ever answers the prompt — node's event
 * loop simply empties and the process exits 0 with no output at all. A
 * silent exit reads exactly like a pass on CI.
 *
 * That is not hypothetical: a mutation test that made the shim re-prompt
 * on 403 produced no output and exit 0 before this guard existed. The
 * watchdog and the completion flag turn both failure modes loud. */
let completed = false;

const watchdog = setTimeout(() => {
  console.log('\n  FAIL: the suite hung — a promise was never resolved');
  console.log('EXIT GATE: FAIL');
  process.exit(1);
}, 15000);

process.on('exit', (code) => {
  if (!completed && code === 0) {
    console.log('\n  FAIL: the suite exited without finishing');
    console.log('EXIT GATE: FAIL');
    process.exitCode = 1;
  }
});

run().then(
  () => { completed = true; clearTimeout(watchdog); },
  (err) => {
    clearTimeout(watchdog);
    console.log(`\n  FAIL: the suite threw — ${err && err.stack}`);
    console.log('EXIT GATE: FAIL');
    process.exit(1);
  }
);
