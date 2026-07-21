// Service worker — offline-capable app shell for Sovereign AI dashboard
const CACHE = 'kai-shell-v1';

const SHELL_ASSETS = [
  '/',
  '/app',
  '/static/index.html',
  '/static/app.html',
  '/static/chat.html',
  '/static/style.css',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/manifest.json',
];

// Install: cache the app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

// Activate: remove old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: network-first for API calls; cache-first for shell assets
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never intercept API, SSE, or WebSocket routes
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/stream') ||
    url.pathname.startsWith('/health') ||
    event.request.headers.get('accept') === 'text/event-stream'
  ) {
    return;
  }

  // Network-first for navigation and dynamic routes
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() =>
        caches.match('/app') || caches.match('/')
      )
    );
    return;
  }

  // Cache-first for static assets
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response && response.status === 200 && response.type === 'basic') {
          const clone = response.clone();
          caches.open(CACHE).then((cache) => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});
