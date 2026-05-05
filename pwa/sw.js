const CACHE = 'synapse-v1';
const SHELL = ['/', '/manifest.json', '/icon.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Network-first for API calls
  if (['/inbox', '/pending', '/validate', '/health'].some(p => url.pathname.startsWith(p))) {
    e.respondWith(fetch(e.request).catch(() => new Response('{"error":"offline"}', {
      headers: { 'Content-Type': 'application/json' }
    })));
    return;
  }
  // Cache-first for app shell assets
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
