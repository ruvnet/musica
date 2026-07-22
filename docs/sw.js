// Musica VJ landing page — app-shell service worker.
//
// This page is a marketing shell, not the app itself (see ADR-181): the only
// job of this worker is to make the shell installable and reachable offline.
// Same-origin shell assets are cache-first (bump CACHE_NAME to invalidate).
// Everything cross-origin (the three.js CDN import) is always network-first
// and never cached here, so a stale worker can never pin a broken/outdated
// copy of a third-party dependency.

const CACHE_NAME = "musica-shell-v1";
const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./manifest.json",
  "./assets/icons/icon-192.png",
  "./assets/icons/icon-512.png",
  "./assets/icons/icon-512-maskable.png",
  "./assets/icons/apple-touch-icon-180.png",
  "./assets/screenshots/musica-vj-neon-fold.png",
  "./assets/screenshots/musica-vj-welcome.png",
  "./assets/screenshots/musica-vj-studio.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    // Cross-origin (e.g. the unpkg three.js module): always go to the
    // network, never cache. Let it fail naturally offline rather than serve
    // a stale build of a dependency we don't control.
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => cached);
    })
  );
});
