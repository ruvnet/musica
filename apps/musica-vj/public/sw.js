// Musica VJ web app — app-shell service worker (see ADR-182).
//
// Same policy as the marketing page's docs/sw.js: same-origin shell assets
// are cache-first under a versioned cache name; anything cross-origin (the
// Lyria RealTime WebSocket, Cognitum auth/broker fetches) is always left to
// the network untouched — this worker never intercepts or caches live API
// traffic, only the static app shell.

const CACHE_NAME = "musica-app-shell-v1";
const SHELL_ASSETS = ["./", "./index.html", "./manifest.json"];

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
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // never touch cross-origin (WS/API) traffic

  // Never intercept the WebSocket upgrade path or anything under /assets
  // hashed filenames change per build — always fetch those fresh so a
  // deployed update is visible without a stale cached bundle.
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached && SHELL_ASSETS.some((shell) => request.url.endsWith(shell.replace("./", "")))) return cached;
      return fetch(request)
        .then((response) => {
          if (response.ok && SHELL_ASSETS.some((shell) => request.url.endsWith(shell.replace("./", "")))) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => cached);
    })
  );
});
