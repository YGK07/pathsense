/* PathSense service worker: lets the site install as an app and open without a connection.
   The app shell is cached; map data, routes and photos always come from the server when it is reachable. */
const SHELL = "pathsense-shell-v1";
const FILES = ["/", "/static/app.css", "/static/app.js", "/static/icon-192.png", "/static/manifest.webmanifest"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener("activate", (e) => e.waitUntil(caches.keys().then((ks) =>
  Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin) return;     // map tiles, fonts: browser default
  e.respondWith(fetch(e.request).then((r) => {                                 // network first, cache as fallback
    if (r.ok && (u.pathname === "/" || u.pathname.startsWith("/static/") || u.pathname.startsWith("/api/map") ||
                 u.pathname.startsWith("/api/places") || u.pathname.startsWith("/api/real"))) {
      const copy = r.clone(); caches.open(SHELL).then((c) => c.put(e.request, copy));
    }
    return r;
  }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
