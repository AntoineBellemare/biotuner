/* Biotuner Engine — minimal service worker.
   Strategy:
     - App shell: cache-first, fall back to network.
     - /api/*:     network-first, never serve stale analysis.
     - Other GETs: stale-while-revalidate.
*/

const VERSION = 'biotuner-v1'
const APP_SHELL = `${VERSION}-shell`
const RUNTIME   = `${VERSION}-runtime`

const SHELL_URLS = [
  '/',
  '/index.html',
  '/manifest.webmanifest',
  '/favicon.ico',
  '/kairos-logo.webp',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(APP_SHELL).then((c) => c.addAll(SHELL_URLS)).catch(() => null)
  )
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k)))
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return

  const url = new URL(req.url)

  // Skip cross-origin and non-http(s).
  if (url.origin !== self.location.origin) return

  // API: network-first.
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ws/')) {
    event.respondWith(
      fetch(req).catch(() => caches.match(req))
    )
    return
  }

  // Navigation: try network, fall back to cached shell.
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(() => caches.match('/index.html'))
    )
    return
  }

  // Static asset: stale-while-revalidate.
  event.respondWith(
    caches.open(RUNTIME).then(async (cache) => {
      const cached = await cache.match(req)
      const fetched = fetch(req).then((resp) => {
        if (resp && resp.status === 200 && resp.type === 'basic') {
          cache.put(req, resp.clone()).catch(() => null)
        }
        return resp
      }).catch(() => null)
      return cached || fetched || fetch(req)
    })
  )
})
