/**
 * Cross-origin isolation via Service Worker.
 *
 * The threaded engine needs `SharedArrayBuffer`, which the browser only hands
 * out to a cross-origin isolated document — that is, one served with both
 * `Cross-Origin-Opener-Policy: same-origin` and
 * `Cross-Origin-Embedder-Policy`. `vite.config.ts` sends both, but a host or
 * preview proxy in front of the dev server is free to drop them, and several
 * do: COOP survives, COEP is stripped, and isolation silently never happens.
 *
 * Headers cannot be added from the page — but a Service Worker sits in front of
 * the network and re-issues every response, so it can attach them itself. Once
 * this worker controls the page, the next navigation is isolated regardless of
 * what the upstream host sent.
 *
 * `credentialless` rather than `require-corp`, matching the dev server: it lets
 * cross-origin subresources (the MediaPipe models on Google's CDN) load without
 * a CORP header, at the cost of sending them without credentials — which is
 * exactly what a public CDN wants anyway.
 */

const COEP = 'credentialless';

self.addEventListener('install', () => self.skipWaiting());

// Claim the pages that are already open, so the client half can reload straight
// into an isolated document instead of waiting for a second navigation.
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const request = event.request;

  // A `only-if-cached` request outside same-origin mode cannot be re-issued;
  // passing it to fetch() throws. Leave it to the browser.
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;

  // Under `credentialless`, no-cors cross-origin loads must go out without
  // credentials or the response is opaque-blocked.
  const outgoing =
    COEP === 'credentialless' && request.mode === 'no-cors'
      ? new Request(request, { credentials: 'omit' })
      : request;

  event.respondWith(
    fetch(outgoing).then((response) => {
      // Opaque responses have an immutable, empty header list; rebuilding one
      // would only destroy it.
      if (response.status === 0) return response;

      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Embedder-Policy', COEP);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }),
  );
});
