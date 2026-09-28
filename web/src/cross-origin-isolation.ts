/**
 * Client half of the cross-origin isolation workaround.
 *
 * See `public/coi-serviceworker.js` for why a Service Worker is involved at
 * all: some hosts and preview proxies strip the `Cross-Origin-Embedder-Policy`
 * header the dev server sends, and without it there is no `SharedArrayBuffer`
 * and therefore no threaded engine.
 *
 * Registering the worker is not enough — the *current* document was already
 * fetched without the headers. One reload, once the worker is in control, is
 * what actually promotes the page. Everything here is best effort: if the
 * worker cannot install, the app carries on and the engine loader falls back to
 * the single-threaded build.
 */

/** Marks that this tab has already spent its one reload, so we never loop. */
const RELOADED_KEY = 'aether:coi-reloaded';
/** One-shot: this document is the reload, so the loader continues rather than re-entering. */
const RESUME_KEY = 'aether:coi-resume';

/** True once, in the document that the isolation reload produced. */
export function resumedFromIsolationReload(): boolean {
  try {
    const resumed = sessionStorage.getItem(RESUME_KEY) === '1';
    sessionStorage.removeItem(RESUME_KEY);
    return resumed;
  } catch {
    return false;
  }
}

/**
 * Makes the document cross-origin isolated if it is not already, reloading once
 * to pick up the Service Worker's headers.
 *
 * Resolves when isolation is settled one way or the other. When a reload is
 * needed it triggers it and never resolves — the page is going away.
 */
export async function ensureCrossOriginIsolation(): Promise<void> {
  // Already isolated: the host sends both headers, nothing to do.
  if (globalThis.crossOriginIsolated) return;

  // In a frame, isolation is a property of the whole tree — the top-level
  // document must opt in too, and a reload here cannot change that. Skip
  // rather than reload an embedded preview forever.
  if (window.top !== window.self) return;

  if (!('serviceWorker' in navigator)) return;

  // Service workers need a secure context; plain-http origins other than
  // localhost cannot register one.
  if (!window.isSecureContext) return;

  // The worker exists to restore headers a proxy stripped. If both arrived and
  // the page is still not isolated, the browser cannot honour them (Safari has
  // no `credentialless`), and a reload would only cost every iPhone visitor a
  // second boot for the same single-threaded result. Only without a
  // controller: a controlling worker adds the headers itself.
  if (!navigator.serviceWorker.controller && (await isolationHeadersArrived())) return;

  try {
    const existing = await navigator.serviceWorker.getRegistration('/');

    if (!existing) {
      await navigator.serviceWorker.register('/coi-serviceworker.js', { scope: '/' });
    }

    // The worker only rewrites responses it actually intercepts, so a document
    // loaded before it took control stays un-isolated until the next
    // navigation. Wait for control, then reload exactly once.
    if (!navigator.serviceWorker.controller) {
      await new Promise<void>((resolve) => {
        navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), {
          once: true,
        });
        // Don't hang the boot if the worker never claims this client.
        setTimeout(resolve, 2000);
      });
    }

    if (!navigator.serviceWorker.controller) return;

    if (sessionStorage.getItem(RELOADED_KEY)) return;
    sessionStorage.setItem(RELOADED_KEY, '1');
    sessionStorage.setItem(RESUME_KEY, '1');

    console.info('[aether] reloading to pick up cross-origin isolation');
    window.location.reload();

    // The reload is in flight; hold the boot sequence rather than racing it.
    await new Promise<void>(() => {});
  } catch (err) {
    console.warn('[aether] could not enable cross-origin isolation', err);
  }
}

/** The probe runs before the engine loads, so it may not hold the boot for long. */
const PROBE_MS = 1500;

/**
 * Whether this document's origin already sends both isolation headers. A HEAD
 * of the current URL is the only way to read response headers from the page;
 * any failure or a slow answer means "no", which keeps the Service Worker path.
 */
async function isolationHeadersArrived(): Promise<boolean> {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), PROBE_MS);
  try {
    const res = await fetch(location.href, { method: 'HEAD', cache: 'no-store', signal: abort.signal });
    return (
      res.headers.has('cross-origin-embedder-policy') && res.headers.has('cross-origin-opener-policy')
    );
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
