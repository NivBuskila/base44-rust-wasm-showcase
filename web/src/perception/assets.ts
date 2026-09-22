/**
 * Where the MediaPipe runtime and the two model bundles come from, and how the
 * loader's chatter is kept out of the console.
 *
 * Every function here is about *finding* bytes, never about inference, so the
 * probing rules live in one place instead of being re-derived per asset.
 */

/** Vendored WASM runtime, served from our own origin by `sync-mp-assets.mjs`. */
const WASM_PATH = '/mp-wasm';

/**
 * Public copy of the same runtime, used when the vendored one is not there.
 *
 * `sync-mp-assets.mjs` copies it out of node_modules on install, so it is
 * missing whenever that has not run: a partial install, a network that blocks
 * the npm registry, or any deployment of `dist/` that did not carry the 24 MB
 * along. The models already fall back this way and the runtime did not, which
 * meant perception died with a 404 while a byte-identical copy sat one
 * hostname away. The version is pinned to the installed package so the
 * fallback can never be a different build than the loader expects.
 */
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';

/** Google's public model host, used when the local copy is absent. */
const CDN = 'https://storage.googleapis.com/mediapipe-models';

/** Local path first, CDN second: works offline after one `npm run fetch:models`. */
export const HAND_MODEL = {
  local: '/models/gesture_recognizer.task',
  remote: `${CDN}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`,
};
export const POSE_MODEL = {
  local: '/models/pose_landmarker_lite.task',
  remote: `${CDN}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
};

/**
 * Smallest plausible `.task` bundle. A dev server that answers unknown paths
 * with `index.html` returns 200 for a missing model, so the probe checks the
 * size as well: the real bundles are 5-8 MB.
 */
const MIN_MODEL_BYTES = 1_000_000;

/**
 * Probes for a locally served model and falls back to the CDN, so the app
 * works out of the box and offline after one `npm run fetch:models`.
 *
 * `HEAD` rather than `GET` so the 8 MB body is never pulled twice. A 200 is not
 * proof on its own: a dev server with an SPA fallback answers a missing model
 * with `index.html` and status 200, which is why the content type is checked
 * too. The size is only checked when the server declares one — rejecting a
 * response for lacking `content-length` would fail the local path on servers
 * that stream it.
 */
export async function resolveModel(
  model: { local: string; remote: string },
  probe: typeof fetch = fetch,
): Promise<string> {
  try {
    const res = await probe(model.local, { method: 'HEAD', cache: 'no-store' });
    const declared = res.headers.get('content-length');
    const length = declared === null ? Number.POSITIVE_INFINITY : Number(declared);
    const type = res.headers.get('content-type') ?? '';
    if (res.ok && length >= MIN_MODEL_BYTES && !type.includes('html')) return model.local;
  } catch {
    // A blocked or failed probe is not an error; the CDN is the answer.
  }
  return model.remote;
}

/**
 * Probes for the vendored WASM runtime and falls back to jsdelivr.
 *
 * Mirrors [`resolveModel`], including why a 200 is not proof: a dev server with
 * an SPA fallback answers a missing file with `index.html`. The loader script
 * is the thing probed because `FilesetResolver` fetches it first, so if it is
 * there the binary beside it is too.
 */
export async function resolveWasmPath(probe: typeof fetch = fetch): Promise<string> {
  try {
    const res = await probe(`${WASM_PATH}/vision_wasm_internal.js`, {
      method: 'HEAD',
      cache: 'no-store',
    });
    const type = res.headers.get('content-type') ?? '';
    if (res.ok && !type.includes('html')) return WASM_PATH;
  } catch {
    // A blocked or failed probe is not an error; the CDN is the answer.
  }
  return WASM_CDN;
}

/**
 * Demotes the runtime's `INFO:` chatter for the duration of a load, and
 * returns the undo.
 *
 * MediaPipe pipes its stderr to `console.error`, and building the graphs writes
 * `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` — not an error by
 * any reading, but enough to fail the "no console errors" bar the headless
 * suite holds this app to. Only lines the library itself marks INFO are
 * demoted; everything else goes straight through, so a real failure inside the
 * WASM runtime is still loud.
 */
export function muteInfoLogs(): () => void {
  const original = console.error;
  const filtered = (...args: unknown[]): void => {
    if (typeof args[0] === 'string' && args[0].startsWith('INFO:')) console.info(...args);
    else original.apply(console, args);
  };
  console.error = filtered;
  return () => {
    // Only undo our own wrapper: another module may have wrapped console while
    // 14 MB of models were in flight, and clobbering it would be worse.
    if (console.error === filtered) console.error = original;
  };
}

/** Human-readable form of whatever a rejected promise carried. */
export function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  const text = String(err);
  return text.length > 200 ? `${text.slice(0, 200)}…` : text;
}
