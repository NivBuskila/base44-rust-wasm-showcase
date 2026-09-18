/**
 * `importScripts` for a module worker, so MediaPipe's runtime can load.
 *
 * `@mediapipe/tasks-vision` loads its Emscripten runtime by calling
 * `importScripts(wasmLoaderPath)` and, if that throws a `TypeError` — which is
 * what a module worker does, since module workers deliberately have no
 * `importScripts` — falling back to `import(wasmLoaderPath)`.
 *
 * That fallback does not work here. `vision_wasm_internal.js` is a classic
 * script whose top-level `var ModuleFactory` is meant to become a global; loaded
 * as an ES module it stays in module scope instead, and the library fails with
 * "ModuleFactory not set."
 *
 * Restoring `importScripts` is the smaller lie than the alternatives. A classic
 * worker would fix it too, but Vite bundles all workers with one `worker.format`
 * and the threaded engine's rayon workers need `es` — so the choice is per
 * project, not per worker. This keeps the module worker and gives the library
 * the exact semantics it asks for: fetch synchronously, then run in global
 * scope, which is what `importScripts` is.
 *
 * Import this for its side effect before anything loads MediaPipe.
 */

/// <reference lib="webworker" />

type ImportScripts = (...urls: string[]) => void;

const scope = self as unknown as { importScripts?: ImportScripts };

/**
 * Whether the real `importScripts` can actually load anything.
 *
 * A module worker still *has* the function — it is inherited from
 * `WorkerGlobalScope`, so `typeof` reports `'function'` — it just throws
 * "Module scripts don't support importScripts()" on use. Calling it with no
 * arguments is a no-op in a classic worker (the spec iterates an empty list)
 * and throws in a module worker, which makes it a precise capability test.
 */
function importScriptsWorks(): boolean {
  try {
    scope.importScripts?.();
    return true;
  } catch {
    return false;
  }
}

if (!importScriptsWorks()) {
  const polyfill = (...urls: string[]): void => {
    for (const url of urls) {
      // Synchronous by design: `importScripts` is, and the caller relies on the
      // runtime being installed by the time the call returns. Blocking a worker
      // thread is exactly what worker threads are for.
      const request = new XMLHttpRequest();
      request.open('GET', url, false);
      request.send();

      if (request.status !== 0 && (request.status < 200 || request.status >= 300)) {
        throw new Error(`importScripts: ${url} responded ${request.status}`);
      }

      // Indirect eval, so the script's top-level declarations land on the global
      // object the way a classic script's would. A direct `eval` would scope
      // them to this module and reproduce the bug this file exists to fix.
      const globalEval = eval;
      globalEval(request.responseText);
    }
  };

  // Plain assignment would be swallowed: the inherited accessor on
  // `WorkerGlobalScope.prototype` wins, and the override would look like it
  // took while the native version kept throwing.
  Object.defineProperty(self, 'importScripts', {
    value: polyfill,
    writable: true,
    configurable: true,
  });
}
