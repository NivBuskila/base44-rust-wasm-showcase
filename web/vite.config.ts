import { defineConfig, type Plugin } from 'vite';

// Cross-origin isolation unlocks SharedArrayBuffer, which the threaded engine
// (web/src/wasm-mt) needs for its shared linear memory. `credentialless`
// rather than `require-corp` so the MediaPipe models can still be fetched from
// Google's CDN when they are not vendored locally. Any production host must
// send the same two headers for the multithreaded build to activate.
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

/**
 * Lets the vendored MediaPipe runtime be imported from a module worker.
 *
 * The runtime is a prebuilt Emscripten bundle in `public/mp-wasm/`, loaded
 * either by the worker's `importScripts` shim (a plain XHR) or by a dynamic
 * import. Either way Vite's dev server routes a `.js` request into its
 * transform pipeline, which refuses anything under `public/` on principle
 * ("should not be imported from source code"). Dropping the `?import` query
 * before any internal middleware sees it hands the request back to the static
 * public-file handler, which serves the bundle verbatim.
 *
 * Dev-only: a production build copies `public/` as plain files and never
 * transforms them.
 */
function serveMediaPipeRuntimeAsAsset(): Plugin {
  return {
    name: 'aether:mp-wasm-as-asset',
    configureServer(server) {
      // Registered inside `configureServer` rather than the returned hook, so
      // it sits ahead of Vite's own transform middleware — and ahead of the
      // public-file middleware, which refuses a request that still carries
      // `?import`.
      // Typed structurally: the Node request types are not in this project's
      // `lib`, and `url` is the only field this needs.
      server.middlewares.use((req: { url?: string }, _res: unknown, next: () => void) => {
        if (req.url?.startsWith('/mp-wasm/')) {
          const query = req.url.indexOf('?');
          if (query !== -1) req.url = req.url.slice(0, query);
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [serveMediaPipeRuntimeAsAsset()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    watch: { usePolling: true },
    headers: crossOriginIsolationHeaders,
    // The MediaPipe runtime is 12 MB of WASM served out of public/; without a
    // generous timeout a cold load on a slow disk trips Vite's default.
    warmup: { clientFiles: ['./src/main.ts'] },
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  build: {
    target: 'es2022',
    // The two .task bundles are already compressed; inlining would balloon
    // the JS chunk and defeat browser caching of the models.
    assetsInlineLimit: 4096,
    sourcemap: true,
  },
  // wasm-pack's `--target web` output fetches aether_bg.wasm via
  // `new URL(..., import.meta.url)`, which Vite handles natively. Excluding it
  // from dep optimisation keeps the pre-bundler from rewriting that URL. The
  // threaded build's worker helper additionally does `import('../../..')` to
  // reach its own module, which must stay un-prebundled for the same reason.
  optimizeDeps: {
    exclude: ['./src/wasm/aether.js', './src/wasm-mt/aether.js'],
  },
  worker: { format: 'es' },
});
