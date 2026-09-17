import { defineConfig } from 'vite';

// Cross-origin isolation unlocks SharedArrayBuffer, which the threaded engine
// (web/src/wasm-mt) needs for its shared linear memory. `credentialless`
// rather than `require-corp` so the MediaPipe models can still be fetched from
// Google's CDN when they are not vendored locally. Any production host must
// send the same two headers for the multithreaded build to activate.
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
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
