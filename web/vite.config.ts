import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    // The MediaPipe runtime is 12 MB of WASM served out of public/; without a
    // generous timeout a cold load on a slow disk trips Vite's default.
    warmup: { clientFiles: ['./src/main.ts'] },
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
  // from dep optimisation keeps the pre-bundler from rewriting that URL.
  optimizeDeps: {
    exclude: ['./src/wasm/aether.js'],
  },
  worker: { format: 'es' },
});
