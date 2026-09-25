import { defineConfig } from 'vitest/config';

/**
 * Unit tests for the browser-side modules that have logic worth pinning
 * without a GPU, a camera or the WASM engine: the quality governor, the QA
 * recorder, the camera's error mapping and luma extraction, the HUD's pure
 * helpers.
 *
 * Kept apart from `vite.config.ts` so the dev-server plugins (COOP/COEP
 * headers, the MediaPipe asset shim, the QA drop box) never run under the test
 * runner, and from `tests/`, which is Playwright's.
 */
export default defineConfig({
  resolve: {
    // `wgsl_reflect` ships a CommonJS `main` inside a `"type": "module"`
    // package, which Node refuses to load; point at its real ESM build.
    alias: { wgsl_reflect: 'wgsl_reflect/wgsl_reflect.module.js' },
  },
  test: {
    include: ['src/**/*.test.ts'],
    // Most suites are DOM-free, and building a jsdom per file was ~90% of the
    // run. A suite that needs one says so in its first line with
    // `// @vitest-environment jsdom`.
    environment: 'node',
  },
});
