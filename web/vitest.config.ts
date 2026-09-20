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
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'jsdom',
  },
});
