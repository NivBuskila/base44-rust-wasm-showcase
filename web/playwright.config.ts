import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

/**
 * Headless verification with no webcam and no GPU.
 *
 * Chromium's fake capture device is fed the synthetic Y4M clip from
 * `scripts/make-fixture-video.mjs`, so `getUserMedia` returns a stream with
 * known, deterministic motion. That is what lets the optical-flow path be
 * asserted against ground truth instead of eyeballed.
 */

const FIXTURE = resolve(import.meta.dirname, 'tests', 'fixtures', 'motion.y4m');

/**
 * Normally left unset, so Playwright uses the browser `npx playwright install`
 * downloaded. Set it to a Chromium binary when running somewhere that already
 * has one at a revision Playwright does not manage (CI images, sandboxes).
 */
const CHROMIUM = process.env.AETHER_CHROMIUM;

export default defineConfig({
  testDir: resolve(import.meta.dirname, 'tests'),
  outputDir: resolve(import.meta.dirname, 'test-results'),
  // The simulation needs to run for a while before the assertions mean
  // anything, and a cold MediaPipe load pulls 12 MB of WASM.
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],

  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    viewport: { width: 1280, height: 720 },
    screenshot: 'only-on-failure',
    permissions: ['camera'],
    launchOptions: {
      ...(CHROMIUM ? { executablePath: CHROMIUM } : {}),
      args: [
        // Auto-accept the camera prompt and serve the fixture as the device.
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-video-capture=${FIXTURE}`,
        // Headless has no real GPU; SwiftShader provides a conformant WebGL2.
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--enable-unsafe-swiftshader',
        '--ignore-gpu-blocklist',
        // Keep rAF running at full rate even though the window is not visible.
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        '--disable-background-timer-throttling',
        '--autoplay-policy=no-user-gesture-required',
      ],
    },
  },

  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    cwd: import.meta.dirname,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
