import { expect, test } from '@playwright/test';

/**
 * The baseline contract: the engine loads, the loop runs, the screen is not
 * black, and the simulation never produces a non-finite number.
 *
 * These assertions hold with no camera, no models and no GPU, which is exactly
 * why they are the smoke test — every richer test builds on this being true.
 */

/** Waits until the app has published its test hooks and run a few frames. */
async function waitForEngine(page: import('@playwright/test').Page, minFrames = 30) {
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction(
    (n) => (window.__aether?.diagnostics().frames ?? 0) >= n,
    minFrames,
    { timeout: 60_000 },
  );
}

test('boots, runs the engine, and draws something', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`);
  });

  await page.goto('/');
  await waitForEngine(page, 60);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());

  expect(diag.frames, 'engine should have stepped').toBeGreaterThan(30);
  expect(diag.particleCount, 'particles should be live').toBeGreaterThan(0);

  // Index 13 of the packed stats is the NaN repair count; see constants.ts.
  expect(diag.stats[13], 'simulation produced non-finite values').toBe(0);

  expect(errors, `unexpected console errors:\n${errors.join('\n')}`).toEqual([]);
});

test('the boot overlay clears', async ({ page }) => {
  await page.goto('/');
  await waitForEngine(page);
  await expect(page.locator('#boot')).toHaveClass(/done/);
});

test('renders a non-black frame', async ({ page }) => {
  await page.goto('/');
  await waitForEngine(page, 120);

  // Ambient mode injects colour within a couple of seconds even with no
  // perception at all, so a black centre pixel here means the render path is
  // broken rather than the simulation being quiet.
  await page.waitForFunction(() => (window.__aether?.diagnostics().luminance ?? 0) > 0.01, null, {
    timeout: 30_000,
  });

  const luminance = await page.evaluate(() => window.__aether!.diagnostics().luminance);
  expect(luminance, 'centre pixel is black').toBeGreaterThan(0.01);
});

test('the fake camera stream is picked up', async ({ page }) => {
  await page.goto('/');
  await waitForEngine(page, 90);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.cameraAvailable, 'Chromium fake device should satisfy getUserMedia').toBe(true);
});

test('holds a usable frame rate', async ({ page }) => {
  await page.goto('/');
  await waitForEngine(page, 60);
  // Let the smoothed average settle before reading it.
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) >= 240, null, {
    timeout: 60_000,
  });

  const { fps, stepMs } = await page.evaluate(() => window.__aether!.diagnostics());
  // SwiftShader on a 4-core container is the floor case; the bar is only that
  // the loop is not pathologically slow.
  expect(fps, `fps too low: ${fps}`).toBeGreaterThan(10);
  expect(stepMs, `simulation step too slow: ${stepMs} ms`).toBeLessThan(60);
});
