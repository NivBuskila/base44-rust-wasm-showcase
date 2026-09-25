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

  await page.goto('/?perception=off');
  await waitForEngine(page, 60);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());

  expect(diag.frames, 'engine should have stepped').toBeGreaterThan(30);
  expect(diag.particleCount, 'particles should be live').toBeGreaterThan(0);

  // Index 13 of the packed stats is the NaN repair count; see constants.ts.
  expect(diag.stats[13], 'simulation produced non-finite values').toBe(0);

  expect(errors, `unexpected console errors:\n${errors.join('\n')}`).toEqual([]);
});

// A fresh browser context is always a first visit, and the first visit waits
// on the welcome's "Enter the field" (or a raised hand). A return visit opens by
// itself once boot finishes. Both must end with `#boot.done`, which is hidden
// and ignores the pointer, so the overlay can never eat a gesture.
test('a return visit clears the boot overlay by itself', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('aether.landing.seen', '1'));
  await page.goto('/?perception=off');
  await waitForEngine(page);
  await expect(page.locator('#boot')).toHaveClass(/done/);
});

test('a first visit clears the boot overlay from the Enter button', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page);
  const enter = page.locator('#boot .landing-enter');
  await expect(enter).toBeEnabled({ timeout: 60_000 });
  await enter.click();
  await expect(page.locator('#boot')).toHaveClass(/done/);
  expect(await page.evaluate(() => localStorage.getItem('aether.landing.seen'))).toBe('1');
});

test('renders a non-black frame', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page, 120);

  // Ambient mode injects colour within a couple of seconds even with no
  // perception at all, so a black centre pixel here means the render path is
  // broken rather than the simulation being quiet.
  // Luminance is a framebuffer readback, so it is polled on its own cadence
  // rather than through `waitForFunction`, which fires every animation frame.
  let luminance = 0;
  for (let attempt = 0; attempt < 40 && luminance <= 0.01; attempt++) {
    luminance = await page.evaluate(() => window.__aether!.luminance());
    if (luminance > 0.01) break;
    await page.waitForTimeout(500);
  }
  expect(luminance, 'centre pixel is black').toBeGreaterThan(0.01);
});

test('the fake camera stream is picked up', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page, 90);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.cameraAvailable, 'Chromium fake device should satisfy getUserMedia').toBe(true);
});

test('the engine step fits the frame budget', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page, 90);

  // The whole engine — fluid solve, particle advection, spells, dye encode —
  // timed as one call, with the render loop stopped. Frame *rate* is
  // deliberately not asserted: headless is fill-rate bound on SwiftShader (fps
  // tracks pixel count almost exactly, 922k px -> 3.7 fps, 518k -> 5.5, 230k
  // -> 8.8, 58k -> 12.1, while dropping all 120k particles moves it only
  // 3.7 -> 4.8), so an fps threshold would measure the software rasteriser.
  //
  // The step has to be timed alone for the same reason. Inside the running
  // loop it shares the CPU with that rasteriser: on a 4-core container its
  // median read ~34 ms there against ~28 ms alone, on either engine build.
  // Stopping the app stops the camera too, so the engine runs on its ambient
  // drive; a short busy gap stands in for the render, so worker threads park
  // between steps as they do live.
  const meanMs = await page.evaluate(() => {
    const app = window.__aether!.app;
    app.stop();
    const engine = app.rawEngine;
    const gap = () => {
      const t = performance.now();
      while (performance.now() - t < 4) {}
    };
    for (let i = 0; i < 10; i++) {
      engine.step(1 / 60);
      gap();
    }
    let total = 0;
    for (let i = 0; i < 40; i++) {
      const t0 = performance.now();
      engine.step(1 / 60);
      total += performance.now() - t0;
      gap();
    }
    return total / 40;
  });

  // ~28 ms is the documented browser cost at the 256x144 grid (README,
  // "Measured"). The bound leaves ~40% headroom for a slower runner: it is
  // there to catch a step that got markedly more expensive, not a noisy one.
  expect(meanMs, `engine step averaged ${meanMs.toFixed(1)} ms`).toBeLessThan(40);
});

test('the render loop does not stall', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page, 60);

  // Cheap to poll and the real failure mode worth catching: a loop that has
  // stopped advancing, from a thrown exception in the frame callback, a lost
  // GL context, or a promise that never settles.
  for (let round = 0; round < 3; round++) {
    const before = await page.evaluate(() => window.__aether!.diagnostics().frames);
    await page.waitForFunction((f) => (window.__aether?.diagnostics().frames ?? 0) > f + 5, before, {
      timeout: 30_000,
    });
  }

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.stats[13], 'simulation produced non-finite values while running').toBe(0);
});
