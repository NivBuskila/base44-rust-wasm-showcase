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

test('the boot overlay clears', async ({ page }) => {
  await page.goto('/?perception=off');
  await waitForEngine(page);
  await expect(page.locator('#boot')).toHaveClass(/done/);
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

  // `stepMs` is the one number here that is a property of this code rather
  // than of the machine: the whole engine — fluid solve, particle advection,
  // spells, dye encode — inside one call, measured in-page.
  //
  // Frame *rate* is deliberately not asserted, and the reason is measured
  // rather than assumed. Headless is fill-rate bound on SwiftShader: fps
  // tracks pixel count almost exactly (922k px -> 3.7 fps, 518k -> 5.5,
  // 230k -> 8.8, 58k -> 12.1) while turning off all 120k particles moves it
  // only 3.7 -> 4.8. That is a software rasteriser shading ~20 texture
  // fetches per pixel across the bloom chain, which a GPU does in single-digit
  // milliseconds. An fps threshold here would measure SwiftShader, and the
  // only way to keep it green would be to weaken the renderer.
  const samples: number[] = [];
  for (let i = 0; i < 12; i++) {
    samples.push(await page.evaluate(() => window.__aether!.diagnostics().stepMs));
    await page.waitForTimeout(250);
  }
  const median = [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)];

  expect(
    median,
    `engine step median ${median.toFixed(1)} ms over ${samples.length} samples ` +
      `[${samples.map((v) => v.toFixed(1)).join(', ')}]`,
  ).toBeLessThan(30);
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
