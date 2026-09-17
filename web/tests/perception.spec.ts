import { expect, test } from '@playwright/test';

/**
 * MediaPipe perception, verified for real.
 *
 * Kept separate from every other suite, and given a long budget, because on
 * software rasterisation a single inference measured **~750 ms**. That is not
 * representative of a machine with a GPU, but it is what a headless container
 * has, and letting it bleed into unrelated assertions is how a test suite
 * becomes something nobody runs. Everything that does not specifically need
 * the models loads the app with `?perception=off`.
 *
 * The synthetic camera clip contains no hands, so this deliberately does not
 * assert on detections. What it proves is the part that actually breaks: the
 * 12 MB WASM runtime and 14 MB of model bundles resolve and load, a delegate
 * initialises, inference runs on real video frames without throwing, and the
 * adaptive cadence keeps a 750 ms inference from collapsing the render loop.
 */

test.describe.configure({ timeout: 300_000 });

test('the vision models load and run', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`);
  });

  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });

  // The loop must already be running while the models are still downloading —
  // the whole point of loading them off the critical path.
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 5, null, {
    timeout: 60_000,
  });

  await page.waitForFunction(
    () => window.__aether?.diagnostics().perception.kind === 'ready',
    null,
    { timeout: 240_000 },
  );

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  const perception = diag.perception;
  expect(perception.kind).toBe('ready');
  if (perception.kind === 'ready') {
    expect(['GPU', 'CPU']).toContain(perception.delegate);
    console.log(`[perception] delegate: ${perception.delegate}`);
  }

  // Inference has to have actually run on a frame, not just initialised.
  await page.waitForFunction(() => (window.__aether?.diagnostics().inferenceCostMs ?? 0) > 0, null, {
    timeout: 120_000,
  });

  const after = await page.evaluate(() => window.__aether!.diagnostics());
  console.log(
    `[perception] inference ${after.inferenceCostMs.toFixed(0)} ms, ` +
      `cadence ${after.perceptionHz.toFixed(2)} Hz, fps ${after.fps.toFixed(1)}`,
  );

  expect(after.stats[13], 'inference fed values that broke the simulation').toBe(0);
  expect(errors, `perception logged errors:\n${errors.join('\n')}`).toEqual([]);
});

test('inference cost does not collapse the render loop', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(
    () => window.__aether?.diagnostics().perception.kind === 'ready',
    null,
    { timeout: 240_000 },
  );
  await page.waitForFunction(() => (window.__aether?.diagnostics().inferenceCostMs ?? 0) > 0, null, {
    timeout: 120_000,
  });

  const { inferenceCostMs, perceptionHz } = await page.evaluate(() =>
    window.__aether!.diagnostics(),
  );

  // The cadence is derived from measured cost so inference occupies a bounded
  // share of wall time. A fixed 30 Hz cadence against a 750 ms inference means
  // every frame starts a call that takes 22 frames, and the loop runs at
  // inference speed — which is exactly what this used to do.
  const dutyCycle = (inferenceCostMs * perceptionHz) / 1000;
  expect(
    dutyCycle,
    `inference is consuming ${(dutyCycle * 100).toFixed(0)}% of wall time ` +
      `(${inferenceCostMs.toFixed(0)} ms at ${perceptionHz.toFixed(2)} Hz)`,
  ).toBeLessThan(0.55);

  // And the loop must keep advancing while inference runs.
  const before = await page.evaluate(() => window.__aether!.diagnostics().frames);
  await page.waitForFunction((f) => (window.__aether?.diagnostics().frames ?? 0) > f + 20, before, {
    timeout: 60_000,
  });
});

test('the app is fully usable before the models finish loading', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });

  // Everything below happens while MediaPipe is still loading: the Rust
  // optical-flow path, the fluid, particles and the renderer need nothing from
  // it. Awaiting the models before the first frame meant ~25 s of boot screen
  // in front of a simulation that was ready to run.
  await page.waitForFunction(() => window.__aether?.diagnostics().perception.kind === 'loading', null, {
    timeout: 30_000,
  });

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.perception.kind, 'models should still be loading at this point').toBe('loading');
  expect(diag.particleCount, 'particles should already be live').toBeGreaterThan(0);

  const before = diag.frames;
  await page.waitForFunction((f) => (window.__aether?.diagnostics().frames ?? 0) > f + 10, before, {
    timeout: 60_000,
  });
  expect(await page.evaluate(() => window.__aether!.diagnostics().stats[13])).toBe(0);
});
