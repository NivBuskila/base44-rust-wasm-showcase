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
 * initialises, inference runs on real video frames without throwing, and a
 * 750 ms inference does not collapse the render loop: on the worker because it
 * never runs there, inline because the adaptive cadence rations it.
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
  // `modelMs` is set by the first completed result on either path; the first
  // one is slow, since MediaPipe compiles its shaders on first use.
  await page.waitForFunction(() => (window.__aether?.diagnostics().modelMs ?? 0) > 0, null, {
    timeout: 120_000,
  });

  const after = await page.evaluate(() => window.__aether!.diagnostics());
  console.log(
    `[perception] ${after.perceptionWorker ? 'worker' : 'inline'}: model ` +
      `${after.modelMs.toFixed(0)} ms, loop paid ${after.inferenceMs.toFixed(0)} ms, ` +
      `fps ${after.fps.toFixed(1)}, status ${after.perception.kind}`,
  );
  // Deliberately no assertion that it is *still* ready: on hardware where
  // inference cannot fit the frame budget, standing down is the correct
  // outcome and the next test covers it. What this test proves is that the
  // models resolved, loaded and ran at all.

  expect(after.stats[13], 'inference fed values that broke the simulation').toBe(0);
  expect(errors, `perception logged errors:\n${errors.join('\n')}`).toEqual([]);
});

test('inference never reclaims the frame budget', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(
    () => window.__aether?.diagnostics().perception.kind === 'ready',
    null,
    { timeout: 240_000 },
  );
  // Before the first completed result there is no cost to judge.
  await page.waitForFunction(() => (window.__aether?.diagnostics().modelMs ?? 0) > 0, null, {
    timeout: 120_000,
  });

  if (await page.evaluate(() => window.__aether!.diagnostics().perceptionWorker)) {
    // On the worker, the default, the model runs off the render loop and the
    // loop pays only for handing it a frame. That guarantee shows in one
    // comparison: a loop that ran the model would pay at least the model's own
    // latency. Checked over several results, starting with the cold first one.
    for (let i = 0; i < 3; i++) {
      const d = await page.evaluate(() => window.__aether!.diagnostics());
      expect(
        d.inferenceMs,
        `the loop paid ${d.inferenceMs.toFixed(0)} ms for a ${d.modelMs.toFixed(0)} ms inference`,
      ).toBeLessThan(d.modelMs);
      await page.waitForFunction(
        (seen) => (window.__aether?.diagnostics().modelMs ?? seen) !== seen,
        d.modelMs,
        { timeout: 60_000 },
      );
    }
    await expectLoopAdvancing(page);
    return;
  }

  // Inline, the fallback, inference runs on the loop and is paced instead.
  // Two outcomes are correct, and which one you get depends on the hardware:
  //
  //  - inference is affordable, so it is throttled to a bounded share of wall
  //    time and keeps running;
  //  - inference is too slow to fit that share even at the maximum interval,
  //    so it is switched off and the app falls back to the model-free optical
  //    flow path, with the HUD saying why.
  //
  // What is *not* correct is the middle ground this used to sit in: clamping
  // the interval at its ceiling and carrying on regardless, which measured at
  // 61% of wall time on a 1224 ms inference — the duty target silently broken
  // by the very cap meant to bound staleness.
  //
  // The invariant is about the settled state, not the first measurement. The
  // opening inferences necessarily overrun the budget: the cost cannot be known
  // before it has been paid once, and the stand-down deliberately waits for
  // three consecutive overruns so MediaPipe's warm-up call does not cost a user
  // hand tracking for the session. So poll until it settles either way — an
  // earlier version of this test sampled once and read 63%, which was the
  // mechanism working, caught mid-decision.
  await page.waitForFunction(
    (limit) => {
      const d = window.__aether?.diagnostics();
      if (!d) return false;
      if (d.perception.kind !== 'ready') return true;
      return (d.inferenceCostMs * d.perceptionBudgetHz) / 1000 <= limit;
    },
    0.55,
    { timeout: 120_000 },
  );

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  const duty = (diag.inferenceCostMs * diag.perceptionBudgetHz) / 1000;

  if (diag.perception.kind === 'ready') {
    expect(
      duty,
      `inference is consuming ${(duty * 100).toFixed(0)}% of wall time ` +
        `(${diag.inferenceCostMs.toFixed(0)} ms at ${diag.perceptionBudgetHz.toFixed(2)} Hz) ` +
        `while still reporting ready`,
    ).toBeLessThan(0.55);
  } else {
    expect(diag.perception.kind).toBe('unavailable');
    if (diag.perception.kind === 'unavailable') {
      // The reason has to be specific enough for a user to act on.
      expect(diag.perception.reason).toMatch(/ms|slow|motion/i);
      console.log(`[perception] stood down: ${diag.perception.reason}`);
    }
  }

  await expectLoopAdvancing(page);
});

/** Whatever perception did, the loop must keep advancing on finite values. */
async function expectLoopAdvancing(page: import('@playwright/test').Page) {
  const before = await page.evaluate(() => window.__aether!.diagnostics().frames);
  await page.waitForFunction((f) => (window.__aether?.diagnostics().frames ?? 0) > f + 20, before, {
    timeout: 60_000,
  });
  expect(await page.evaluate(() => window.__aether!.diagnostics().stats[13])).toBe(0);
}

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
