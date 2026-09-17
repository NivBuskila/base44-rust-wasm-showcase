import { expect, test, type Page } from '@playwright/test';

/**
 * Ground-truth verification of the Rust optical-flow path, with every model
 * disabled.
 *
 * Chromium's fake capture device plays `tests/fixtures/motion.y4m`: a static
 * textured background with one bright disc crossing the frame left-to-right at
 * a known 220 px/s. The camera pipeline mirrors the frame (the whole engine
 * convention is the mirrored view the user sees), so **in engine coordinates
 * the disc moves LEFT** and the reported flow must be negative in x.
 *
 * That sign is the assertion worth making. A magnitude-only test passes just as
 * happily with the direction inverted, and an inverted flow field makes the
 * entire app feel wrong in a way that is hard to trace back to one module.
 */

async function bootWithoutPerception(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  // Detach perception entirely: whatever the models do or do not detect in a
  // synthetic clip must not influence this measurement.
  await page.evaluate(() => window.__aether!.setPerception(null));
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 60, null, {
    timeout: 60_000,
  });
}

/**
 * Samples the dominant flow vector repeatedly.
 *
 * Sampling matters here: the disc wraps from the right edge back to the left,
 * and on those few frames the flow legitimately reverses. A single reading
 * could land on one of them, so the caller works with the distribution.
 */
async function sampleFlow(
  page: Page,
  samples: number,
  intervalMs: number,
): Promise<{ dx: number; dy: number; motion: number }[]> {
  const out: { dx: number; dy: number; motion: number }[] = [];
  for (let i = 0; i < samples; i++) {
    const s = await page.evaluate(() => window.__aether!.diagnostics().stats);
    // Indices 3, 4, 5 are MOTION_ENERGY, FLOW_DX, FLOW_DY; see constants.ts.
    out.push({ motion: s[3], dx: s[4], dy: s[5] });
    await page.waitForTimeout(intervalMs);
  }
  return out;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

test('optical flow detects the moving disc', async ({ page }) => {
  await bootWithoutPerception(page);
  const samples = await sampleFlow(page, 25, 120);

  const motion = samples.map((s) => s.motion);
  expect(
    median(motion),
    `no motion detected in a clip that is nothing but motion: ${JSON.stringify(motion)}`,
  ).toBeGreaterThan(0);

  const anyFinite = samples.every((s) => Number.isFinite(s.dx) && Number.isFinite(s.dy));
  expect(anyFinite, 'flow field went non-finite').toBe(true);
});

test('optical flow reports the correct direction for the mirrored feed', async ({ page }) => {
  await bootWithoutPerception(page);
  const samples = await sampleFlow(page, 30, 120);

  const dx = samples.map((s) => s.dx);
  const negative = dx.filter((v) => v < 0).length;
  const positive = dx.filter((v) => v > 0).length;

  expect(
    negative,
    `the disc moves right in the source and the feed is mirrored, so engine-space ` +
      `flow must be negative in x. Got ${negative} negative vs ${positive} positive ` +
      `samples: ${dx.map((v) => v.toFixed(3)).join(', ')}`,
  ).toBeGreaterThan(positive);

  expect(median(dx), 'x flow magnitude is indistinguishable from zero').toBeLessThan(0);
});

test('sustained motion does not make the flow field run away', async ({ page }) => {
  await bootWithoutPerception(page);

  // The clip loops forever, so flow is being injected continuously. A missing
  // clamp or a per-frame quantity mistaken for a per-second one shows up here
  // as motion energy climbing without bound, rather than settling. Compare the
  // first window against a much later one.
  const early = median((await sampleFlow(page, 10, 100)).map((s) => s.motion));
  await page.waitForTimeout(4000);
  const late = median((await sampleFlow(page, 10, 100)).map((s) => s.motion));

  expect(early, 'baseline motion should be measurable').toBeGreaterThan(0);
  expect(
    late,
    `motion energy grew from ${early} to ${late} over four seconds of the same ` +
      `looping clip; flow is accumulating instead of measuring`,
  ).toBeLessThan(early * 6 + 1);

  const stats = await page.evaluate(() => window.__aether!.diagnostics().stats);
  expect(stats[13], 'sustained flow drive produced non-finite values').toBe(0);
  expect(Number.isFinite(stats[1]), 'fluid max speed went non-finite').toBe(true);
});

test('flow drives the fluid with no models at all', async ({ page }) => {
  await bootWithoutPerception(page);

  await page.evaluate(() => {
    // Isolate the flow path: no hand forces, no ambient drive contribution
    // beyond what the reset clears.
    window.__aether!.setParam('hand_force', 0);
    window.__aether!.setParam('flow_force', 1.5);
    window.__aether!.reset();
  });

  // Index 0 is FLUID_ENERGY. This is the headline claim of the no-ML path: the
  // fluid must respond to real pixel motion with every model disabled.
  await page.waitForFunction(() => (window.__aether?.diagnostics().stats[0] ?? 0) > 0.01, null, {
    timeout: 30_000,
  });

  const stats = await page.evaluate(() => window.__aether!.diagnostics().stats);
  expect(stats[0], 'optical flow alone should energise the fluid').toBeGreaterThan(0.01);
  expect(stats[13], 'flow drive produced non-finite values').toBe(0);
  // Perception is detached, so nothing should be claiming hands or a body.
  expect(stats[9], 'no hands should be reported with perception detached').toBe(0);
});
