import { expect, test, type Page } from '@playwright/test';

/**
 * End-to-end verification of the gesture chain: packed landmarks -> temporal
 * filtering -> spell latching -> forces -> pixels.
 *
 * A scripted `PerceptionSource` is injected through `window.__aether.setPerception`
 * instead of using MediaPipe, because the real recogniser needs an actual hand in
 * front of an actual camera. Driving the exact same buffer layout the real
 * perception module fills means this still exercises everything downstream of it,
 * deterministically and in milliseconds.
 */

/** Mirrors `HAND_STRIDE` / `HAND_BUFFER` / `POSE_STRIDE` in web/src/constants.ts. */
const GESTURE = {
  None: 0,
  Closed_Fist: 1,
  Open_Palm: 2,
  Pointing_Up: 3,
  Thumb_Down: 4,
  Thumb_Up: 5,
  Victory: 6,
  ILoveYou: 7,
} as const;

interface Scenario {
  present: boolean;
  gestureId: number;
  score: number;
  /** Thumb and index tips touching, regardless of the canned label. */
  pinched: boolean;
  px: number;
  py: number;
  /** Palm drift per inference call, to synthesise hand velocity. */
  drift: number;
  /** Emitted for exactly one call, then cleared — for hysteresis tests. */
  flickerGesture: number;
  calls: number;
}

declare global {
  interface Window {
    __scenario?: Scenario;
  }
}

/**
 * Installs the scripted perception source. Everything lives in page context
 * because `setPerception` takes an object whose methods the render loop calls
 * synchronously.
 */
async function installScriptedPerception(page: Page): Promise<void> {
  await page.evaluate(() => {
    const HAND_LANDMARKS = 21;
    const HAND_STRIDE = 4 + HAND_LANDMARKS * 3;
    const HAND_BUFFER = HAND_STRIDE * 2;
    const POSE_STRIDE = 1 + 33 * 4;

    const hands = new Float32Array(HAND_BUFFER);
    const pose = new Float32Array(POSE_STRIDE);

    const scenario: Scenario = {
      present: false,
      gestureId: 0,
      score: 0.92,
      pinched: false,
      px: 0.5,
      py: 0.5,
      drift: 0,
      flickerGesture: -1,
      calls: 0,
    };
    window.__scenario = scenario;

    /** Hand size in normalised units: wrist-to-middle-MCP distance. */
    const S = 0.13;

    // Finger chains, MediaPipe ordering. Index 0 of each chain is the joint
    // nearest the palm, index 3 is the fingertip.
    const CHAINS: readonly (readonly number[])[] = [
      [1, 2, 3, 4], // thumb (CMC..tip)
      [5, 6, 7, 8], // index (MCP..tip)
      [9, 10, 11, 12], // middle
      [13, 14, 15, 16], // ring
      [17, 18, 19, 20], // pinky
    ];
    // Chain bases in hand-local units: origin at the middle MCP, +y toward
    // the wrist, 1 unit = one hand scale.
    const BASES: readonly (readonly [number, number])[] = [
      [-0.62, 0.5],
      [-0.45, 0.0],
      [0.0, 0.0],
      [0.25, 0.06],
      [0.5, 0.16],
    ];
    // Extended and curled fingertip targets, same local frame.
    const EXTENDED: readonly (readonly [number, number])[] = [
      [-1.15, 0.32],
      [-0.5, -1.3],
      [0.0, -1.42],
      [0.26, -1.3],
      [0.52, -1.08],
    ];
    const CURLED: readonly (readonly [number, number])[] = [
      [-0.5, 0.42],
      [-0.36, 0.34],
      [-0.04, 0.3],
      [0.2, 0.32],
      [0.42, 0.36],
    ];

    /** Which fingers are extended for a given canned gesture label. */
    function extension(gestureId: number): boolean[] {
      switch (gestureId) {
        case 2: // Open_Palm
          return [true, true, true, true, true];
        case 1: // Closed_Fist
          return [false, false, false, false, false];
        case 3: // Pointing_Up
          return [false, true, false, false, false];
        case 6: // Victory
          return [false, true, true, false, false];
        case 5: // Thumb_Up
          return [true, false, false, false, false];
        default:
          return [true, true, true, false, false];
      }
    }

    function writeHand(slot: number, gestureId: number, pinched: boolean): void {
      const base = slot * HAND_STRIDE;
      hands[base] = 1; // present
      hands[base + 1] = 1; // right hand
      hands[base + 2] = gestureId;
      hands[base + 3] = scenario.score;

      const put = (id: number, lx: number, ly: number) => {
        const o = base + 4 + id * 3;
        hands[o] = scenario.px + lx * S;
        hands[o + 1] = scenario.py + ly * S;
        hands[o + 2] = 0;
      };

      put(0, 0, 1.0); // wrist, exactly one scale from the middle MCP

      const ext = extension(gestureId);
      for (let c = 0; c < CHAINS.length; c++) {
        const chain = CHAINS[c];
        const [bx, by] = BASES[c];
        const [tx, ty] = ext[c] ? EXTENDED[c] : CURLED[c];
        for (let j = 0; j < chain.length; j++) {
          const t = j / (chain.length - 1);
          put(chain[j], bx + (tx - bx) * t, by + (ty - by) * t);
        }
      }

      if (pinched) {
        // Thumb tip and index tip nearly touching: ~0.12 hand scales apart,
        // which is what the scale-normalised pinch metric should read as ~1.
        put(4, -0.5, -0.62);
        put(8, -0.44, -0.72);
      }
    }

    window.__aether!.setPerception({
      status: { kind: 'ready', delegate: 'CPU' },
      init: async () => {},
      close: () => {},
      process: () => {
        scenario.calls++;
        hands.fill(0);
        pose.fill(0);

        if (!scenario.present) {
          return { hands, pose, mask: null, latencyMs: 0.4, anyHand: false };
        }

        scenario.px += scenario.drift;
        if (scenario.px > 0.82 || scenario.px < 0.18) scenario.drift = -scenario.drift;

        let gid = scenario.gestureId;
        if (scenario.flickerGesture >= 0) {
          gid = scenario.flickerGesture;
          scenario.flickerGesture = -1;
        }
        writeHand(0, gid, scenario.pinched);

        return { hands, pose, mask: null, latencyMs: 0.4, anyHand: true };
      },
    });
  });
}

async function setScenario(page: Page, patch: Partial<Scenario>): Promise<void> {
  await page.evaluate((p) => {
    Object.assign(window.__scenario!, p);
  }, patch);
}

/** Waits for slot 0's latched spell to settle on `expected`. */
async function expectSpell(page: Page, expected: string): Promise<void> {
  await page
    .waitForFunction(
      (want) => window.__aether?.diagnostics().spells[0] === want,
      expected,
      { timeout: 20_000 },
    )
    .catch(async () => {
      const got = await page.evaluate(() => window.__aether!.diagnostics().spells[0]);
      throw new Error(`expected spell "${expected}" but it stayed "${got}"`);
    });
}

async function boot(page: Page): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 30, null, {
    timeout: 60_000,
  });
  await installScriptedPerception(page);
}

test('a tracked hand registers and suppresses ambient mode', async ({ page }) => {
  await boot(page);
  await setScenario(page, { present: true, gestureId: GESTURE.Open_Palm });

  // Index 9 is HANDS_PRESENT, 15 is AMBIENT; see STAT in web/src/constants.ts.
  await page.waitForFunction(() => (window.__aether?.diagnostics().stats[9] ?? 0) >= 1, null, {
    timeout: 20_000,
  });
  const stats = await page.evaluate(() => window.__aether!.diagnostics().stats);
  expect(stats[9], 'one hand should be tracked').toBe(1);
  expect(stats[15], 'ambient drive should be off while a hand is tracked').toBe(0);
});

test('canned gestures latch to the right spells', async ({ page }) => {
  await boot(page);

  await setScenario(page, { present: true, gestureId: GESTURE.Open_Palm });
  await expectSpell(page, 'repel');

  await setScenario(page, { gestureId: GESTURE.Closed_Fist });
  await expectSpell(page, 'attract');

  await setScenario(page, { gestureId: GESTURE.Pointing_Up });
  await expectSpell(page, 'ignite');

  await setScenario(page, { gestureId: GESTURE.Victory });
  await expectSpell(page, 'freeze');

  await setScenario(page, { gestureId: GESTURE.Thumb_Up });
  await expectSpell(page, 'shatter');
});

test('a pinch wins over a weak canned label', async ({ page }) => {
  await boot(page);
  await setScenario(page, {
    present: true,
    gestureId: GESTURE.None,
    score: 0.1,
    pinched: true,
  });
  await expectSpell(page, 'vortex');
});

test('a single-frame gesture flicker does not change the latched spell', async ({ page }) => {
  await boot(page);
  await setScenario(page, { present: true, gestureId: GESTURE.Closed_Fist });
  await expectSpell(page, 'attract');

  // One inference call reports Open_Palm, then it goes back to the fist. The
  // commit threshold must absorb it — MediaPipe's classifier really does
  // flicker like this, and without hysteresis the visuals twitch constantly.
  await setScenario(page, { flickerGesture: GESTURE.Open_Palm });
  const callsAtFlicker = await page.evaluate(() => window.__scenario!.calls);
  await page.waitForFunction((n) => (window.__scenario?.calls ?? 0) > n + 2, callsAtFlicker, {
    timeout: 20_000,
  });

  const spell = await page.evaluate(() => window.__aether!.diagnostics().spells[0]);
  expect(spell, 'a one-frame flicker should not dislodge the latched spell').toBe('attract');
});

test('losing the hand clears tracking', async ({ page }) => {
  await boot(page);
  await setScenario(page, { present: true, gestureId: GESTURE.Open_Palm });
  await expectSpell(page, 'repel');

  await setScenario(page, { present: false });
  await page.waitForFunction(() => (window.__aether?.diagnostics().stats[9] ?? 1) === 0, null, {
    timeout: 20_000,
  });
  expect(await page.evaluate(() => window.__aether!.diagnostics().stats[9])).toBe(0);
});

test('a moving hand drives the fluid and lights the screen', async ({ page }) => {
  await boot(page);

  // Settle with no hand, so the baseline is not the ambient drive.
  await setScenario(page, { present: false });
  await page.evaluate(() => window.__aether!.setParam('flow_force', 0));
  await page.evaluate(() => window.__aether!.reset());
  await page.waitForTimeout(500);

  await setScenario(page, {
    present: true,
    gestureId: GESTURE.Open_Palm,
    px: 0.3,
    drift: 0.02,
  });

  // Index 0 is FLUID_ENERGY. A hand sweeping across the frame must inject
  // energy through the hand-force path alone, with optical flow disabled.
  await page.waitForFunction(() => (window.__aether?.diagnostics().stats[0] ?? 0) > 0.05, null, {
    timeout: 30_000,
  });
  const energy = await page.evaluate(() => window.__aether!.diagnostics().stats[0]);
  expect(energy, 'a moving hand should inject fluid energy').toBeGreaterThan(0.05);

  await page.waitForFunction(() => (window.__aether?.diagnostics().luminance ?? 0) > 0.005, null, {
    timeout: 30_000,
  });
});

test('the simulation stays finite through a full gesture sequence', async ({ page }) => {
  await boot(page);
  const sequence = [
    GESTURE.Open_Palm,
    GESTURE.Closed_Fist,
    GESTURE.Thumb_Up,
    GESTURE.Victory,
    GESTURE.Pointing_Up,
  ];
  await setScenario(page, { present: true, drift: 0.03, px: 0.25 });

  for (const gestureId of sequence) {
    await setScenario(page, { gestureId });
    await page.waitForTimeout(700);
    const stats = await page.evaluate(() => window.__aether!.diagnostics().stats);
    // Index 13 is NAN_REPAIRS: the engine scrubs non-finite cells and counts
    // them, so any non-zero value means the physics went unstable.
    expect(stats[13], `gesture ${gestureId} produced non-finite values`).toBe(0);
    expect(Number.isFinite(stats[1]), 'max speed went non-finite').toBe(true);
  }

  // Also assert the pinch path, which is geometric rather than label-driven.
  await setScenario(page, { pinched: true, gestureId: GESTURE.None, score: 0.1 });
  await page.waitForTimeout(700);
  expect(await page.evaluate(() => window.__aether!.diagnostics().stats[13])).toBe(0);
});
