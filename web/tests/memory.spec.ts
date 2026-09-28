import { expect, test } from '@playwright/test';

/**
 * WASM memory can grow in the middle of a frame: `push_hands` and `push_pose`
 * copy their arrays into WASM memory, and on the single-threaded build (the
 * one every iPhone runs, since Safari is never cross-origin isolated) a grow
 * detaches every typed-array view over that memory. The mask and camera writes
 * that follow in the same frame must not go through a view built before it.
 *
 * A real hand is what triggers it in the field; here a scripted perception
 * source grows the memory itself, from inside the perception step, so the
 * failure is deterministic.
 */
test('memory growing mid-frame does not break the mask and camera writes', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));

  await page.goto('/?perception=off&engine=single');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 30, null, {
    timeout: 60_000,
  });
  const before = await page.evaluate(() => window.__aether!.diagnostics());
  expect(before.engine.name).toBe('single');
  expect(before.cameraAvailable, 'the fake camera feeds the luma write').toBe(true);

  const grows = await page.evaluate(async () => {
    const app = window.__aether!.app as unknown as { views: { memory: WebAssembly.Memory } };
    const memory = app.views.memory;
    // Mirrors HAND_BUFFER / POSE_STRIDE in web/src/constants.ts.
    const hands = new Float32Array(2 * (4 + 21 * 3));
    const pose = new Float32Array(1 + 33 * 4);
    const mask = { data: new Float32Array(64 * 36), width: 64, height: 36 };
    let grown = 0;
    window.__aether!.setPerception({
      status: { kind: 'ready', delegate: 'CPU' },
      init: async () => {},
      close: () => {},
      process: () => {
        if (grown < 5) {
          memory.grow(1);
          grown++;
        }
        return { hands, pose, mask, latencyMs: 0.4, anyHand: false };
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 3000));
    return grown;
  });

  expect(grows, 'the scripted source never ran').toBeGreaterThan(0);
  expect(errors, `unexpected page errors:\n${errors.join('\n')}`).toEqual([]);
  const after = await page.evaluate(() => window.__aether!.diagnostics());
  expect(after.frames).toBeGreaterThan(before.frames);
  expect(after.stats[13], 'the simulation produced non-finite values').toBe(0);
});
