import { expect, test, type Page } from '@playwright/test';

/**
 * No camera at all: denied, missing, or an in-app webview that never asks.
 *
 * The app must say so plainly and still give the visitor something to do: a
 * drag paints a blob into the optical-flow plane, so the Rust flow has to read
 * the blob's motion, point the way the pointer moved (the plane is in screen
 * space, so no mirror flip here, unlike `flow.spec.ts`), and fall back to zero
 * when the pointer stops, since the engine applies the last flow every step.
 */

/** Makes every `getUserMedia` call fail the way a declined prompt does. */
async function denyCamera(page: Page): Promise<void> {
  await page.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {
      configurable: true,
      value: () => Promise.reject(new DOMException('Permission denied', 'NotAllowedError')),
    });
    // A return visit opens by itself, so the stage is reachable without a click.
    localStorage.setItem('aether.landing.seen', '1');
  });
}

async function bootWithoutCamera(page: Page): Promise<void> {
  await denyCamera(page);
  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 60, null, {
    timeout: 60_000,
  });
  await expect(page.locator('#boot')).toHaveClass(/done/);
}

/** [MOTION_ENERGY, FLOW_DX]; indices mirror `STAT` in constants.ts. */
async function flow(page: Page): Promise<{ motion: number; dx: number }> {
  const s = await page.evaluate(() => window.__aether!.diagnostics().stats);
  return { motion: s[3], dx: s[4] };
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

test('without a camera the HUD says why and offers dragging instead', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await bootWithoutCamera(page);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.cameraAvailable).toBe(false);
  expect(diag.perception).toMatchObject({ kind: 'unavailable', noCamera: true });

  await expect(page.locator('#hud')).toHaveAttribute('data-camera', 'off');
  await expect(page.locator('[data-hint-head]')).toHaveText('drag to stir the fluid');
  await expect(page.locator('[data-hint-note]')).toBeVisible();
  expect(errors).toEqual([]);
});

test('without a camera a drag stirs the fluid the way it moved, then lets go', async ({ page }) => {
  await bootWithoutCamera(page);
  const { width, height } = page.viewportSize()!;
  const y = height / 2;

  // Still: nothing has been pushed, so there is no flow at all.
  expect((await flow(page)).motion).toBe(0);

  const samples: { motion: number; dx: number }[] = [];
  await page.mouse.move(width * 0.15, y);
  for (let i = 1; i <= 30; i++) {
    await page.mouse.move(width * (0.15 + (0.7 * i) / 30), y);
    await page.waitForTimeout(40);
    samples.push(await flow(page));
  }

  const moving = samples.filter((s) => s.motion > 1e-4);
  expect(moving.length, 'the drag never registered as motion').toBeGreaterThan(5);
  expect(median(moving.map((s) => s.dx)), 'a rightward drag must push right').toBeGreaterThan(0);

  // Released: the settle planes bring the flow back to zero.
  await expect.poll(async () => (await flow(page)).motion, { timeout: 5_000 }).toBeLessThan(1e-4);

  const diag = await page.evaluate(() => window.__aether!.diagnostics());
  expect(diag.stats[13], 'stirring produced non-finite values').toBe(0);
});
