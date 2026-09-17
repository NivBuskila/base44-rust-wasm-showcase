import { expect, test, type Page } from '@playwright/test';

/**
 * Visual verification: every view mode renders, none of them crashes, and the
 * result is not a black screen.
 *
 * Screenshots go to `test-results/` unconditionally rather than only on
 * failure, because "it renders" and "it looks right" are different claims and
 * only the second one matters for this project. The assertions cover the first;
 * the artefacts exist so a human can check the second.
 */

const MODES = ['aether', 'camera', 'debug', 'particles'] as const;

async function boot(page: Page, minFrames = 150): Promise<void> {
  await page.goto('/');
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction((n) => (window.__aether?.diagnostics().frames ?? 0) >= n, minFrames, {
    timeout: 60_000,
  });
}

test('every view mode renders without errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`);
  });

  await boot(page);

  const luminance: Record<string, number> = {};
  for (const mode of MODES) {
    await page.evaluate((m) => window.__aether!.forceMode(m), mode);
    // Let the mode take effect over several frames before measuring.
    await page.waitForFunction(() => window.__aether!.diagnostics().mode !== undefined, null, {
      timeout: 10_000,
    });
    await page.waitForTimeout(600);

    const diag = await page.evaluate(() => window.__aether!.diagnostics());
    expect(diag.mode, `forceMode('${mode}') did not take`).toBe(mode);
    expect(diag.stats[13], `mode '${mode}' produced non-finite values`).toBe(0);
    luminance[mode] = diag.luminance;

    await page.screenshot({ path: `test-results/view-${mode}.png` });
  }

  expect(errors, `unexpected console errors:\n${errors.join('\n')}`).toEqual([]);

  // `aether`, `camera` and `debug` all have a guaranteed non-black source: the
  // ambient dye drive, the camera feed, and the debug texture's 0.5-biased flow
  // channels respectively. `particles` legitimately can be near-black when the
  // pool is sparse, so it is exempt from the brightness floor.
  for (const mode of ['aether', 'camera', 'debug'] as const) {
    expect(luminance[mode], `mode '${mode}' rendered a black frame`).toBeGreaterThan(0.004);
  }
});

test('the canvas fills the viewport at device pixel ratio', async ({ page }) => {
  await boot(page, 60);

  const box = await page.evaluate(() => {
    const canvas = document.getElementById('stage') as HTMLCanvasElement;
    return {
      cssW: canvas.clientWidth,
      cssH: canvas.clientHeight,
      bufW: canvas.width,
      bufH: canvas.height,
      dpr: Math.min(2, window.devicePixelRatio || 1),
      innerW: window.innerWidth,
      innerH: window.innerHeight,
    };
  });

  expect(box.cssW, 'canvas should span the viewport width').toBe(box.innerW);
  expect(box.cssH, 'canvas should span the viewport height').toBe(box.innerH);
  // Rounding means this can be off by one pixel; a factor-of-two error is the
  // bug worth catching (a canvas sized in CSS pixels looks soft on retina).
  expect(Math.abs(box.bufW - box.cssW * box.dpr)).toBeLessThanOrEqual(1);
  expect(Math.abs(box.bufH - box.cssH * box.dpr)).toBeLessThanOrEqual(1);
});

test('survives a resize without losing the context', async ({ page }) => {
  await boot(page, 60);

  for (const size of [
    { width: 640, height: 480 },
    { width: 1920, height: 1080 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(size);
    await page.waitForTimeout(400);
    const diag = await page.evaluate(() => window.__aether!.diagnostics());
    expect(diag.stats[13], `resize to ${size.width}x${size.height} broke the simulation`).toBe(0);
    await page.screenshot({ path: `test-results/resize-${size.width}x${size.height}.png` });
  }

  // The loop must still be advancing after all that.
  const before = await page.evaluate(() => window.__aether!.diagnostics().frames);
  await page.waitForTimeout(500);
  const after = await page.evaluate(() => window.__aether!.diagnostics().frames);
  expect(after, 'the render loop stopped after resizing').toBeGreaterThan(before);
});

test('the particle pool can be resized while running', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console.error: ${m.text()}`);
  });

  await boot(page, 60);

  // Resizing the pool is the one operation that reallocates engine buffers and
  // therefore detaches every typed-array view JS holds over WASM memory. If the
  // views are not rebuilt, the very next frame either throws
  // "detached ArrayBuffer" or renders garbage. Growing past the initial
  // allocation also forces WASM memory to grow, which detaches them again.
  for (const n of [2_000, 200_000, 50_000]) {
    await page.evaluate((count) => window.__aether!.setParticleCount(count), n);

    const before = await page.evaluate(() => window.__aether!.diagnostics().frames);
    await page.waitForFunction((f) => (window.__aether?.diagnostics().frames ?? 0) > f + 10, before, {
      timeout: 20_000,
    });

    const diag = await page.evaluate(() => window.__aether!.diagnostics());
    expect(diag.particleCount, `pool should hold ${n} particles`).toBe(n);
    expect(diag.stats[13], `resizing the pool to ${n} produced non-finite values`).toBe(0);
  }

  expect(errors, `resizing the pool logged errors:\n${errors.join('\n')}`).toEqual([]);
});
