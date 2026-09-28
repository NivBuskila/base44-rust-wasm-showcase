import { expect, test, type Page } from '@playwright/test';

/**
 * The app on a phone, upright and turned sideways.
 *
 * Emulated with a touch screen, so `(pointer: coarse)` matches and the phone
 * paths run: WebGL2 by default, a lower starting quality tier, the touch copy
 * and the phone layout. The sizes are an iPhone with a Dynamic Island in each
 * orientation, full screen and with Safari's bars taking their share, plus the
 * widest iPhone upright, where the narrow-screen welcome rules must still hold.
 * Screenshots land in test-results/ for a human pass; the assertions are about
 * what must never happen on a small screen: the way in cut off below the fold,
 * a label wrapping, the page scrolling sideways, one overlay painted over
 * another.
 */

const PHONES = [
  { name: 'portrait', width: 393, height: 852 },
  { name: 'portrait-safari', width: 393, height: 698 },
  { name: 'landscape', width: 852, height: 393 },
  { name: 'landscape-safari', width: 852, height: 340 },
  { name: 'portrait-max', width: 440, height: 956 },
] as const;

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

function inside(box: Box, w: number, h: number): boolean {
  return box.x >= -0.5 && box.y >= -0.5 && box.x + box.width <= w + 0.5 && box.y + box.height <= h + 0.5;
}

function overlap(a: Box, b: Box): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
}

async function waitForEngine(page: Page): Promise<void> {
  await page.waitForFunction(() => window.__aether !== undefined, null, { timeout: 60_000 });
  await page.waitForFunction(() => (window.__aether?.diagnostics().frames ?? 0) > 30, null, {
    timeout: 60_000,
  });
}

/** Bounding boxes of the visible HUD overlays, keyed by selector. */
async function overlays(page: Page): Promise<Record<string, Box>> {
  return page.evaluate(() => {
    const out: Record<string, { x: number; y: number; width: number; height: number }> = {};
    for (const sel of ['.hud-panel', '.hud-combos', '.hud-hint', '.tut', '.help-sheet']) {
      const el = document.querySelector<HTMLElement>(sel);
      if (!el) continue;
      const style = getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) continue;
      if (el.closest('[hidden]') || el.closest('.hud-help:not(.is-open)')) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      out[sel] = { x: r.x, y: r.y, width: r.width, height: r.height };
    }
    return out;
  });
}

for (const phone of PHONES) {
  test.describe(`on a phone, ${phone.name} ${phone.width}x${phone.height}`, () => {
    test.use({
      viewport: { width: phone.width, height: phone.height },
      deviceScaleFactor: 3,
      isMobile: true,
      hasTouch: true,
    });

    test('the welcome fits, the way in is on screen, and the stage stays uncluttered', async ({ page }) => {
      const errors: string[] = [];
      page.on('pageerror', (e) => errors.push(e.message));

      // A fresh context is a first visit: the full welcome, not the loader.
      await page.goto('/?perception=off');
      await waitForEngine(page);
      const enter = page.locator('#boot .landing-enter');
      await expect(enter).toBeEnabled({ timeout: 60_000 });
      // The armed button must end on its call to action, and it is measured
      // with that label, the widest it gets.
      await expect(page.locator('.landing-enter-label')).toHaveText('ENTER THE FIELD');

      const layout = await page.evaluate(() => {
        const boot = document.getElementById('boot')!;
        const label = document.querySelector<HTMLElement>('.landing-enter-label')!;
        // Line boxes, counted by their tops: `line-height` is `normal` here,
        // so a height ratio would read one line whatever happened.
        const range = document.createRange();
        range.selectNodeContents(label);
        const tops = new Set(Array.from(range.getClientRects(), (r) => Math.round(r.top)));
        const rotate = document.querySelector<HTMLElement>('.landing-rotate');
        return {
          coarse: matchMedia('(pointer: coarse)').matches,
          portrait: matchMedia('(orientation: portrait)').matches,
          bootOverflow: boot.scrollHeight - boot.clientHeight,
          labelLines: tops.size,
          rotateShown: !!rotate && getComputedStyle(rotate).display !== 'none',
          note: document.querySelector('.landing-note')?.textContent ?? '',
        };
      });
      expect(layout.coarse, 'the emulated phone must report a coarse pointer').toBe(true);

      const box = (await enter.boundingBox())!;
      expect(inside(box, phone.width, phone.height), `Enter is off screen: ${JSON.stringify(box)}`).toBe(true);
      expect(layout.bootOverflow, 'the welcome scrolls').toBeLessThanOrEqual(1);
      expect(layout.labelLines, 'ENTER THE FIELD wraps').toBe(1);
      expect(layout.rotateShown, 'the sideways hint belongs to portrait only').toBe(layout.portrait);
      expect(layout.note).toContain('drag to stir');
      await page.screenshot({ path: `test-results/mobile-${phone.name}-welcome.png` });

      const diag = await page.evaluate(() => window.__aether!.diagnostics());
      expect(diag.renderBackend).toBe('webgl2');
      expect(diag.renderGate).toContain('touch device');
      expect(diag.qualityTier, 'a phone starts below full quality').toBeGreaterThanOrEqual(2);

      await enter.tap();
      await expect(page.locator('#boot')).toHaveClass(/done/);
      await page.waitForTimeout(600);

      const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      expect(scrollWidth, 'the page scrolls sideways').toBeLessThanOrEqual(phone.width);

      const staged = await overlays(page);
      for (const [sel, b] of Object.entries(staged)) {
        expect(inside(b, phone.width, phone.height), `${sel} leaves the screen: ${JSON.stringify(b)}`).toBe(true);
      }
      expect(staged['.hud-hint'], 'the first-run hint should be showing').toBeDefined();
      expect(staged['.hud-combos'], 'the spellbook must wait for the hint to clear').toBeUndefined();
      await page.screenshot({ path: `test-results/mobile-${phone.name}-stage.png` });

      // The tutorial card and the help sheet, one at a time, over the stage.
      await page.locator('[data-act="tutorial"]').tap();
      await expect(page.locator('.tut')).toBeVisible();
      await page.waitForTimeout(400);
      const withTutorial = await overlays(page);
      const tut = withTutorial['.tut'];
      expect(tut).toBeDefined();
      expect(inside(tut!, phone.width, phone.height), `the tutorial card leaves the screen: ${JSON.stringify(tut)}`).toBe(true);
      if (withTutorial['.hud-combos']) {
        expect(overlap(tut!, withTutorial['.hud-combos']), 'the tutorial card covers the spellbook').toBe(false);
      }
      await page.screenshot({ path: `test-results/mobile-${phone.name}-tutorial.png` });
      await page.locator('[data-act="tutorial"]').tap();

      await page.locator('[data-act="help"]').tap();
      await expect(page.locator('.hud-help')).toHaveClass(/is-open/);
      await page.waitForTimeout(400);
      const onTop = await page.evaluate(() => {
        const sheet = document.querySelector('.help-sheet')!.getBoundingClientRect();
        const hit = document.elementFromPoint(sheet.x + sheet.width / 2, sheet.y + Math.min(sheet.height / 2, 120));
        return !!hit?.closest('.help-sheet');
      });
      expect(onTop, 'something is painted over the help sheet').toBe(true);
      await page.screenshot({ path: `test-results/mobile-${phone.name}-help.png` });

      const after = await page.evaluate(() => window.__aether!.diagnostics());
      expect(after.stats[13], 'the phone session produced non-finite values').toBe(0);
      expect(errors).toEqual([]);
    });
  });
}
