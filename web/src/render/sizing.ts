/**
 * Frame sizing, as arithmetic — shared by both backends.
 *
 * `look.ts` is what a frame does with the style table; this is what it does with
 * the *canvas*: the DPR ceiling, the drawing-buffer size, the particle view
 * scale and the internal scene scale (pixel budget × governor multiplier, with
 * the `?rscale=` pin outranking both). Each of those was written twice, once per
 * backend, which is exactly how the two chains drift apart — a 5K retina panel
 * must land on the same internal resolution whichever backend won the boot gate.
 *
 * Pure and DOM-free, pinned by `sizing.test.ts`. Reading `window`/the canvas and
 * resizing the targets stays in each renderer.
 */

/**
 * Device pixel ratio ceiling. Beyond 2 the fill cost buys nothing visible for a
 * field this soft, and it halves the frame rate on phones.
 */
export const MAX_DPR = 2;

/**
 * Ceiling on the scene and bloom pixel count, per frame.
 *
 * The chain is fill-bound: roughly twenty texture fetches per output pixel. That
 * is nothing at 1080p on a GPU and ruinous at 5K on a retina panel with the DPR
 * cap doubling it again, so the scene and the bloom are drawn at whatever
 * fraction keeps them under this budget and the composite scales them back up.
 * The composite itself always runs at full canvas resolution, so the grade, the
 * vignette and the dither stay per-pixel and the result does not read as an
 * upscale.
 *
 * 5.2 Mpx leaves 1440p at DPR 2 untouched and pulls 5K retina back to ~0.6. A
 * CPU rasteriser gets a much tighter budget — but one still above 1280x720,
 * because that is the headless test resolution and the screenshots taken there
 * are a deliverable, not just a smoke check.
 */
export const SCENE_PIXEL_BUDGET = 5_200_000;
export const SOFTWARE_PIXEL_BUDGET = 1_600_000;

/** Floor on the internal scale; below this the softening is obvious. */
export const MIN_SCENE_SCALE = 0.4;

/** Viewport width, in CSS pixels, that particle size is calibrated against. */
export const PARTICLE_SIZE_REF_WIDTH = 1200;

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

/**
 * The DPR to render at: a ceiling only, no floor.
 *
 * `devicePixelRatio` drops below 1 whenever the page is zoomed out (0.5 at 50%
 * zoom), and flooring it at 1 there allocates four times the pixels the browser
 * asked for — the opposite of what the user requested. `|| 1` still covers a 0
 * or undefined ratio.
 */
export function deviceScale(ratio: number): number {
  return Math.min(MAX_DPR, ratio || 1);
}

/** Drawing-buffer size for a CSS size at `dpr`, never smaller than a pixel. */
export function bufferSize(
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): [number, number] {
  return [
    Math.max(1, Math.round(cssWidth * dpr)),
    Math.max(1, Math.round(cssHeight * dpr)),
  ];
}

/**
 * Particle size multiplier for a viewport width.
 *
 * A dot sized in CSS pixels is a dot three times larger *relative to the frame*
 * on a 390-wide phone than on a desktop window, which is why the dust reads as
 * coarse confetti there. Size follows the viewport instead, normalised to a
 * typical desktop width, so the grain looks the same fraction of the picture on
 * every screen.
 */
export function viewportScale(cssWidth: number): number {
  return clamp(cssWidth / PARTICLE_SIZE_REF_WIDTH, 0.45, 1);
}

/**
 * Internal scene/bloom scale for a drawing-buffer size.
 *
 * `pinned` is the `?rscale=` diagnostic and outranks both the budget and the
 * governor, so someone comparing resolutions gets the one they asked for.
 */
export function sceneScale(
  width: number,
  height: number,
  opts: {
    budget?: number;
    quality?: number;
    pinned?: number | null;
  } = {},
): number {
  const budget = opts.budget ?? SCENE_PIXEL_BUDGET;
  const quality = opts.quality ?? 1;
  const pixels = Math.max(1, width * height);
  const budgeted = pixels > budget ? Math.sqrt(budget / pixels) : 1;
  return clamp(opts.pinned ?? budgeted * quality, MIN_SCENE_SCALE, 1);
}

/** A target's size at `scale`, never smaller than a pixel. */
export function scaledSize(
  width: number,
  height: number,
  scale: number,
): [number, number] {
  return [
    Math.max(1, Math.round(width * scale)),
    Math.max(1, Math.round(height * scale)),
  ];
}
