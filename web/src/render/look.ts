/**
 * The look, as arithmetic — shared by both backends.
 *
 * `styles.ts` is the table; this is what a frame does *with* it: the camera
 * cover fit, the two folds against `frame.intensity`, the particle energy
 * normalisation and size ramp, and the rush focus that only applies once its
 * power is positive. Each of those was written twice, once per backend, which is
 * exactly how the two chains drift apart: the WebGL2 path pushes the numbers
 * through `uniform*` calls and the WebGPU path packs them into a struct, but the
 * numbers themselves must be identical.
 *
 * Pure and DOM-free, pinned by `look.test.ts`. The packing stays per backend:
 * `gpu/frame-uniforms.ts` for WebGPU, the `p.f1`/`p.f2` calls for GL.
 */

import type { ModeStyle } from "./styles";

export const clampLook = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

/**
 * Whether a style shows the camera at all.
 *
 * Not a style choice but an identity: `particles` tints both the body and the
 * rim to black, so running the camera layer there costs a full-frame upload plus
 * five fetches per pixel to add exactly zero. On a software rasteriser that was
 * a quarter of the frame.
 */
export function usesCamera(style: ModeStyle): boolean {
  const sum = (c: readonly number[]): number => c[0] + c[1] + c[2];
  return sum(style.camTint) > 0 || sum(style.camEdge) > 0;
}

/** Aspect ratio of a size, or `fallback` when it has no area. */
export function aspectOf(w: number, h: number, fallback: number): number {
  return w > 0 && h > 0 ? w / h : fallback;
}

/**
 * Cover fit: crop the long axis so the feed fills the canvas at its own aspect
 * ratio, as the pair of scales the scene shader multiplies its camera uv by.
 * Letterboxing would put black bars inside the simulation, and stretching would
 * make gestures land off their landmarks.
 */
export function cameraFit(
  canvasAspect: number,
  videoAspect: number,
): [number, number] {
  const canvas = Math.max(1e-6, canvasAspect);
  const video = videoAspect > 0 ? videoAspect : canvas;
  return video > canvas ? [canvas / video, 1] : [1, video / canvas];
}

/**
 * Motion drives the glow, not the exposure: pushing exposure with movement
 * makes the whole frame pump, while pushing bloom makes the bright parts bloom
 * harder, which is what "a burst of movement blazes" should feel like.
 */
export function bloomAmount(style: ModeStyle, intensity: number): number {
  return style.bloom * (0.72 + 0.75 * intensity);
}

/** The camera view is ungraded, so it must not breathe with motion. */
export function exposureAmount(style: ModeStyle, intensity: number): number {
  return style.exposure * (style.grade > 0 ? 0.96 + 0.22 * intensity : 1);
}

/**
 * Energy normalisation: the pool is user-adjustable from 2k to 220k, and
 * without this the 220k setting is a white screen and the 2k setting is nearly
 * invisible. Total emitted light stays roughly constant instead, and the floor
 * sits low enough that the 1M overdrive pool reads as dense dust.
 */
export function particleGain(style: ModeStyle, count: number): number {
  return style.particles * clampLook(90_000 / count, 0.1, 2.0);
}

/**
 * Dot size, in *target* pixels rather than canvas ones: the scene can be drawn
 * below canvas resolution, and a size in canvas pixels would make the dust
 * swell into blobs when the composite scales it back up. The bigger the pool,
 * the finer the grain.
 */
export function particleSize(pixelScale: number, count: number): number {
  return pixelScale * (3.1 + (1.35 - 3.1) * clampLook(count / 200_000, 0, 1));
}

/** The staged time-warp rush, resolved into what a composite pass needs. */
export interface RushFocus {
  x: number;
  y: number;
  progress: number;
  power: number;
}

/**
 * Reads `[x, y, progress, power]` off the engine, centred and inert whenever no
 * rush is staged. `flipY` is for the GL chain: the engine's origin is the dye
 * grid's (y down) and the GL scene target holds that image flipped, while the
 * WGSL target already holds it the way the screen shows it.
 */
export function rushFocus(
  rush: Float32Array | null | undefined,
  flipY: boolean,
): RushFocus {
  const power =
    rush && rush.length >= 4 && Number.isFinite(rush[3])
      ? Math.max(0, rush[3])
      : 0;
  if (power <= 0) return { x: 0.5, y: 0.5, progress: 0, power: 0 };
  const y = rush![1];
  return { x: rush![0], y: flipY ? 1 - y : y, progress: rush![2], power };
}
