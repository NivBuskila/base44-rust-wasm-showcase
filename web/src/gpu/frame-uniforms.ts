/**
 * The per-frame scene and composite uniform blocks, with no GPU in sight.
 *
 * Both blocks are positional: the WGSL structs in `shaders/scene.ts` and
 * `shaders/composite.ts` read every slot by index, and the numbers in them are
 * the look — the camera fit, the grade folds against intensity, the rush focus
 * that only applies while its power is positive. That arithmetic used to sit
 * inline between a texture upload and a render pass, where nothing could test
 * it; it is pure here, the way `sim-uniform.ts` is, and
 * `frame-uniforms.test.ts` pins the slot order and the folds.
 *
 * The WebGL2 chain grades from the same `render/styles.ts` table but packs its
 * own uniforms through GL calls, so this module is the WebGPU side only.
 */

import type { ModeStyle } from "../render/styles";
import { COMPOSITE_UNIFORM_FLOATS } from "./shaders/composite";
import { SCENE_UNIFORM_FLOATS } from "./shaders/scene";

/** Whether a style shows the camera at all — no tint and no edge means no. */
export function usesCamera(style: ModeStyle): boolean {
  const sum = (c: readonly number[]): number => c[0] + c[1] + c[2];
  return sum(style.camTint) > 0 || sum(style.camEdge) > 0;
}

/**
 * Aspect-preserving cover fit of the camera texture inside the canvas, as the
 * pair of scales the scene shader multiplies its camera uv by.
 */
export function cameraFit(
  canvasW: number,
  canvasH: number,
  videoW: number,
  videoH: number,
): [number, number] {
  const canvasAspect = Math.max(1e-6, canvasW / Math.max(1, canvasH));
  const videoAspect = videoH > 0 && videoW > 0 ? videoW / videoH : canvasAspect;
  return videoAspect > canvasAspect
    ? [canvasAspect / videoAspect, 1]
    : [1, videoAspect / canvasAspect];
}

export interface SceneUniformInput {
  style: ModeStyle;
  /** Fluid grid size, and its reciprocal, as the shader samples the dye. */
  fluidW: number;
  fluidH: number;
  /** Camera uv scales; see `cameraFit`. */
  camScaleX: number;
  camScaleY: number;
  /** Allocated camera texture size, zero when there is none yet. */
  videoTexW: number;
  videoTexH: number;
  hasVideo: boolean;
  intensity: number;
  time: number;
}

/** Writes one `Scene` block. `f` must hold `SCENE_UNIFORM_FLOATS` floats. */
export function packSceneUniform(f: Float32Array, i: SceneUniformInput): void {
  if (f.length < SCENE_UNIFORM_FLOATS)
    throw new Error("scene uniform scratch is too small");
  const s = i.style;
  f[0] = i.fluidW;
  f[1] = i.fluidH;
  f[2] = 1 / i.fluidW;
  f[3] = 1 / i.fluidH;
  f[4] = i.camScaleX;
  f[5] = i.camScaleY;
  f[6] = 1 / Math.max(1, i.videoTexW);
  f[7] = 1 / Math.max(1, i.videoTexH);
  f[8] = s.camTint[0];
  f[9] = s.camTint[1];
  f[10] = s.camTint[2];
  f[11] = s.camToe;
  f[12] = s.camEdge[0];
  f[13] = s.camEdge[1];
  f[14] = s.camEdge[2];
  f[15] = s.camRaw;
  f[16] = s.dye;
  f[17] = s.bg;
  f[18] = i.hasVideo ? 1 : 0;
  f[19] = i.intensity;
  f[20] = i.time;
  f[21] = 0;
  f[22] = 0;
  f[23] = 0;
}

export interface CompositeUniformInput {
  style: ModeStyle;
  intensity: number;
  frameIndex: number;
  /** `[x, y, hue, power]` of the time-warp rush, when one is staged. */
  rush: Float32Array | null | undefined;
}

/**
 * Writes one `Composite` block. The scene target already holds the image the
 * way the screen shows it, so the rush focus needs no y flip here — unlike the
 * GL chain.
 */
export function packCompositeUniform(
  c: Float32Array,
  i: CompositeUniformInput,
): void {
  if (c.length < COMPOSITE_UNIFORM_FLOATS)
    throw new Error("composite uniform scratch is too small");
  const s = i.style;
  const rush = i.rush;
  const power =
    rush && rush.length >= 4 && Number.isFinite(rush[3])
      ? Math.max(0, rush[3])
      : 0;
  c[0] = s.bloom * (0.72 + 0.75 * i.intensity);
  c[1] = s.exposure * (s.grade > 0 ? 0.96 + 0.22 * i.intensity : 1);
  c[2] = s.aberration;
  c[3] = s.vignette;
  c[4] = s.grade;
  c[5] = i.frameIndex;
  c[6] = power > 0 ? rush![2] : 0;
  c[7] = power;
  c[8] = power > 0 ? rush![0] : 0.5;
  c[9] = power > 0 ? rush![1] : 0.5;
  c[10] = 0;
  c[11] = 0;
}
