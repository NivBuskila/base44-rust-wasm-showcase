/**
 * The per-frame scene and composite uniform blocks, with no GPU in sight.
 *
 * Both blocks are positional: the WGSL structs in `shaders/scene.ts` and
 * `shaders/composite.ts` read every slot by index. The numbers themselves are
 * the shared look (`render/look.ts`), so the two backends fold intensity and fit
 * the camera identically; this module only decides which slot each one lands in.
 * That packing used to sit inline between a texture upload and a render pass,
 * where nothing could test it; `frame-uniforms.test.ts` pins the slot order now.
 */

import { bloomAmount, exposureAmount, rushFocus } from "../render/look";
import type { ModeStyle } from "../render/styles";
import { COMPOSITE_UNIFORM_FLOATS } from "./shaders/composite";
import { SCENE_UNIFORM_FLOATS } from "./shaders/scene";

export interface SceneUniformInput {
  style: ModeStyle;
  /** Fluid grid size, and its reciprocal, as the shader samples the dye. */
  fluidW: number;
  fluidH: number;
  /** Camera uv scales; see `render/look.ts`'s `cameraFit`. */
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
  /** `[x, y, progress, power]` of the time-warp rush, when one is staged. */
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
  const rush = rushFocus(i.rush, false);
  c[0] = bloomAmount(s, i.intensity);
  c[1] = exposureAmount(s, i.intensity);
  c[2] = s.aberration;
  c[3] = s.vignette;
  c[4] = s.grade;
  c[5] = i.frameIndex;
  c[6] = rush.progress;
  c[7] = rush.power;
  c[8] = rush.x;
  c[9] = rush.y;
  c[10] = 0;
  c[11] = 0;
}
