/**
 * The WebGL2 particle and composite draws, beside `scene-pass.ts`.
 *
 * Both are the per-frame uniform block over objects owned elsewhere (the pool's
 * buffer in `particle-stream.ts`, the targets in `resources.ts`), so they are
 * plain functions: the renderer keeps the frame order and the scales, these keep
 * the uniform names.
 */

import type { RenderFrame } from "../types";
import {
  bloomAmount,
  exposureAmount,
  particleGain,
  particleSize,
  rushFocus,
} from "./look";
import { ParticleStream } from "./particle-stream";
import type { Resources } from "./resources";
import type { ModeStyle } from "./styles";

/**
 * Streams this frame's pool and draws it additively into the scene target.
 * `pixelScale` is DPR × scene scale × viewport scale, the same product the
 * WebGPU draw uses.
 */
export function drawParticlePass(
  res: Resources,
  frame: RenderFrame,
  style: ModeStyle,
  intensity: number,
  pixelScale: number,
): void {
  if (style.particles <= 0) return;
  const count = ParticleStream.countFor(frame.particles, frame.particleCount);
  if (count === 0) return;
  res.particles.upload(frame.particles, count);

  const p = res.programs.particles;
  p.use();
  p.f1("u_gain", particleGain(style, count));
  p.f1("u_size", particleSize(pixelScale, count));
  p.f1("u_intensity", intensity);

  res.scene.bind();
  res.particles.draw(count);
}

/** Grade, fringe, vignette and dither, straight to the default framebuffer. */
export function drawCompositePass(
  gl: WebGL2RenderingContext,
  res: Resources,
  frame: RenderFrame,
  style: ModeStyle,
  intensity: number,
  frameIndex: number,
  canvasW: number,
  canvasH: number,
): void {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvasW, canvasH);

  const p = res.programs.composite;
  p.use();
  p.tex("u_scene", 0, res.scene.texture);
  p.tex("u_bloom", 1, res.bloom.output);
  p.tex("u_noise", 2, res.sources.noiseTex);
  // Motion drives the glow, not the exposure: pushing exposure with movement
  // makes the whole frame pump, while pushing bloom makes the bright parts bloom
  // harder, which is what "a burst of movement blazes" should feel like.
  p.f1("u_bloomAmount", bloomAmount(style, intensity));
  p.f1("u_exposure", exposureAmount(style, intensity));
  p.f1("u_aberration", style.aberration);
  p.f1("u_vignette", style.vignette);
  p.f1("u_grade", style.grade);
  p.f1("u_frame", frameIndex);
  // The engine's origin is in the dye grid's convention (y down); this pass
  // samples the scene target, which holds that image flipped, so the y has to be
  // flipped with it or the rush would dolly in on the mirror of the palm.
  const rush = rushFocus(frame.rush, true);
  p.f2("u_rushAt", rush.x, rush.y);
  p.f1("u_rushProgress", rush.progress);
  p.f1("u_rushPower", rush.power);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
