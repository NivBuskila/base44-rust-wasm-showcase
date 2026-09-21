/**
 * Every program the WebGL2 frame uses, linked once per context.
 *
 * The mirror of `gpu/pipelines.ts`: built from shader source and the probed
 * `ShaderEnv` alone, so `renderer.ts` reads as a list of passes. These die with
 * the context, which is why they are created and disposed as one group.
 */

import { Program } from "./gl";
import { FULLSCREEN_VERT, buildShader } from "./shaders/common";
import type { ShaderEnv } from "./shaders/common";
import { BLIT_FRAG, COMPOSITE_FRAG } from "./shaders/composite";
import { OVERLAY_FRAG, OVERLAY_VERT } from "./shaders/overlay";
import { PARTICLE_FRAG, PARTICLE_VERT } from "./shaders/particles";
import { SCENE_FRAG } from "./shaders/scene";

export interface ScenePrograms {
  scene: Program;
  particles: Program;
  composite: Program;
  overlay: Program;
  blit: Program;
}

export function createPrograms(
  gl: WebGL2RenderingContext,
  env: ShaderEnv,
): ScenePrograms {
  const vert = buildShader(FULLSCREEN_VERT, env);
  const fullscreen = (frag: string, label: string): Program =>
    new Program(gl, vert, buildShader(frag, env), label);
  const program = (v: string, f: string, label: string): Program =>
    new Program(gl, buildShader(v, env), buildShader(f, env), label);

  return {
    scene: fullscreen(SCENE_FRAG, "scene"),
    particles: program(PARTICLE_VERT, PARTICLE_FRAG, "particles"),
    composite: fullscreen(COMPOSITE_FRAG, "composite"),
    overlay: program(OVERLAY_VERT, OVERLAY_FRAG, "overlay"),
    blit: fullscreen(BLIT_FRAG, "blit"),
  };
}

export function disposePrograms(programs: ScenePrograms): void {
  for (const p of Object.values(programs)) p.dispose();
}
