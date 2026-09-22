/**
 * Everything that dies with the WebGL2 context, allocated and released as one
 * unit — split out of `renderer.ts` because a context restore has to rebuild
 * *all* of it: every wrapper still holding a dead GL object is stale, and a
 * restored context is often a different (frequently software) implementation, so
 * the formats are re-probed here rather than once in the constructor.
 */

import { BloomChain } from "./bloom";
import {
  RenderTarget,
  RendererError,
  isSoftwareRasteriser,
  probeHdr,
} from "./gl";
import { HandOverlay } from "./overlay";
import { ParticleStream } from "./particle-stream";
import type { ScenePrograms } from "./programs";
import { createPrograms, disposePrograms } from "./programs";
import type { ShaderEnv } from "./shaders/common";
import { SCENE_PIXEL_BUDGET, SOFTWARE_PIXEL_BUDGET } from "./sizing";
import { SceneSources } from "./sources";

export interface Resources {
  scene: RenderTarget;
  bloom: BloomChain;
  /** The input textures and their uploads; see `sources.ts`. */
  sources: SceneSources;
  /** Every linked program; see `programs.ts`. */
  programs: ScenePrograms;
  /** The pool's streamed vertex buffer; see `particle-stream.ts`. */
  particles: ParticleStream;
  /** The hand skeleton and its buffers; see `overlay.ts`. */
  overlay: HandOverlay;
  /** Empty VAO for the fullscreen passes, which fetch no attributes. */
  quadVao: WebGLVertexArrayObject;
  /** Scene pixel ceiling, tightened when the context is a CPU rasteriser. */
  pixelBudget: number;
}

/** Allocates the whole set against `gl`, probing its formats first. */
export function createResources(
  gl: WebGL2RenderingContext,
  forceSdr: boolean,
): Resources {
  const caps = probeHdr(gl, forceSdr);
  const env: ShaderEnv = { float: caps.float, range: caps.range };
  if (!caps.float) {
    console.info(
      "[aether] float render targets unavailable; HDR chain on 8-bit targets",
    );
  }

  const quadVao = gl.createVertexArray();
  if (!quadVao) throw new RendererError("could not allocate the vertex buffers");

  return {
    scene: new RenderTarget(gl, caps.format),
    bloom: new BloomChain(gl, env, caps.format),
    sources: new SceneSources(gl),
    programs: createPrograms(gl, env),
    particles: new ParticleStream(gl),
    overlay: new HandOverlay(gl),
    quadVao,
    pixelBudget: isSoftwareRasteriser(gl)
      ? SOFTWARE_PIXEL_BUDGET
      : SCENE_PIXEL_BUDGET,
  };
}

export function releaseResources(
  gl: WebGL2RenderingContext,
  res: Resources,
): void {
  res.scene.dispose();
  res.bloom.dispose();
  res.sources.dispose();
  res.particles.dispose();
  res.overlay.dispose();
  gl.deleteVertexArray(res.quadVao);
  disposePrograms(res.programs);
}
