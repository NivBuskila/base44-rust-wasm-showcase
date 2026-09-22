/**
 * Every render pipeline the WebGPU frame uses, built once.
 *
 * Split out of `renderer.ts` by lifetime, like the rest of `gpu/`: these are
 * created at construction from shader source alone and never touched again, so
 * they carry no frame state and no resources. The renderer keeps the returned
 * record and asks each pipeline for its bind group layout when it (re)builds a
 * bind group.
 */

import { OVERLAY_STRIDE } from "../render/handmesh";
import { BLOOM_DOWN_WGSL, BLOOM_UP_WGSL } from "./shaders/bloom";
import { BLIT_WGSL, COMPOSITE_WGSL } from "./shaders/composite";
import { OVERLAY_LINES_WGSL, OVERLAY_POINTS_WGSL } from "./shaders/overlay";
import { PARTICLE_DRAW_WGSL } from "./shaders/particles";
import { SCENE_WGSL } from "./shaders/scene";

/** Intermediate HDR format; guaranteed renderable and blendable in WebGPU. */
export const HDR_FORMAT: GPUTextureFormat = "rgba16float";

/** Additive blend, shared by every light-emitting draw. */
const ADDITIVE: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
};

export interface GpuPipelines {
  scene: GPURenderPipeline;
  particleDraw: GPURenderPipeline;
  overlayLine: GPURenderPipeline;
  overlayPoint: GPURenderPipeline;
  down: GPURenderPipeline;
  up: GPURenderPipeline;
  composite: GPURenderPipeline;
  /** Frame texture to the canvas, in the preferred canvas format. */
  blit: GPURenderPipeline;
  /** Same blit, but into the `rgba8unorm` probe: the canvas is often BGRA. */
  probe: GPURenderPipeline;
}

/**
 * `canvasFormat` is the preferred canvas format; only `blit` uses it, which is
 * why the probe gets its own pipeline over the same shader.
 */
export function createPipelines(
  d: GPUDevice,
  canvasFormat: GPUTextureFormat,
): GpuPipelines {
  const module = (code: string, label: string): GPUShaderModule =>
    d.createShaderModule({ code, label });

  const fullscreen = (
    code: string,
    label: string,
    format: GPUTextureFormat,
    blend?: GPUBlendState,
  ): GPURenderPipeline => {
    const m = module(code, label);
    return d.createRenderPipeline({
      label,
      layout: "auto",
      vertex: { module: m, entryPoint: "vs_main" },
      fragment: {
        module: m,
        entryPoint: "fs_main",
        targets: [{ format, blend }],
      },
      primitive: { topology: "triangle-strip" },
    });
  };

  const drawModule = module(PARTICLE_DRAW_WGSL, "particles-draw");
  const particleDraw = d.createRenderPipeline({
    label: "particles-draw",
    layout: "auto",
    vertex: { module: drawModule, entryPoint: "vs_main" },
    fragment: {
      module: drawModule,
      entryPoint: "fs_main",
      targets: [{ format: HDR_FORMAT, blend: ADDITIVE }],
    },
    primitive: { topology: "triangle-list" },
  });

  const overlayAttributes: GPUVertexAttribute[] = [
    { shaderLocation: 0, offset: 0, format: "float32x2" },
    { shaderLocation: 1, offset: 8, format: "float32" },
  ];
  const overlayPipe = (
    code: string,
    label: string,
    topology: GPUPrimitiveTopology,
    stepMode: GPUVertexStepMode,
  ): GPURenderPipeline => {
    const m = module(code, label);
    return d.createRenderPipeline({
      label,
      layout: "auto",
      vertex: {
        module: m,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: OVERLAY_STRIDE * 4,
            attributes: overlayAttributes,
            stepMode,
          },
        ],
      },
      fragment: {
        module: m,
        entryPoint: "fs_main",
        targets: [{ format: HDR_FORMAT, blend: ADDITIVE }],
      },
      primitive: { topology },
    });
  };

  return {
    scene: fullscreen(SCENE_WGSL, "scene", HDR_FORMAT),
    particleDraw,
    overlayLine: overlayPipe(
      OVERLAY_LINES_WGSL,
      "overlay-lines",
      "line-list",
      "vertex",
    ),
    overlayPoint: overlayPipe(
      OVERLAY_POINTS_WGSL,
      "overlay-points",
      "triangle-list",
      "instance",
    ),
    down: fullscreen(BLOOM_DOWN_WGSL, "bloom-down", HDR_FORMAT),
    up: fullscreen(BLOOM_UP_WGSL, "bloom-up", HDR_FORMAT, ADDITIVE),
    composite: fullscreen(COMPOSITE_WGSL, "composite", "rgba8unorm"),
    blit: fullscreen(BLIT_WGSL, "blit", canvasFormat),
    probe: fullscreen(BLIT_WGSL, "probe-blit", "rgba8unorm"),
  };
}
