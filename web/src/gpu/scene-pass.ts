/**
 * The WebGPU scene pass and the debug blit, split out of `renderer.ts`.
 *
 * `ScenePass` owns the one bind group over the input textures plus the scene
 * uniform buffer, so the camera-texture rule lives in a single place: a stream
 * whose resolution changed reallocates `SceneSources.videoTex`, bumps its
 * `version`, and only that may rebuild the bind groups over those views.
 *
 * The debug blit lives here for the same reason — it is the second (and only
 * other) bind group over a `SceneSources` texture, and nothing about it is part
 * of the graded chain.
 */

import { FLUID_H, FLUID_W } from "../constants";
import { aspectOf, cameraFit, usesCamera } from "../render/look";
import type { ModeStyle } from "../render/styles";
import type { RenderFrame } from "../types";
import { packSceneUniform } from "./frame-uniforms";
import type { GpuPipelines } from "./pipelines";
import { SCENE_UNIFORM_FLOATS } from "./shaders/scene";
import type { SceneSources } from "./sources";
import type { FullscreenPass } from "./target";

export class ScenePass {
  private readonly uniform: GPUBuffer;
  private readonly scratch = new Float32Array(SCENE_UNIFORM_FLOATS);
  private sceneBind: GPUBindGroup | null = null;
  private debugBind: GPUBindGroup | null = null;
  /** Camera-texture generation the bind groups were built against. */
  private boundVersion = -1;

  constructor(
    private readonly device: GPUDevice,
    private readonly pipes: GpuPipelines,
    private readonly sources: SceneSources,
  ) {
    this.uniform = device.createBuffer({
      label: "scene-uniform",
      size: SCENE_UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.buildBinds();
  }

  private buildBinds(): void {
    const d = this.device;
    const src = this.sources;
    this.boundVersion = src.version;
    this.sceneBind = d.createBindGroup({
      layout: this.pipes.scene.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: src.dyeTex.createView() },
        { binding: 2, resource: src.videoTex.createView() },
        { binding: 3, resource: src.noiseTex.createView() },
        { binding: 4, resource: src.clampSampler },
        { binding: 5, resource: src.repeatSampler },
      ],
    });
    this.debugBind = d.createBindGroup({
      layout: this.pipes.blit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.debugTex.createView() },
        { binding: 1, resource: src.nearestSampler },
      ],
    });
  }

  /**
   * Uploads this frame's dye and camera textures and records the scene pass
   * into `view`.
   *
   * `video` is the fallback stream the renderer holds; `frame.video` wins. The
   * camera layer is skipped outright in a mode whose style tints it to black —
   * running it would cost a full-frame upload to add exactly zero.
   */
  record(
    encoder: GPUCommandEncoder,
    pass: FullscreenPass,
    view: GPUTextureView,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
    time: number,
    canvasW: number,
    canvasH: number,
    video: HTMLVideoElement | null,
  ): void {
    const src = this.sources;
    src.uploadGrid(src.dyeTex, frame.dye);

    const wantCamera =
      usesCamera(style) &&
      (frame.mode === "camera" || frame.mode === "blend" || frame.showCamera);
    const v = frame.video ?? video;
    const hasVideo = wantCamera && this.uploadVideo(v);

    const [camScaleX, camScaleY] = cameraFit(
      aspectOf(canvasW, canvasH, 1),
      hasVideo && v ? aspectOf(v.videoWidth, v.videoHeight, 0) : 0,
    );

    packSceneUniform(this.scratch, {
      style,
      fluidW: FLUID_W,
      fluidH: FLUID_H,
      camScaleX,
      camScaleY,
      videoTexW: src.videoTexW,
      videoTexH: src.videoTexH,
      hasVideo,
      intensity,
      time,
    });
    this.device.queue.writeBuffer(this.uniform, 0, this.scratch);
    pass(encoder, view, this.pipes.scene, this.sceneBind!);
  }

  /** Rebuilds the binds when the stream's resolution moved the texture. */
  private uploadVideo(v: HTMLVideoElement | null): boolean {
    const ok = this.sources.uploadVideo(v);
    if (this.boundVersion !== this.sources.version) this.buildBinds();
    return ok;
  }

  /**
   * Raw obstacle/flow grid straight to `view`, ungraded. Clears to a flat field
   * when the engine has nothing to show, rather than leaving the last frame up.
   */
  recordDebug(
    encoder: GPUCommandEncoder,
    pass: FullscreenPass,
    view: GPUTextureView,
    frame: RenderFrame,
  ): void {
    const ok = frame.debug
      ? this.sources.uploadGrid(this.sources.debugTex, frame.debug)
      : false;
    if (!ok) {
      const p = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0.03, g: 0.035, b: 0.05, a: 1 },
          },
        ],
      });
      p.end();
      return;
    }
    pass(encoder, view, this.pipes.blit, this.debugBind!);
  }
}
