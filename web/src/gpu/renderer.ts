/**
 * WebGPU backend: the same frame as `render/renderer.ts`, plus the particle
 * pool itself.
 *
 * One frame, in order:
 *
 * 1. **step** — a compute pass advances the pool, replaying the engine's op log
 *    first, so the particles never cross the WASM boundary at all.
 * 2. **scene** — treated camera behind the cubic-resampled dye field.
 * 3. **particles** — one instanced draw straight from the storage buffer.
 * 4. **overlay** — the hand skeleton (lines + instanced joint quads).
 * 5. **bloom** — 13-tap downsample chain, 9-tap tent fold-back.
 * 6. **composite** — grade, fringe, vignette, dither into an 8-bit frame
 *    texture, which is then blitted to the canvas and sampled down to a 16x16
 *    probe for `sampleLuminance`.
 *
 * Invariants, matching the WebGL2 chain:
 * - nothing is allocated per frame on the steady-state path; textures, targets
 *   and bind groups are rebuilt only from `resize`, from a camera resolution
 *   change or when the pool grows;
 * - the look comes from the shared `render/styles.ts` table, so the two
 *   backends grade a frame identically;
 * - the engine stays the source of truth: the fluid, the spells and the pool
 *   size are its, and the only thing coming back is the live particle count.
 *
 * `sampleLuminance` and the alive count are one frame behind: GPU readback is
 * asynchronous and stalling the queue for either would cost more than they are
 * worth.
 */

import { FLUID_H, FLUID_W } from "../constants";
import type { ModeStyle } from "../render/styles";
import { STYLES } from "../render/styles";
import type {
  GpuSimFrame,
  RenderBackend,
  RenderFrame,
  SceneRenderer,
} from "../types";
import { BloomChain } from "./bloom";
import type { GpuContext } from "./device";
import { recordGpuError } from "./error-log";
import { packCompositeUniform, packSceneUniform } from "./frame-uniforms";
import {
  aspectOf,
  cameraFit,
  particleGain,
  particleSize,
  usesCamera,
} from "../render/look";
import {
  MIN_SCENE_SCALE,
  bufferSize,
  deviceScale,
  scaledSize,
  sceneScale,
  viewportScale,
} from "../render/sizing";
import { HandOverlay } from "./overlay";
import { LuminanceProbe } from "./probe";
import { ParticleSim } from "./particle-sim";
import type { GpuPipelines } from "./pipelines";
import { HDR_FORMAT, createPipelines } from "./pipelines";
import { COMPOSITE_UNIFORM_FLOATS } from "./shaders/composite";
import { DRAW_UNIFORM_FLOATS } from "./shaders/particles";
import { SCENE_UNIFORM_FLOATS } from "./shaders/scene";
import { SceneSources } from "./sources";
import type { FullscreenPass } from "./target";
import { Target } from "./target";
import { clamp, finite } from "./sim-uniform";

export class GpuRendererError extends Error {}

export class GpuRenderer implements SceneRenderer {
  readonly backend: RenderBackend = "webgpu";

  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  readonly adapterLabel: string;

  // --- targets
  private readonly scene: Target;
  private readonly bloom: BloomChain;
  private readonly frameTarget: Target;
  private readonly probe: LuminanceProbe;

  /** Input textures and samplers; see `sources.ts`. */
  private readonly sources: SceneSources;
  /** Camera-texture generation the static bind groups were built against. */
  private boundSourceVersion = 0;

  /** Every render pipeline; see `pipelines.ts`. */
  private readonly pipes: GpuPipelines;

  // --- uniforms and buffers
  private readonly sceneUniform: GPUBuffer;
  private readonly compositeUniform: GPUBuffer;
  private readonly drawUniform: GPUBuffer;
  /** The hand skeleton and its buffers; see `overlay.ts`. */
  private readonly overlay: HandOverlay;

  /** The pool and its compute passes; see `particle-sim.ts`. */
  private readonly sim: ParticleSim;
  /** Pool generation the draw bind group was built against. */
  private drawnPoolVersion = -1;

  // --- bind groups (rebuilt when the resources behind them change)
  private sceneBind: GPUBindGroup | null = null;
  private compositeBind: GPUBindGroup | null = null;
  private blitSceneBind: GPUBindGroup | null = null;
  private blitDebugBind: GPUBindGroup | null = null;
  private drawBind: GPUBindGroup | null = null;

  // --- scratch
  private readonly sceneScratch = new Float32Array(SCENE_UNIFORM_FLOATS);
  private readonly compositeScratch = new Float32Array(
    COMPOSITE_UNIFORM_FLOATS,
  );
  private readonly drawScratch = new Float32Array(DRAW_UNIFORM_FLOATS);

  // --- frame state
  private dpr = 1;
  private viewScale = 1;
  private sceneScale = 1;
  private qualityScale = 1;
  private frameIndex = 0;
  private video: HTMLVideoElement | null = null;
  private disposed = false;

  private readonly onResize = (): void => this.resize();

  constructor(canvas: HTMLCanvasElement, ctx: GpuContext) {
    this.canvas = canvas;
    this.device = ctx.device;
    this.adapterLabel = ctx.label;
    const context = canvas.getContext("webgpu");
    if (!context)
      throw new GpuRendererError("could not get a webgpu canvas context");
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "opaque",
    });

    // A dropped command buffer is silent otherwise: the frame just never
    // arrives, which looks like flicker rather than an error.
    this.device.onuncapturederror = (event) => {
      const message = (event as GPUUncapturedErrorEvent).error.message;
      recordGpuError(message);
      console.error("[aether] WebGPU error", message);
    };

    const d = this.device;
    this.scene = new Target(d, HDR_FORMAT, "scene");
    this.frameTarget = new Target(d, "rgba8unorm", "frame");

    this.sources = new SceneSources(d);
    this.pipes = createPipelines(d, this.format);

    // --- buffers
    const uniform = (floats: number, label: string): GPUBuffer =>
      d.createBuffer({
        label,
        size: floats * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.sceneUniform = uniform(SCENE_UNIFORM_FLOATS, "scene-uniform");
    this.compositeUniform = uniform(
      COMPOSITE_UNIFORM_FLOATS,
      "composite-uniform",
    );
    this.drawUniform = uniform(DRAW_UNIFORM_FLOATS, "draw-uniform");
    this.overlay = new HandOverlay(
      d,
      this.pipes.overlayLine,
      this.pipes.overlayPoint,
    );
    this.sim = new ParticleSim(d);
    this.bloom = new BloomChain(
      d,
      this.pipes.down,
      this.pipes.up,
      this.sources.clampSampler,
      HDR_FORMAT,
    );
    this.probe = new LuminanceProbe(
      d,
      this.pipes.probe,
      this.sources.clampSampler,
    );

    this.buildStaticBinds();
    window.addEventListener("resize", this.onResize);
    this.resize();
  }

  // ------------------------------------------------------------- resources

  /** Bind groups over resources that outlive a resize. */
  private buildStaticBinds(): void {
    const d = this.device;
    const src = this.sources;
    this.boundSourceVersion = src.version;
    this.sceneBind = d.createBindGroup({
      layout: this.pipes.scene.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniform } },
        { binding: 1, resource: src.dyeTex.createView() },
        { binding: 2, resource: src.videoTex.createView() },
        { binding: 3, resource: src.noiseTex.createView() },
        { binding: 4, resource: src.clampSampler },
        { binding: 5, resource: src.repeatSampler },
      ],
    });
    this.blitDebugBind = d.createBindGroup({
      layout: this.pipes.blit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.debugTex.createView() },
        { binding: 1, resource: src.nearestSampler },
      ],
    });
  }

  /**
   * Keeps the instanced draw's bind group pointing at the live pool. The pool
   * belongs to `ParticleSim` and is reallocated when it grows, so the version it
   * reports — not a size comparison here — is what invalidates this.
   */
  private ensureDrawBind(): boolean {
    const pool = this.sim.pool;
    if (!pool) return false;
    if (this.drawnPoolVersion !== this.sim.poolVersion) {
      this.drawnPoolVersion = this.sim.poolVersion;
      this.drawBind = this.device.createBindGroup({
        layout: this.pipes.particleDraw.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.drawUniform } },
          { binding: 1, resource: { buffer: pool } },
        ],
      });
    }
    return this.drawBind !== null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    window.removeEventListener("resize", this.onResize);
    this.scene.dispose();
    this.bloom.dispose();
    this.frameTarget.dispose();
    this.sources.dispose();
    this.overlay.dispose();
    this.sim.dispose();
    this.probe.dispose();
    this.video = null;
  }

  // ------------------------------------------------------------------ size

  setQualityScale(scale: number): void {
    const next = clamp(scale, MIN_SCENE_SCALE, 1);
    if (next === this.qualityScale) return;
    this.qualityScale = next;
    this.resize();
  }

  /** Matches the canvas and every target to the CSS size and DPR. */
  resize(): void {
    if (this.disposed) return;
    this.dpr = deviceScale(window.devicePixelRatio);
    const [w, h] = bufferSize(
      this.canvas.clientWidth,
      this.canvas.clientHeight,
      this.dpr,
    );
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.viewScale = viewportScale(this.canvas.clientWidth);

    this.sceneScale = sceneScale(w, h, { quality: this.qualityScale });
    const [sw, sh] = scaledSize(w, h, this.sceneScale);

    let dirty = this.scene.resize(sw, sh);
    dirty = this.bloom.resize(sw, sh) || dirty;
    dirty = this.frameTarget.resize(w, h) || dirty;
    dirty = this.probe.resize() || dirty;
    if (dirty) this.buildTargetBinds();
  }

  /** Bind groups that reference a target view, so they die with a resize. */
  private buildTargetBinds(): void {
    const d = this.device;
    const sampled = (
      pipe: GPURenderPipeline,
      views: GPUTextureView[],
      uniformBuffer?: GPUBuffer,
    ): GPUBindGroup => {
      const entries: GPUBindGroupEntry[] = [];
      let binding = 0;
      if (uniformBuffer)
        entries.push({
          binding: binding++,
          resource: { buffer: uniformBuffer },
        });
      for (const view of views)
        entries.push({ binding: binding++, resource: view });
      entries.push({ binding, resource: this.sources.clampSampler });
      return d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    };

    this.bloom.buildBinds(this.scene);
    this.compositeBind = sampled(
      this.pipes.composite,
      [this.scene.view!, this.bloom.output, this.sources.noiseTex.createView()],
      this.compositeUniform,
    );
    this.blitSceneBind = d.createBindGroup({
      layout: this.pipes.blit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.frameTarget.view! },
        { binding: 1, resource: this.sources.clampSampler },
      ],
    });
    this.probe.buildBind(this.frameTarget.view!);
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    this.sources.resetVideo();
  }

  /**
   * Uploads this frame's camera texture, rebuilding the bind groups over it
   * when the stream's resolution changed the texture underneath them.
   */
  private uploadVideo(v: HTMLVideoElement | null): boolean {
    const ok = this.sources.uploadVideo(v);
    if (this.boundSourceVersion !== this.sources.version)
      this.buildStaticBinds();
    return ok;
  }

  // ----------------------------------------------------------------- frame

  render(frame: RenderFrame): void {
    if (this.disposed) return;
    this.resize();
    this.frameIndex = (this.frameIndex + 1) % 1024;
    const intensity = clamp(finite(frame.intensity, 0), 0, 1);
    const time = finite(frame.time, 0);

    const encoder = this.device.createCommandEncoder();

    if (frame.mode === "debug") {
      this.drawDebug(encoder, frame);
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    const style = STYLES[frame.mode];
    const drawn = frame.sim ? this.stepParticles(encoder, frame.sim, style) : 0;
    this.drawScene(encoder, frame, style, intensity, time);
    if (drawn > 0)
      this.drawParticles(encoder, drawn, style, intensity, frame.sim!);
    if (style.overlay > 0 && frame.hands)
      this.overlay.record(
        encoder,
        this.scene,
        frame.hands,
        style.overlay,
        this.dpr * this.sceneScale,
      );
    this.bloom.record(encoder, this.pass, style, this.scene);
    this.drawComposite(encoder, frame, style, intensity);
    this.blit(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.readback();
  }

  /** One render pass over a target, drawing the fullscreen strip. */
  private readonly pass: FullscreenPass = (
    encoder,
    view,
    pipe,
    bind,
    load = "clear",
  ): void => {
    const p = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          loadOp: load,
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    p.setPipeline(pipe);
    p.setBindGroup(0, bind);
    p.draw(4);
    p.end();
  };

  private drawDebug(encoder: GPUCommandEncoder, frame: RenderFrame): void {
    const view = this.context.getCurrentTexture().createView();
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
    this.pass(encoder, view, this.pipes.blit, this.blitDebugBind!);
  }

  // ------------------------------------------------------------ simulation

  /**
   * Steps the pool for this frame and returns how many particles to draw — zero
   * in a mode whose style emits no particle light, even though the simulation
   * still advanced, so the pool never freezes behind a mode switch.
   */
  private stepParticles(
    encoder: GPUCommandEncoder,
    sim: GpuSimFrame,
    style: ModeStyle,
  ): number {
    const stepped = this.sim.step(encoder, sim, this.frameIndex);
    if (stepped === 0 || !this.ensureDrawBind()) return 0;
    return style.particles > 0 ? stepped : 0;
  }

  // ---------------------------------------------------------------- passes

  private drawScene(
    encoder: GPUCommandEncoder,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
    time: number,
  ): void {
    this.sources.uploadGrid(this.sources.dyeTex, frame.dye);
    const wantCamera =
      usesCamera(style) &&
      (frame.mode === "camera" || frame.mode === "blend" || frame.showCamera);
    const v = frame.video ?? this.video;
    const hasVideo = wantCamera && this.uploadVideo(v);

    const canvasAspect = aspectOf(this.canvas.width, this.canvas.height, 1);
    const [camScaleX, camScaleY] = cameraFit(
      canvasAspect,
      hasVideo && v ? aspectOf(v.videoWidth, v.videoHeight, 0) : 0,
    );

    const s = this.sceneScratch;
    packSceneUniform(s, {
      style,
      fluidW: FLUID_W,
      fluidH: FLUID_H,
      camScaleX,
      camScaleY,
      videoTexW: this.sources.videoTexW,
      videoTexH: this.sources.videoTexH,
      hasVideo,
      intensity,
      time,
    });
    this.device.queue.writeBuffer(this.sceneUniform, 0, s);
    this.pass(encoder, this.scene.view!, this.pipes.scene, this.sceneBind!);
  }

  private drawParticles(
    encoder: GPUCommandEncoder,
    count: number,
    style: ModeStyle,
    intensity: number,
    sim: GpuSimFrame,
  ): void {
    const d = this.drawScratch;
    const px = this.dpr * this.sceneScale * this.viewScale;
    d[0] = particleSize(px, count);
    d[1] = particleGain(style, count);
    d[2] = intensity;
    d[3] = 0;
    d[4] = this.scene.width;
    d[5] = this.scene.height;
    d[6] = Math.max(1, sim.gridW - 1);
    d[7] = Math.max(1, sim.gridH - 1);
    this.device.queue.writeBuffer(this.drawUniform, 0, d);

    const p = encoder.beginRenderPass({
      label: "particles",
      colorAttachments: [
        { view: this.scene.view!, loadOp: "load", storeOp: "store" },
      ],
    });
    p.setPipeline(this.pipes.particleDraw);
    p.setBindGroup(0, this.drawBind!);
    p.draw(6, count);
    p.end();
  }

  private drawComposite(
    encoder: GPUCommandEncoder,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
  ): void {
    const c = this.compositeScratch;
    packCompositeUniform(c, {
      style,
      intensity,
      frameIndex: this.frameIndex,
      rush: frame.rush,
    });
    this.device.queue.writeBuffer(this.compositeUniform, 0, c);
    this.pass(
      encoder,
      this.frameTarget.view!,
      this.pipes.composite,
      this.compositeBind!,
    );
  }

  /** Frame texture to the canvas, plus the 16x16 luminance probe. */
  private blit(encoder: GPUCommandEncoder): void {
    this.pass(
      encoder,
      this.context.getCurrentTexture().createView(),
      this.pipes.blit,
      this.blitSceneBind!,
    );
    this.probe.record(encoder, this.pass);
  }

  // ------------------------------------------------------------ inspection

  /** Starts this frame's reads. Must run after the submit, never before it. */
  private readback(): void {
    this.sim.pollAlive();
    this.probe.poll();
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  sampleLuminance(): number {
    return this.probe.luminance;
  }

  particlesAlive(): number {
    return this.sim.alive;
  }
}
