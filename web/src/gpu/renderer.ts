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
import {
  OVERLAY_CAPACITY,
  OVERLAY_STRIDE,
  buildHandMesh,
} from "../render/handmesh";
import { NOISE_SIZE, buildNoiseTile } from "../render/noise";
import type { ModeStyle } from "../render/styles";
import { STYLES } from "../render/styles";
import type {
  GpuSimFrame,
  RenderBackend,
  RenderFrame,
  SceneRenderer,
} from "../types";
import type { GpuContext } from "./device";
import { recordGpuError } from "./error-log";
import { ParticleSim } from "./particle-sim";
import { Readback } from "./readback";
import {
  BLOOM_DOWN_WGSL,
  BLOOM_UNIFORM_FLOATS,
  BLOOM_UP_WGSL,
} from "./shaders/bloom";
import {
  BLIT_WGSL,
  COMPOSITE_UNIFORM_FLOATS,
  COMPOSITE_WGSL,
} from "./shaders/composite";
import {
  OVERLAY_LINES_WGSL,
  OVERLAY_POINTS_WGSL,
  OVERLAY_UNIFORM_FLOATS,
} from "./shaders/overlay";
import { DRAW_UNIFORM_FLOATS, PARTICLE_DRAW_WGSL } from "./shaders/particles";
import { SCENE_UNIFORM_FLOATS, SCENE_WGSL } from "./shaders/scene";
import { clamp, finite } from "./sim-uniform";

/** Intermediate HDR format; guaranteed renderable and blendable in WebGPU. */
const HDR_FORMAT: GPUTextureFormat = "rgba16float";

const MAX_DPR = 2;
const BLOOM_LEVELS = 5;
const BLOOM_TENT = 1.1;
const SCENE_PIXEL_BUDGET = 5_200_000;
const MIN_SCENE_SCALE = 0.4;
const PARTICLE_SIZE_REF_WIDTH = 1200;

/** Side of the luminance probe; a 16x16 reduction of the finished frame. */
const PROBE = 16;
/** `copyTextureToBuffer` wants rows aligned to 256 bytes. */
const PROBE_ROW_BYTES = 256;

/** Mean Rec.709 luminance of the probe readback, `[0, 1]`, rows 256-aligned. */
function meanLuminance(px: Uint8Array): number {
  let total = 0;
  for (let row = 0; row < PROBE; row++) {
    const base = row * PROBE_ROW_BYTES;
    for (let i = 0; i < PROBE; i++) {
      const p = base + i * 4;
      total += 0.2126 * px[p] + 0.7152 * px[p + 1] + 0.0722 * px[p + 2];
    }
  }
  return total / (PROBE * PROBE) / 255;
}

export class GpuRendererError extends Error {}

/** A colour target plus its size, resized in place. */
class Target {
  texture: GPUTexture | null = null;
  view: GPUTextureView | null = null;
  width = 0;
  height = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly format: GPUTextureFormat,
    private readonly label: string,
    private readonly extraUsage: number = 0,
  ) {}

  resize(w: number, h: number): boolean {
    if (this.width === w && this.height === h && this.texture) return false;
    this.texture?.destroy();
    this.texture = this.device.createTexture({
      label: this.label,
      size: { width: w, height: h },
      format: this.format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        this.extraUsage,
    });
    this.view = this.texture.createView();
    this.width = w;
    this.height = h;
    return true;
  }

  dispose(): void {
    this.texture?.destroy();
    this.texture = null;
    this.view = null;
    this.width = 0;
    this.height = 0;
  }
}

export class GpuRenderer implements SceneRenderer {
  readonly backend: RenderBackend = "webgpu";

  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  readonly adapterLabel: string;

  // --- targets
  private readonly scene: Target;
  private readonly bloom: Target[];
  private readonly frameTarget: Target;
  private readonly probeTarget: Target;
  private bloomLevels = 1;

  // --- textures and samplers
  private dyeTex: GPUTexture;
  private debugTex: GPUTexture;
  private noiseTex: GPUTexture;
  private videoTex: GPUTexture;
  private videoTexW = 0;
  private videoTexH = 0;
  private videoTime = -1;
  private readonly clampSampler: GPUSampler;
  private readonly repeatSampler: GPUSampler;
  private readonly nearestSampler: GPUSampler;

  // --- pipelines
  private readonly scenePipe: GPURenderPipeline;
  private readonly particleDrawPipe: GPURenderPipeline;
  private readonly overlayLinePipe: GPURenderPipeline;
  private readonly overlayPointPipe: GPURenderPipeline;
  private readonly downPipe: GPURenderPipeline;
  private readonly upPipe: GPURenderPipeline;
  private readonly compositePipe: GPURenderPipeline;
  private readonly blitPipe: GPURenderPipeline;
  /** Same blit, but into the `rgba8unorm` probe: the canvas is often BGRA. */
  private readonly probePipe: GPURenderPipeline;

  // --- uniforms and buffers
  private readonly sceneUniform: GPUBuffer;
  private readonly compositeUniform: GPUBuffer;
  private readonly drawUniform: GPUBuffer;
  private readonly overlayLineUniform: GPUBuffer;
  private readonly overlayPointUniform: GPUBuffer;
  private readonly downUniforms: GPUBuffer[];
  private readonly upUniforms: GPUBuffer[];
  private readonly overlayBuffer: GPUBuffer;

  /** The pool and its compute passes; see `particle-sim.ts`. */
  private readonly sim: ParticleSim;
  /** Pool generation the draw bind group was built against. */
  private drawnPoolVersion = -1;
  private readonly probeRead: Readback;

  // --- bind groups (rebuilt when the resources behind them change)
  private sceneBind: GPUBindGroup | null = null;
  private compositeBind: GPUBindGroup | null = null;
  private blitSceneBind: GPUBindGroup | null = null;
  private blitDebugBind: GPUBindGroup | null = null;
  private probeBind: GPUBindGroup | null = null;
  private downBinds: GPUBindGroup[] = [];
  private upBinds: GPUBindGroup[] = [];
  private overlayLineBind: GPUBindGroup | null = null;
  private overlayPointBind: GPUBindGroup | null = null;
  private drawBind: GPUBindGroup | null = null;

  // --- scratch
  private readonly sceneScratch = new Float32Array(24);
  private readonly compositeScratch = new Float32Array(12);
  private readonly drawScratch = new Float32Array(DRAW_UNIFORM_FLOATS);
  private readonly overlayScratchU = new Float32Array(8);
  private readonly bloomScratch = new Float32Array(4);
  private readonly overlayVerts = new Float32Array(
    OVERLAY_CAPACITY * OVERLAY_STRIDE,
  );

  // --- frame state
  private dpr = 1;
  private viewScale = 1;
  private sceneScale = 1;
  private qualityScale = 1;
  private frameIndex = 0;
  private video: HTMLVideoElement | null = null;
  private luminance = 0;
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
    this.bloom = Array.from(
      { length: BLOOM_LEVELS },
      (_, i) => new Target(d, HDR_FORMAT, `bloom${i}`),
    );
    this.frameTarget = new Target(d, "rgba8unorm", "frame");
    this.probeTarget = new Target(
      d,
      "rgba8unorm",
      "probe",
      GPUTextureUsage.COPY_SRC,
    );

    this.clampSampler = d.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.repeatSampler = d.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
    this.nearestSampler = d.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });

    const grid = (label: string): GPUTexture =>
      d.createTexture({
        label,
        size: { width: FLUID_W, height: FLUID_H },
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    this.dyeTex = grid("dye");
    this.debugTex = grid("debug");
    this.noiseTex = d.createTexture({
      label: "noise",
      size: { width: NOISE_SIZE, height: NOISE_SIZE },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    d.queue.writeTexture(
      { texture: this.noiseTex },
      buildNoiseTile() as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: NOISE_SIZE * 4 },
      { width: NOISE_SIZE, height: NOISE_SIZE },
    );
    this.videoTex = this.createVideoTexture(2, 2);

    // --- pipelines
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
    const additive: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
    };

    this.scenePipe = fullscreen(SCENE_WGSL, "scene", HDR_FORMAT);
    this.downPipe = fullscreen(BLOOM_DOWN_WGSL, "bloom-down", HDR_FORMAT);
    this.upPipe = fullscreen(BLOOM_UP_WGSL, "bloom-up", HDR_FORMAT, additive);
    this.compositePipe = fullscreen(COMPOSITE_WGSL, "composite", "rgba8unorm");
    this.blitPipe = fullscreen(BLIT_WGSL, "blit", this.format);
    this.probePipe = fullscreen(BLIT_WGSL, "probe-blit", "rgba8unorm");

    const drawModule = module(PARTICLE_DRAW_WGSL, "particles-draw");
    this.particleDrawPipe = d.createRenderPipeline({
      label: "particles-draw",
      layout: "auto",
      vertex: { module: drawModule, entryPoint: "vs_main" },
      fragment: {
        module: drawModule,
        entryPoint: "fs_main",
        targets: [{ format: HDR_FORMAT, blend: additive }],
      },
      primitive: { topology: "triangle-list" },
    });

    const overlayVertexLayout: GPUVertexBufferLayout[] = [
      {
        arrayStride: OVERLAY_STRIDE * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x2" },
          { shaderLocation: 1, offset: 8, format: "float32" },
        ],
      },
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
          buffers: [{ ...overlayVertexLayout[0], stepMode }],
        },
        fragment: {
          module: m,
          entryPoint: "fs_main",
          targets: [{ format: HDR_FORMAT, blend: additive }],
        },
        primitive: { topology },
      });
    };
    this.overlayLinePipe = overlayPipe(
      OVERLAY_LINES_WGSL,
      "overlay-lines",
      "line-list",
      "vertex",
    );
    this.overlayPointPipe = overlayPipe(
      OVERLAY_POINTS_WGSL,
      "overlay-points",
      "triangle-list",
      "instance",
    );

    // --- buffers
    const uniform = (floats: number, label: string): GPUBuffer =>
      d.createBuffer({
        label,
        size: floats * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.sceneUniform = uniform(SCENE_UNIFORM_FLOATS, "scene-uniform");
    this.compositeUniform = uniform(COMPOSITE_UNIFORM_FLOATS, "composite-uniform");
    this.drawUniform = uniform(DRAW_UNIFORM_FLOATS, "draw-uniform");
    // Two buffers, not one: `queue.writeBuffer` lands before the whole encoder
    // is submitted, so a second write would also change the first draw.
    this.overlayLineUniform = uniform(OVERLAY_UNIFORM_FLOATS, "overlay-lines-uniform");
    this.overlayPointUniform = uniform(OVERLAY_UNIFORM_FLOATS, "overlay-points-uniform");
    this.downUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) =>
      uniform(BLOOM_UNIFORM_FLOATS, `down${i}`),
    );
    this.upUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) =>
      uniform(BLOOM_UNIFORM_FLOATS, `up${i}`),
    );

    this.overlayBuffer = d.createBuffer({
      label: "overlay",
      size: this.overlayVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.sim = new ParticleSim(d);
    this.probeRead = new Readback(d, "probe-read", PROBE_ROW_BYTES * PROBE);

    this.buildStaticBinds();
    window.addEventListener("resize", this.onResize);
    this.resize();
  }

  // ------------------------------------------------------------- resources

  private createVideoTexture(w: number, h: number): GPUTexture {
    this.videoTexW = w;
    this.videoTexH = h;
    return this.device.createTexture({
      label: "video",
      size: { width: w, height: h },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  /** Bind groups over resources that outlive a resize. */
  private buildStaticBinds(): void {
    const d = this.device;
    this.sceneBind = d.createBindGroup({
      layout: this.scenePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.sceneUniform } },
        { binding: 1, resource: this.dyeTex.createView() },
        { binding: 2, resource: this.videoTex.createView() },
        { binding: 3, resource: this.noiseTex.createView() },
        { binding: 4, resource: this.clampSampler },
        { binding: 5, resource: this.repeatSampler },
      ],
    });
    this.blitDebugBind = d.createBindGroup({
      layout: this.blitPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.debugTex.createView() },
        { binding: 1, resource: this.nearestSampler },
      ],
    });
    this.overlayLineBind = d.createBindGroup({
      layout: this.overlayLinePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.overlayLineUniform } }],
    });
    this.overlayPointBind = d.createBindGroup({
      layout: this.overlayPointPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.overlayPointUniform } }],
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
        layout: this.particleDrawPipe.getBindGroupLayout(0),
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
    for (const level of this.bloom) level.dispose();
    this.frameTarget.dispose();
    this.probeTarget.dispose();
    for (const tex of [
      this.dyeTex,
      this.debugTex,
      this.noiseTex,
      this.videoTex,
    ])
      tex.destroy();
    this.sim.dispose();
    this.probeRead.dispose();
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
    this.dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * this.dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.viewScale = clamp(
      this.canvas.clientWidth / PARTICLE_SIZE_REF_WIDTH,
      0.45,
      1,
    );

    const pixels = w * h;
    const budgeted =
      pixels > SCENE_PIXEL_BUDGET ? Math.sqrt(SCENE_PIXEL_BUDGET / pixels) : 1;
    this.sceneScale = clamp(budgeted * this.qualityScale, MIN_SCENE_SCALE, 1);
    const sw = Math.max(1, Math.round(w * this.sceneScale));
    const sh = Math.max(1, Math.round(h * this.sceneScale));

    let dirty = this.scene.resize(sw, sh);
    let lw = sw;
    let lh = sh;
    let levels = 0;
    while (levels < BLOOM_LEVELS) {
      const nw = Math.max(1, lw >> 1);
      const nh = Math.max(1, lh >> 1);
      // Level 0 always exists — the composite samples it — even when the
      // canvas has not been laid out yet and is only a few pixels wide.
      if (levels > 0 && (nw < 8 || nh < 8)) break;
      dirty = this.bloom[levels].resize(nw, nh) || dirty;
      lw = nw;
      lh = nh;
      levels++;
    }
    this.bloomLevels = Math.max(1, levels);
    dirty = this.frameTarget.resize(w, h) || dirty;
    dirty = this.probeTarget.resize(PROBE, PROBE) || dirty;
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
      entries.push({ binding, resource: this.clampSampler });
      return d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    };

    this.downBinds = [];
    this.upBinds = [];
    for (let i = 0; i < this.bloomLevels; i++) {
      const src = i === 0 ? this.scene : this.bloom[i - 1];
      this.downBinds.push(
        sampled(this.downPipe, [src.view!], this.downUniforms[i]),
      );
      this.upBinds.push(
        sampled(this.upPipe, [this.bloom[i].view!], this.upUniforms[i]),
      );
    }
    this.compositeBind = sampled(
      this.compositePipe,
      [this.scene.view!, this.bloom[0].view!, this.noiseTex.createView()],
      this.compositeUniform,
    );
    this.blitSceneBind = d.createBindGroup({
      layout: this.blitPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.frameTarget.view! },
        { binding: 1, resource: this.clampSampler },
      ],
    });
    this.probeBind = d.createBindGroup({
      layout: this.probePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.frameTarget.view! },
        { binding: 1, resource: this.clampSampler },
      ],
    });
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    this.videoTime = -1;
  }

  // --------------------------------------------------------------- uploads

  /** True when the video texture holds a usable frame. */
  private uploadVideo(v: HTMLVideoElement | null): boolean {
    if (!v || v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0)
      return false;
    if (this.videoTexW !== v.videoWidth || this.videoTexH !== v.videoHeight) {
      this.videoTex.destroy();
      this.videoTex = this.createVideoTexture(v.videoWidth, v.videoHeight);
      this.videoTime = -1;
      this.buildStaticBinds();
    }
    if (v.currentTime !== this.videoTime) {
      try {
        this.device.queue.copyExternalImageToTexture(
          { source: v },
          { texture: this.videoTex },
          { width: v.videoWidth, height: v.videoHeight },
        );
      } catch {
        // Not decodable yet, or a stream the copy refuses; keep the last frame.
        return this.videoTime >= 0;
      }
      this.videoTime = v.currentTime;
    }
    return true;
  }

  private uploadGrid(tex: GPUTexture, data: Uint8Array): boolean {
    const bytes = FLUID_W * FLUID_H * 4;
    if (data.length < bytes) return false;
    this.device.queue.writeTexture(
      { texture: tex },
      // A view over WASM memory, which may be a SharedArrayBuffer.
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: FLUID_W * 4, rowsPerImage: FLUID_H },
      { width: FLUID_W, height: FLUID_H },
    );
    return true;
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
    this.drawOverlay(encoder, frame, style);
    this.drawBloom(encoder, style);
    this.drawComposite(encoder, frame, style, intensity);
    this.blit(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.readback();
  }

  /** One render pass over a target, drawing the fullscreen strip. */
  private pass(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    pipe: GPURenderPipeline,
    bind: GPUBindGroup,
    load: GPULoadOp = "clear",
  ): void {
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
  }

  private drawDebug(encoder: GPUCommandEncoder, frame: RenderFrame): void {
    const view = this.context.getCurrentTexture().createView();
    const ok = frame.debug
      ? this.uploadGrid(this.debugTex, frame.debug)
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
    this.pass(encoder, view, this.blitPipe, this.blitDebugBind!);
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
    this.uploadGrid(this.dyeTex, frame.dye);
    const camUsed =
      style.camTint[0] + style.camTint[1] + style.camTint[2] > 0 ||
      style.camEdge[0] + style.camEdge[1] + style.camEdge[2] > 0;
    const wantCamera =
      camUsed &&
      (frame.mode === "camera" || frame.mode === "blend" || frame.showCamera);
    const v = frame.video ?? this.video;
    const hasVideo = wantCamera && this.uploadVideo(v);

    const canvasAspect = Math.max(
      1e-6,
      this.canvas.width / Math.max(1, this.canvas.height),
    );
    const videoAspect =
      hasVideo && v && v.videoHeight > 0
        ? v.videoWidth / v.videoHeight
        : canvasAspect;
    const camScaleX =
      videoAspect > canvasAspect ? canvasAspect / videoAspect : 1;
    const camScaleY =
      videoAspect > canvasAspect ? 1 : videoAspect / canvasAspect;

    const s = this.sceneScratch;
    s[0] = FLUID_W;
    s[1] = FLUID_H;
    s[2] = 1 / FLUID_W;
    s[3] = 1 / FLUID_H;
    s[4] = camScaleX;
    s[5] = camScaleY;
    s[6] = 1 / Math.max(1, this.videoTexW);
    s[7] = 1 / Math.max(1, this.videoTexH);
    s[8] = style.camTint[0];
    s[9] = style.camTint[1];
    s[10] = style.camTint[2];
    s[11] = style.camToe;
    s[12] = style.camEdge[0];
    s[13] = style.camEdge[1];
    s[14] = style.camEdge[2];
    s[15] = style.camRaw;
    s[16] = style.dye;
    s[17] = style.bg;
    s[18] = hasVideo ? 1 : 0;
    s[19] = intensity;
    s[20] = time;
    s[21] = 0;
    s[22] = 0;
    s[23] = 0;
    this.device.queue.writeBuffer(this.sceneUniform, 0, s);
    this.pass(encoder, this.scene.view!, this.scenePipe, this.sceneBind!);
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
    d[0] = px * (3.1 + (1.35 - 3.1) * clamp(count / 200_000, 0, 1));
    // Same energy normalisation as the WebGL2 path: total emitted light stays
    // roughly constant as the pool is resized.
    d[1] = style.particles * clamp(90_000 / count, 0.1, 2.0);
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
    p.setPipeline(this.particleDrawPipe);
    p.setBindGroup(0, this.drawBind!);
    p.draw(6, count);
    p.end();
  }

  private drawOverlay(
    encoder: GPUCommandEncoder,
    frame: RenderFrame,
    style: ModeStyle,
  ): void {
    if (style.overlay <= 0 || !frame.hands) return;
    const mesh = buildHandMesh(frame.hands, this.overlayVerts);
    const vertices = mesh.lineVertices + mesh.pointVertices;
    if (vertices === 0) return;
    this.device.queue.writeBuffer(
      this.overlayBuffer,
      0,
      this.overlayVerts,
      0,
      vertices * OVERLAY_STRIDE,
    );

    const u = this.overlayScratchU;
    u[0] = style.overlay;
    u[2] = this.scene.width;
    u[3] = this.scene.height;
    u[4] = 0.55;
    u[5] = 0.88;
    u[6] = 1.0;

    const p = encoder.beginRenderPass({
      label: "overlay",
      colorAttachments: [
        { view: this.scene.view!, loadOp: "load", storeOp: "store" },
      ],
    });
    if (mesh.lineVertices > 0) {
      u[1] = 1;
      u[7] = 0;
      this.device.queue.writeBuffer(this.overlayLineUniform, 0, u);
      p.setPipeline(this.overlayLinePipe);
      p.setBindGroup(0, this.overlayLineBind!);
      p.setVertexBuffer(0, this.overlayBuffer);
      p.draw(mesh.lineVertices);
    }
    if (mesh.pointVertices > 0) {
      // Joints need their own uniform write, so they go in a second pass: the
      // line draw above is already recorded against the previous values.
      p.end();
      u[1] = 5 * this.dpr * this.sceneScale;
      u[7] = 1;
      this.device.queue.writeBuffer(this.overlayPointUniform, 0, u);
      const q = encoder.beginRenderPass({
        label: "overlay-joints",
        colorAttachments: [
          { view: this.scene.view!, loadOp: "load", storeOp: "store" },
        ],
      });
      q.setPipeline(this.overlayPointPipe);
      q.setBindGroup(0, this.overlayPointBind!);
      q.setVertexBuffer(
        0,
        this.overlayBuffer,
        mesh.lineVertices * OVERLAY_STRIDE * 4,
        mesh.pointVertices * OVERLAY_STRIDE * 4,
      );
      q.draw(6, mesh.pointVertices);
      q.end();
      return;
    }
    p.end();
  }

  private drawBloom(encoder: GPUCommandEncoder, style: ModeStyle): void {
    const knee = Math.max(0.05, style.threshold * 0.7);
    for (let i = 0; i < this.bloomLevels; i++) {
      const src = i === 0 ? this.scene : this.bloom[i - 1];
      const b = this.bloomScratch;
      b[0] = 1 / src.width;
      b[1] = 1 / src.height;
      b[2] = i === 0 ? style.threshold : 0;
      b[3] = knee;
      this.device.queue.writeBuffer(this.downUniforms[i], 0, b);
      this.pass(encoder, this.bloom[i].view!, this.downPipe, this.downBinds[i]);
    }
    for (let i = this.bloomLevels - 1; i > 0; i--) {
      const src = this.bloom[i];
      const b = this.bloomScratch;
      b[0] = BLOOM_TENT / src.width;
      b[1] = BLOOM_TENT / src.height;
      b[2] = 1;
      b[3] = 0;
      this.device.queue.writeBuffer(this.upUniforms[i], 0, b);
      this.pass(
        encoder,
        this.bloom[i - 1].view!,
        this.upPipe,
        this.upBinds[i],
        "load",
      );
    }
  }

  private drawComposite(
    encoder: GPUCommandEncoder,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
  ): void {
    const c = this.compositeScratch;
    c[0] = style.bloom * (0.72 + 0.75 * intensity);
    c[1] = style.exposure * (style.grade > 0 ? 0.96 + 0.22 * intensity : 1);
    c[2] = style.aberration;
    c[3] = style.vignette;
    c[4] = style.grade;
    c[5] = this.frameIndex;
    const rush = frame.rush;
    const power =
      rush && rush.length >= 4 && Number.isFinite(rush[3])
        ? Math.max(0, rush[3])
        : 0;
    c[6] = power > 0 ? rush![2] : 0;
    c[7] = power;
    // The scene target already holds the image the way the screen shows it, so
    // the origin needs no y flip here — unlike the GL chain.
    c[8] = power > 0 ? rush![0] : 0.5;
    c[9] = power > 0 ? rush![1] : 0.5;
    c[10] = 0;
    c[11] = 0;
    this.device.queue.writeBuffer(this.compositeUniform, 0, c);
    this.pass(
      encoder,
      this.frameTarget.view!,
      this.compositePipe,
      this.compositeBind!,
    );
  }

  /** Frame texture to the canvas, plus the 16x16 luminance probe. */
  private blit(encoder: GPUCommandEncoder): void {
    this.pass(
      encoder,
      this.context.getCurrentTexture().createView(),
      this.blitPipe,
      this.blitSceneBind!,
    );
    // Skipped while the read is in flight: a copy into a busy staging buffer
    // would cost the whole submission. See `readback.ts`.
    if (!this.probeRead.idle) return;
    this.pass(encoder, this.probeTarget.view!, this.probePipe, this.probeBind!);
    encoder.copyTextureToBuffer(
      { texture: this.probeTarget.texture! },
      {
        buffer: this.probeRead.buffer,
        bytesPerRow: PROBE_ROW_BYTES,
        rowsPerImage: PROBE,
      },
      { width: PROBE, height: PROBE },
    );
  }

  // ------------------------------------------------------------ inspection

  /** Starts this frame's reads. Must run after the submit, never before it. */
  private readback(): void {
    this.sim.pollAlive();
    this.probeRead.poll((bytes) => {
      this.luminance = meanLuminance(new Uint8Array(bytes));
    });
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  sampleLuminance(): number {
    return this.luminance;
  }

  particlesAlive(): number {
    return this.sim.alive;
  }
}
