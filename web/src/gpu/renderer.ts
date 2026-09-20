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

import {
  FLUID_H,
  FLUID_W,
  MAX_PARTICLE_OPS,
  MAX_PARTICLES,
  PARTICLE_OP_STRIDE,
} from '../constants';
import { OVERLAY_CAPACITY, OVERLAY_STRIDE, buildHandMesh } from '../render/handmesh';
import { NOISE_SIZE, buildNoiseTile } from '../render/noise';
import type { ModeStyle } from '../render/styles';
import { STYLES } from '../render/styles';
import type { GpuSimFrame, RenderBackend, RenderFrame, SceneRenderer } from '../types';
import type { GpuContext } from './device';
import { BLOOM_DOWN_WGSL, BLOOM_UP_WGSL } from './shaders/bloom';
import { BLIT_WGSL, COMPOSITE_WGSL } from './shaders/composite';
import { OVERLAY_LINES_WGSL, OVERLAY_POINTS_WGSL } from './shaders/overlay';
import {
  DRAW_UNIFORM_FLOATS,
  PARTICLE_BYTES,
  PARTICLE_COMPUTE_WGSL,
  PARTICLE_DRAW_WGSL,
  SIM_UNIFORM_FLOATS,
  STEP_WORKGROUP,
} from './shaders/particles';
import { SCENE_WGSL } from './shaders/scene';

/** Intermediate HDR format; guaranteed renderable and blendable in WebGPU. */
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';

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

/** Pool allocations are rounded up to this, so a slider drag is not a realloc. */
const PARTICLE_BUCKET = 32768;

/** Mirrors `aether_core::particles`: heat decay and rise rates. */
const HEAT_DECAY = 0.85;
const HEAT_RISE = 3.5;
/** `aether_core::particles::MAX_STEP`. */
const MAX_SIM_STEP = 0.25;

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));
/** `aether_core::math::decay`. */
const decay = (rate: number, dt: number): number => Math.exp(-rate * dt);
const finite = (v: number, fallback: number): number => (Number.isFinite(v) ? v : fallback);

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
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | this.extraUsage,
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
  readonly backend: RenderBackend = 'webgpu';

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
  private readonly stepPipe: GPUComputePipeline;
  private readonly seedPipe: GPUComputePipeline;

  // --- uniforms and buffers
  private readonly sceneUniform: GPUBuffer;
  private readonly compositeUniform: GPUBuffer;
  private readonly drawUniform: GPUBuffer;
  private readonly simUniform: GPUBuffer;
  private readonly overlayLineUniform: GPUBuffer;
  private readonly overlayPointUniform: GPUBuffer;
  private readonly downUniforms: GPUBuffer[];
  private readonly upUniforms: GPUBuffer[];
  private readonly overlayBuffer: GPUBuffer;
  private readonly opsBuffer: GPUBuffer;
  private readonly velUBuffer: GPUBuffer;
  private readonly velVBuffer: GPUBuffer;
  private obstacleBuffer: GPUBuffer;
  private readonly counters: GPUBuffer;
  private readonly counterStaging: GPUBuffer;
  private readonly probeStaging: GPUBuffer;
  private particleBuffer: GPUBuffer | null = null;
  private particleCapacity = 0;
  private seededCount = 0;

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
  private stepBind: GPUBindGroup | null = null;
  private seedBind: GPUBindGroup | null = null;
  private drawBind: GPUBindGroup | null = null;

  // --- scratch
  private readonly sceneScratch = new Float32Array(24);
  private readonly compositeScratch = new Float32Array(12);
  private readonly drawScratch = new Float32Array(DRAW_UNIFORM_FLOATS);
  private readonly overlayScratchU = new Float32Array(8);
  private readonly bloomScratch = new Float32Array(4);
  private readonly simScratch = new ArrayBuffer(SIM_UNIFORM_FLOATS * 4);
  private readonly simF32 = new Float32Array(this.simScratch);
  private readonly simU32 = new Uint32Array(this.simScratch);
  private readonly overlayVerts = new Float32Array(OVERLAY_CAPACITY * OVERLAY_STRIDE);
  private readonly zeroCounters = new Uint32Array(2);

  // --- frame state
  private dpr = 1;
  private viewScale = 1;
  private sceneScale = 1;
  private qualityScale = 1;
  private frameIndex = 0;
  private video: HTMLVideoElement | null = null;
  private respawnCredit = 0;
  private alive = 0;
  private luminance = 0;
  private countersPending = false;
  private probePending = false;
  private disposed = false;

  private readonly onResize = (): void => this.resize();

  constructor(canvas: HTMLCanvasElement, ctx: GpuContext) {
    this.canvas = canvas;
    this.device = ctx.device;
    this.adapterLabel = ctx.label;
    const context = canvas.getContext('webgpu');
    if (!context) throw new GpuRendererError('could not get a webgpu canvas context');
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    const d = this.device;
    this.scene = new Target(d, HDR_FORMAT, 'scene');
    this.bloom = Array.from({ length: BLOOM_LEVELS }, (_, i) => new Target(d, HDR_FORMAT, `bloom${i}`));
    this.frameTarget = new Target(d, 'rgba8unorm', 'frame');
    this.probeTarget = new Target(d, 'rgba8unorm', 'probe', GPUTextureUsage.COPY_SRC);

    this.clampSampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.repeatSampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    this.nearestSampler = d.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    const grid = (label: string): GPUTexture =>
      d.createTexture({
        label,
        size: { width: FLUID_W, height: FLUID_H },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    this.dyeTex = grid('dye');
    this.debugTex = grid('debug');
    this.noiseTex = d.createTexture({
      label: 'noise',
      size: { width: NOISE_SIZE, height: NOISE_SIZE },
      format: 'rgba8unorm',
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
        layout: 'auto',
        vertex: { module: m, entryPoint: 'vs_main' },
        fragment: { module: m, entryPoint: 'fs_main', targets: [{ format, blend }] },
        primitive: { topology: 'triangle-strip' },
      });
    };
    const additive: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    };

    this.scenePipe = fullscreen(SCENE_WGSL, 'scene', HDR_FORMAT);
    this.downPipe = fullscreen(BLOOM_DOWN_WGSL, 'bloom-down', HDR_FORMAT);
    this.upPipe = fullscreen(BLOOM_UP_WGSL, 'bloom-up', HDR_FORMAT, additive);
    this.compositePipe = fullscreen(COMPOSITE_WGSL, 'composite', 'rgba8unorm');
    this.blitPipe = fullscreen(BLIT_WGSL, 'blit', this.format);

    const drawModule = module(PARTICLE_DRAW_WGSL, 'particles-draw');
    this.particleDrawPipe = d.createRenderPipeline({
      label: 'particles-draw',
      layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vs_main' },
      fragment: {
        module: drawModule,
        entryPoint: 'fs_main',
        targets: [{ format: HDR_FORMAT, blend: additive }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const overlayVertexLayout: GPUVertexBufferLayout[] = [
      {
        arrayStride: OVERLAY_STRIDE * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' },
          { shaderLocation: 1, offset: 8, format: 'float32' },
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
        layout: 'auto',
        vertex: {
          module: m,
          entryPoint: 'vs_main',
          buffers: [{ ...overlayVertexLayout[0], stepMode }],
        },
        fragment: {
          module: m,
          entryPoint: 'fs_main',
          targets: [{ format: HDR_FORMAT, blend: additive }],
        },
        primitive: { topology },
      });
    };
    this.overlayLinePipe = overlayPipe(OVERLAY_LINES_WGSL, 'overlay-lines', 'line-list', 'vertex');
    this.overlayPointPipe = overlayPipe(
      OVERLAY_POINTS_WGSL,
      'overlay-points',
      'triangle-list',
      'instance',
    );

    const computeModule = module(PARTICLE_COMPUTE_WGSL, 'particles-step');
    this.stepPipe = d.createComputePipeline({
      label: 'particles-step',
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'step_main' },
    });
    this.seedPipe = d.createComputePipeline({
      label: 'particles-seed',
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'seed_main' },
    });

    // --- buffers
    const uniform = (floats: number, label: string): GPUBuffer =>
      d.createBuffer({
        label,
        size: floats * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.sceneUniform = uniform(24, 'scene-uniform');
    this.compositeUniform = uniform(12, 'composite-uniform');
    this.drawUniform = uniform(DRAW_UNIFORM_FLOATS, 'draw-uniform');
    this.simUniform = uniform(SIM_UNIFORM_FLOATS, 'sim-uniform');
    // Two buffers, not one: `queue.writeBuffer` lands before the whole encoder
    // is submitted, so a second write would also change the first draw.
    this.overlayLineUniform = uniform(8, 'overlay-lines-uniform');
    this.overlayPointUniform = uniform(8, 'overlay-points-uniform');
    this.downUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) => uniform(4, `down${i}`));
    this.upUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) => uniform(4, `up${i}`));

    this.overlayBuffer = d.createBuffer({
      label: 'overlay',
      size: this.overlayVerts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const storage = (bytes: number, label: string): GPUBuffer =>
      d.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    this.opsBuffer = storage(MAX_PARTICLE_OPS * PARTICLE_OP_STRIDE * 4, 'ops');
    this.velUBuffer = storage(FLUID_W * FLUID_H * 4, 'vel-u');
    this.velVBuffer = storage(FLUID_W * FLUID_H * 4, 'vel-v');
    this.obstacleBuffer = storage(FLUID_W * FLUID_H * 4, 'obstacle');
    this.counters = d.createBuffer({
      label: 'counters',
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.counterStaging = d.createBuffer({
      label: 'counters-read',
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.probeStaging = d.createBuffer({
      label: 'probe-read',
      size: PROBE_ROW_BYTES * PROBE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.buildStaticBinds();
    window.addEventListener('resize', this.onResize);
    this.resize();
  }

  // ------------------------------------------------------------- resources

  private createVideoTexture(w: number, h: number): GPUTexture {
    this.videoTexW = w;
    this.videoTexH = h;
    return this.device.createTexture({
      label: 'video',
      size: { width: w, height: h },
      format: 'rgba8unorm',
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

  /** Bind groups over the pool; rebuilt whenever it is reallocated. */
  private ensurePool(active: number): boolean {
    const want = Math.min(MAX_PARTICLES, Math.max(1, active));
    if (want > this.particleCapacity) {
      const capacity = Math.min(
        MAX_PARTICLES,
        Math.ceil(want / PARTICLE_BUCKET) * PARTICLE_BUCKET,
      );
      this.particleBuffer?.destroy();
      this.particleBuffer = this.device.createBuffer({
        label: 'particles',
        size: capacity * PARTICLE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.particleCapacity = capacity;
      this.seededCount = 0;

      const pool = this.particleBuffer;
      const simEntries = (layout: GPUBindGroupLayout): GPUBindGroup =>
        this.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: this.simUniform } },
            { binding: 1, resource: { buffer: pool } },
            { binding: 2, resource: { buffer: this.velUBuffer } },
            { binding: 3, resource: { buffer: this.velVBuffer } },
            { binding: 4, resource: { buffer: this.obstacleBuffer } },
            { binding: 5, resource: { buffer: this.opsBuffer } },
            { binding: 6, resource: { buffer: this.counters } },
          ],
        });
      this.stepBind = simEntries(this.stepPipe.getBindGroupLayout(0));
      this.seedBind = simEntries(this.seedPipe.getBindGroupLayout(0));
      this.drawBind = this.device.createBindGroup({
        layout: this.particleDrawPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.drawUniform } },
          { binding: 1, resource: { buffer: pool } },
        ],
      });
    }
    return this.particleBuffer !== null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    window.removeEventListener('resize', this.onResize);
    this.scene.dispose();
    for (const level of this.bloom) level.dispose();
    this.frameTarget.dispose();
    this.probeTarget.dispose();
    for (const tex of [this.dyeTex, this.debugTex, this.noiseTex, this.videoTex]) tex.destroy();
    this.particleBuffer?.destroy();
    this.particleBuffer = null;
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
    this.viewScale = clamp(this.canvas.clientWidth / PARTICLE_SIZE_REF_WIDTH, 0.45, 1);

    const pixels = w * h;
    const budgeted = pixels > SCENE_PIXEL_BUDGET ? Math.sqrt(SCENE_PIXEL_BUDGET / pixels) : 1;
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
      if (uniformBuffer) entries.push({ binding: binding++, resource: { buffer: uniformBuffer } });
      for (const view of views) entries.push({ binding: binding++, resource: view });
      entries.push({ binding, resource: this.clampSampler });
      return d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    };

    this.downBinds = [];
    this.upBinds = [];
    for (let i = 0; i < this.bloomLevels; i++) {
      const src = i === 0 ? this.scene : this.bloom[i - 1];
      this.downBinds.push(sampled(this.downPipe, [src.view!], this.downUniforms[i]));
      this.upBinds.push(sampled(this.upPipe, [this.bloom[i].view!], this.upUniforms[i]));
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
      layout: this.blitPipe.getBindGroupLayout(0),
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
    if (!v || v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0) return false;
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

  /** Copies a WASM-memory view into a storage buffer, never past its size. */
  private uploadFloats(buffer: GPUBuffer, data: Float32Array): void {
    const floats = Math.min(data.length, buffer.size / 4);
    if (floats <= 0) return;
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, floats * 4);
  }

  // ----------------------------------------------------------------- frame

  render(frame: RenderFrame): void {
    if (this.disposed) return;
    this.resize();
    this.frameIndex = (this.frameIndex + 1) % 1024;
    const intensity = clamp(finite(frame.intensity, 0), 0, 1);
    const time = finite(frame.time, 0);

    const encoder = this.device.createCommandEncoder();

    if (frame.mode === 'debug') {
      this.drawDebug(encoder, frame);
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    const style = STYLES[frame.mode];
    const drawn = frame.sim ? this.stepParticles(encoder, frame.sim, style) : 0;
    this.drawScene(encoder, frame, style, intensity, time);
    if (drawn > 0) this.drawParticles(encoder, drawn, style, intensity, frame.sim!);
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
    load: GPULoadOp = 'clear',
  ): void {
    const p = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: load, storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    p.setPipeline(pipe);
    p.setBindGroup(0, bind);
    p.draw(4);
    p.end();
  }

  private drawDebug(encoder: GPUCommandEncoder, frame: RenderFrame): void {
    const view = this.context.getCurrentTexture().createView();
    const ok = frame.debug ? this.uploadGrid(this.debugTex, frame.debug) : false;
    if (!ok) {
      const p = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            loadOp: 'clear',
            storeOp: 'store',
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
   * Advances the pool: uploads the fluid, the obstacle and the op log, then one
   * compute pass per frame. Returns the particles that are worth drawing.
   */
  private stepParticles(encoder: GPUCommandEncoder, sim: GpuSimFrame, style: ModeStyle): number {
    const active = Math.min(MAX_PARTICLES, Math.max(0, Math.floor(sim.active)));
    if (active === 0 || sim.gridW < 2 || sim.gridH < 2) return 0;
    if (!this.ensurePool(active)) return 0;

    const cells = sim.gridW * sim.gridH;
    if (sim.velU.length < cells || sim.velV.length < cells) return 0;
    this.uploadFloats(this.velUBuffer, sim.velU);
    this.uploadFloats(this.velVBuffer, sim.velV);

    const ow = Math.max(1, Math.floor(sim.obstacleInfo[0] ?? 0));
    const oh = Math.max(1, Math.floor(sim.obstacleInfo[1] ?? 0));
    const maxAbs = sim.obstacleInfo[2] ?? 0;
    const usableObstacle = sim.obstacle.length >= ow * oh && maxAbs >= 0.5;
    if (usableObstacle) {
      const bytes = ow * oh * 4;
      if (this.obstacleBuffer.size < bytes) {
        this.obstacleBuffer.destroy();
        this.obstacleBuffer = this.device.createBuffer({
          label: 'obstacle',
          size: Math.ceil(bytes / 256) * 256,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // The bind groups hold the old buffer; force a rebuild.
        this.particleCapacity = 0;
        if (!this.ensurePool(active)) return 0;
      }
      this.uploadFloats(this.obstacleBuffer, sim.obstacle);
    }

    const opCount = Math.min(MAX_PARTICLE_OPS, Math.max(0, Math.floor(sim.opCount)));
    if (opCount > 0) {
      this.device.queue.writeBuffer(
        this.opsBuffer,
        0,
        sim.ops.buffer,
        sim.ops.byteOffset,
        opCount * PARTICLE_OP_STRIDE * 4,
      );
    }

    // Everything the Rust `Frame` folds through dt, folded the same way here.
    const dt = clamp(finite(sim.params[0], 0), 0, MAX_SIM_STEP);
    const maxx = sim.gridW - 1;
    const maxy = sim.gridH - 1;
    const life = clamp(finite(sim.params[1], 1), 0.05, 60);
    const drag = clamp(finite(sim.params[2], 1), 0, 8);
    const spawnRate = Math.max(0, finite(sim.params[3], 0));
    const damping = Math.max(0, finite(sim.params[4], 0));
    const swirl = finite(sim.params[5], 0) * dt;
    const gravity = finite(sim.params[6], 0) * dt;

    const perFrame = spawnRate * dt;
    const credit = Math.min(this.respawnCredit + perFrame, perFrame + 1);
    const budget = Math.floor(credit);
    this.respawnCredit = credit - budget;

    const f = this.simF32;
    const u = this.simU32;
    f[0] = maxx;
    f[1] = maxy;
    f[2] = (ow - 1) / Math.max(1, maxx);
    f[3] = (oh - 1) / Math.max(1, maxy);
    f[4] = dt;
    f[5] = decay(damping, dt);
    f[6] = decay(HEAT_DECAY, dt);
    f[7] = 1 - decay(HEAT_RISE, dt);
    f[8] = dt > 0 ? 1 / dt : 0;
    f[9] = swirl;
    f[10] = gravity;
    f[11] = drag;
    f[12] = life;
    f[13] = ow;
    f[14] = oh;
    f[15] = usableObstacle ? 1 : 0;
    u[16] = active;
    u[17] = opCount;
    u[18] = this.frameIndex;
    u[19] = budget;
    u[20] = sim.gridW;
    u[21] = sim.gridH;
    u[22] = 0;
    u[23] = 0;
    this.device.queue.writeBuffer(this.simUniform, 0, this.simScratch);
    this.device.queue.writeBuffer(this.counters, 0, this.zeroCounters);

    const groups = Math.ceil(active / STEP_WORKGROUP);
    const pass = encoder.beginComputePass({ label: 'particles' });
    if (this.seededCount < active) {
      // A fresh or grown pool: scatter it once, exactly as `seed_uniform` does.
      pass.setPipeline(this.seedPipe);
      pass.setBindGroup(0, this.seedBind!);
      pass.dispatchWorkgroups(groups);
      this.seededCount = active;
    }
    pass.setPipeline(this.stepPipe);
    pass.setBindGroup(0, this.stepBind!);
    pass.dispatchWorkgroups(groups);
    pass.end();

    encoder.copyBufferToBuffer(this.counters, 0, this.counterStaging, 0, 8);
    return style.particles > 0 ? active : 0;
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
      camUsed && (frame.mode === 'camera' || frame.mode === 'blend' || frame.showCamera);
    const v = frame.video ?? this.video;
    const hasVideo = wantCamera && this.uploadVideo(v);

    const canvasAspect = Math.max(1e-6, this.canvas.width / Math.max(1, this.canvas.height));
    const videoAspect =
      hasVideo && v && v.videoHeight > 0 ? v.videoWidth / v.videoHeight : canvasAspect;
    const camScaleX = videoAspect > canvasAspect ? canvasAspect / videoAspect : 1;
    const camScaleY = videoAspect > canvasAspect ? 1 : videoAspect / canvasAspect;

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
      label: 'particles',
      colorAttachments: [{ view: this.scene.view!, loadOp: 'load', storeOp: 'store' }],
    });
    p.setPipeline(this.particleDrawPipe);
    p.setBindGroup(0, this.drawBind!);
    p.draw(6, count);
    p.end();
  }

  private drawOverlay(encoder: GPUCommandEncoder, frame: RenderFrame, style: ModeStyle): void {
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
      label: 'overlay',
      colorAttachments: [{ view: this.scene.view!, loadOp: 'load', storeOp: 'store' }],
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
        label: 'overlay-joints',
        colorAttachments: [{ view: this.scene.view!, loadOp: 'load', storeOp: 'store' }],
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
      this.pass(encoder, this.bloom[i - 1].view!, this.upPipe, this.upBinds[i], 'load');
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
    const power = rush && rush.length >= 4 && Number.isFinite(rush[3]) ? Math.max(0, rush[3]) : 0;
    c[6] = power > 0 ? rush![2] : 0;
    c[7] = power;
    // The scene target already holds the image the way the screen shows it, so
    // the origin needs no y flip here — unlike the GL chain.
    c[8] = power > 0 ? rush![0] : 0.5;
    c[9] = power > 0 ? rush![1] : 0.5;
    c[10] = 0;
    c[11] = 0;
    this.device.queue.writeBuffer(this.compositeUniform, 0, c);
    this.pass(encoder, this.frameTarget.view!, this.compositePipe, this.compositeBind!);
  }

  /** Frame texture to the canvas, plus the 16x16 luminance probe. */
  private blit(encoder: GPUCommandEncoder): void {
    this.pass(
      encoder,
      this.context.getCurrentTexture().createView(),
      this.blitPipe,
      this.blitSceneBind!,
    );
    if (this.probePending) return;
    this.pass(encoder, this.probeTarget.view!, this.blitPipe, this.probeBind!);
    encoder.copyTextureToBuffer(
      { texture: this.probeTarget.texture! },
      { buffer: this.probeStaging, bytesPerRow: PROBE_ROW_BYTES, rowsPerImage: PROBE },
      { width: PROBE, height: PROBE },
    );
  }

  // ------------------------------------------------------------ inspection

  /** Maps this frame's counters and probe when the previous map has landed. */
  private readback(): void {
    if (!this.countersPending) {
      this.countersPending = true;
      void this.counterStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const v = new Uint32Array(this.counterStaging.getMappedRange().slice(0));
          this.counterStaging.unmap();
          this.alive = v[1];
        })
        .catch(() => undefined)
        .finally(() => {
          this.countersPending = false;
        });
    }
    if (!this.probePending) {
      this.probePending = true;
      void this.probeStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const px = new Uint8Array(this.probeStaging.getMappedRange().slice(0));
          this.probeStaging.unmap();
          let total = 0;
          for (let row = 0; row < PROBE; row++) {
            const base = row * PROBE_ROW_BYTES;
            for (let i = 0; i < PROBE; i++) {
              const p = base + i * 4;
              total += 0.2126 * px[p] + 0.7152 * px[p + 1] + 0.0722 * px[p + 2];
            }
          }
          this.luminance = total / (PROBE * PROBE) / 255;
        })
        .catch(() => undefined)
        .finally(() => {
          this.probePending = false;
        });
    }
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  sampleLuminance(): number {
    return this.luminance;
  }

  particlesAlive(): number {
    return this.alive;
  }
}
