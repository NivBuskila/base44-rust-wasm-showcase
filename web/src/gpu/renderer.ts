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
 *    texture, which is then blitted to the canvas and periodically sampled by
 *    a 16x16 probe for `sampleLuminance`.
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
 * `sampleLuminance` (periodic) and the alive count lag behind: GPU readback is
 * asynchronous and stalling the queue for either would cost more than they are
 * worth.
 */

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
import { packCompositeUniform } from "./frame-uniforms";
import { FrameChain } from "./frame-chain";
import {
  MIN_SCENE_SCALE,
  bufferSize,
  deviceScale,
  scaledSize,
  sceneScale,
  viewportScale,
} from "../render/sizing";
import { HandOverlay } from "./overlay";
import { ParticlePass } from "./particle-pass";
import { ParticleSim } from "./particle-sim";
import type { GpuPipelines } from "./pipelines";
import { createPipelines } from "./pipelines";
import { ScenePass } from "./scene-pass";
import { COMPOSITE_UNIFORM_FLOATS } from "./shaders/composite";
import { SceneSources } from "./sources";
import type { FullscreenPass } from "./target";
import { clamp, finite } from "./sim-uniform";

export class GpuRendererError extends Error {}

export class GpuRenderer implements SceneRenderer {
  readonly backend: RenderBackend = "webgpu";

  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  readonly adapterLabel: string;

  /** Targets, post chain and the binds over them; see `frame-chain.ts`. */
  private readonly chain: FrameChain;

  /** Input textures and samplers; see `sources.ts`. */
  private readonly sources: SceneSources;
  /** The scene draw, its uniform and the debug blit; see `scene-pass.ts`. */
  private readonly scenePass: ScenePass;

  /** Every render pipeline; see `pipelines.ts`. */
  private readonly pipes: GpuPipelines;

  // --- uniforms and buffers
  private readonly compositeUniform: GPUBuffer;
  /** The hand skeleton and its buffers; see `overlay.ts`. */
  private readonly overlay: HandOverlay;

  /** The pool and its compute passes; see `particle-sim.ts`. */
  private readonly sim: ParticleSim;
  /** The instanced draw over the pool; see `particle-pass.ts`. */
  private readonly particles: ParticlePass;

  // --- scratch
  private readonly compositeScratch = new Float32Array(
    COMPOSITE_UNIFORM_FLOATS,
  );

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
    this.sources = new SceneSources(d);
    this.pipes = createPipelines(d, this.format);

    this.compositeUniform = d.createBuffer({
      label: "composite-uniform",
      size: COMPOSITE_UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.overlay = new HandOverlay(
      d,
      this.pipes.overlayLine,
      this.pipes.overlayPoint,
    );
    this.sim = new ParticleSim(d);
    this.particles = new ParticlePass(d, this.pipes, this.sim);
    this.scenePass = new ScenePass(d, this.pipes, this.sources);
    this.chain = new FrameChain(
      d,
      this.pipes,
      this.sources,
      this.compositeUniform,
    );

    window.addEventListener("resize", this.onResize);
    this.resize();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    window.removeEventListener("resize", this.onResize);
    this.chain.dispose();
    this.sources.dispose();
    this.overlay.dispose();
    this.sim.dispose();
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
    this.chain.resize(w, h, sw, sh);
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    this.sources.resetVideo();
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
      this.scenePass.recordDebug(
        encoder,
        this.pass,
        this.context.getCurrentTexture().createView(),
        frame,
      );
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    const style = STYLES[frame.mode];
    const drawn = frame.sim ? this.stepParticles(encoder, frame.sim, style) : 0;
    this.scenePass.record(
      encoder,
      this.pass,
      this.chain.scene.view!,
      frame,
      style,
      intensity,
      time,
      this.canvas.width,
      this.canvas.height,
      this.video,
    );
    if (drawn > 0)
      this.particles.record(
        encoder,
        this.chain.scene,
        drawn,
        style,
        intensity,
        frame.sim!,
        this.dpr * this.sceneScale * this.viewScale,
      );
    if (style.overlay > 0 && frame.hands)
      this.overlay.record(
        encoder,
        this.chain.scene,
        frame.hands,
        style.overlay,
        this.dpr * this.sceneScale,
      );
    this.chain.recordBloom(encoder, this.pass, style);
    this.drawComposite(encoder, frame, style, intensity);
    this.chain.recordBlit(
      encoder,
      this.pass,
      this.context.getCurrentTexture().createView(),
    );
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
    if (stepped === 0 || !this.particles.ensureBind()) return 0;
    return style.particles > 0 ? stepped : 0;
  }

  // ---------------------------------------------------------------- passes

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
    this.chain.recordComposite(encoder, this.pass);
  }

  // ------------------------------------------------------------ inspection

  /** Starts this frame's reads. Must run after the submit, never before it. */
  private readback(): void {
    this.sim.pollAlive();
    this.chain.pollProbe();
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  sampleLuminance(): number {
    return this.chain.luminance;
  }

  particlesAlive(): number {
    return this.sim.alive;
  }
}
