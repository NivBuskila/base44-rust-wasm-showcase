/**
 * Boot and the frame loop.
 *
 * Three clocks run at different rates and must not be conflated:
 * - **render** (~60 Hz): `engine.step` + draw.
 * - **camera** (~30 Hz): luma readback for optical flow, only when the video
 *   has actually advanced.
 * - **perception** (~30 Hz, best effort): MediaPipe inference, which can take
 *   longer than a frame and must never block the loop.
 *
 * Everything degrades rather than fails. No camera, no models, or a WebGL2
 * context loss each drop one capability and leave the rest running — the
 * engine's ambient mode keeps the screen alive with nothing connected at all.
 */

import './styles.css';

import init, { AetherEngine } from './wasm/aether';
import { Camera, CameraError } from './camera';
import { Hud } from './hud';
import { MediaPipePerception } from './perception';
import { Renderer } from './render/renderer';
import { PARTICLE_STRIDE, STAT, assertLayout } from './constants';
import type { PerceptionSource, PerceptionStatus, RenderFrame, ViewMode } from './types';

/** Engine seed; fixed so a session is reproducible given the same input. */
const SEED = 0xa37e5eed;

/** Don't run inference more often than this, in ms. */
const PERCEPTION_INTERVAL_MS = 1000 / 30;

/** Rolling window for the fps readout. */
const FPS_SMOOTHING = 0.9;

interface Views {
  buffer: ArrayBufferLike;
  dye: Uint8Array;
  luma: Uint8Array;
  mask: Float32Array;
}

class App {
  private readonly engine: AetherEngine;
  private readonly memory: WebAssembly.Memory;
  private readonly renderer: Renderer;
  private readonly hud: Hud;
  private readonly camera = new Camera();

  private perception: PerceptionSource | null = null;
  private perceptionStatus: PerceptionStatus = { kind: 'loading' };
  /**
   * The most recent packed hand buffer, kept for the renderer's landmark
   * overlay. Held by reference — perception reuses its own arrays, so this
   * tracks the live values rather than a snapshot, which is what the overlay
   * wants anyway.
   */
  private lastHands: Float32Array | null = null;

  private views: Views;

  private running = false;
  private rafId = 0;
  private lastFrameMs = 0;
  private lastCameraTime = -1;
  private lastCameraMs = 0;
  private lastPerceptionMs = 0;
  private cameraAvailable = false;

  private mode: ViewMode = 'aether';
  private showCamera = true;

  private fps = 0;
  private stepMs = 0;
  private renderMs = 0;
  private inferenceMs = 0;
  private frames = 0;

  constructor(engine: AetherEngine, memory: WebAssembly.Memory, renderer: Renderer) {
    this.engine = engine;
    this.memory = memory;
    this.renderer = renderer;
    this.views = this.makeViews();
    this.hud = new Hud(document.getElementById('hud')!, {
      onParam: (key, value) => {
        if (!this.engine.set_param(key, value)) {
          console.warn(`[aether] unknown engine parameter: ${key}`);
        }
      },
      onParticleCount: (n) => {
        // Resizing the pool can reallocate, which detaches every view.
        this.engine.set_particle_count(n);
        this.views = this.makeViews();
      },
      onViewMode: (mode) => {
        this.mode = mode;
      },
      onToggleCamera: (on) => {
        this.showCamera = on;
      },
      onReset: () => {
        this.engine.reset();
        this.views = this.makeViews();
      },
    });
  }

  /**
   * Rebuilds the typed-array views over WASM linear memory.
   *
   * Views detach whenever the memory grows, and the pointers themselves move
   * when the engine reallocates, so this is called at boot, after any
   * reallocating engine call, and whenever the loop notices a new buffer.
   */
  private makeViews(): Views {
    const buffer = this.memory.buffer;
    return {
      buffer,
      dye: new Uint8Array(buffer, this.engine.dye_ptr(), this.engine.dye_len()),
      luma: new Uint8Array(buffer, this.engine.luma_ptr(), this.engine.luma_len()),
      mask: new Float32Array(buffer, this.engine.mask_ptr(), this.engine.mask_capacity()),
    };
  }

  /** A fresh particle view; its length tracks the live particle count. */
  private particleView(): Float32Array {
    return new Float32Array(
      this.views.buffer,
      this.engine.particle_ptr(),
      this.engine.particle_count() * PARTICLE_STRIDE,
    );
  }

  async start(setStatus: (text: string) => void): Promise<void> {
    setStatus('opening the camera…');
    try {
      await this.camera.start();
      this.cameraAvailable = true;
    } catch (err) {
      this.cameraAvailable = false;
      const message =
        err instanceof CameraError
          ? err.denied
            ? 'Camera denied — running in ambient mode.'
            : 'No camera found — running in ambient mode.'
          : `Camera failed: ${String(err)}`;
      console.warn(`[aether] ${message}`);
      this.perceptionStatus = { kind: 'unavailable', reason: message };
    }

    if (this.cameraAvailable) {
      setStatus('loading the vision models…');
      const perception = new MediaPipePerception();
      try {
        await perception.init();
        this.perception = perception;
        this.perceptionStatus = perception.status;
      } catch (err) {
        // Optical flow still drives the fluid, so this is a downgrade rather
        // than a failure. The HUD says so instead of the app dying.
        const reason = `Vision models unavailable: ${String(err)}`;
        console.warn(`[aether] ${reason}`);
        this.perceptionStatus = { kind: 'unavailable', reason };
      }
    }

    this.renderer.setVideo(this.cameraAvailable ? this.camera.video : null);
    this.running = true;
    this.lastFrameMs = performance.now();
    this.rafId = requestAnimationFrame(this.frame);
  }

  private readonly frame = (nowMs: number): void => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this.frame);

    const dt = Math.min(0.05, Math.max(1 / 480, (nowMs - this.lastFrameMs) / 1000));
    this.lastFrameMs = nowMs;
    this.fps = this.fps * FPS_SMOOTHING + (1 / dt) * (1 - FPS_SMOOTHING);
    this.frames++;

    if (this.views.buffer !== this.memory.buffer || this.views.dye.length === 0) {
      this.views = this.makeViews();
    }

    this.pumpCamera(nowMs);
    this.pumpPerception(nowMs);

    const stepStart = performance.now();
    this.engine.step(dt);
    this.stepMs = performance.now() - stepStart;

    const stats = this.engine.stats();
    const renderStart = performance.now();
    this.renderer.render(this.buildFrame(stats));
    this.renderMs = performance.now() - renderStart;

    this.hud.update({
      fps: this.fps,
      stepMs: this.stepMs,
      renderMs: this.renderMs,
      inferenceMs: this.inferenceMs,
      stats,
      spells: [this.engine.spell_name(0), this.engine.spell_name(1)],
      perception: this.perceptionStatus,
    });
  };

  /** Feeds the luma plane, but only when the camera produced a new frame. */
  private pumpCamera(nowMs: number): void {
    if (!this.cameraAvailable || !this.camera.hasFrame) return;
    const t = this.camera.video.currentTime;
    if (t === this.lastCameraTime) return;

    const luma = this.camera.readLuma(true);
    if (!luma) return;
    this.views.luma.set(luma);

    const dt = this.lastCameraMs === 0 ? 1 / 30 : (nowMs - this.lastCameraMs) / 1000;
    this.engine.push_luma(dt);
    this.lastCameraTime = t;
    this.lastCameraMs = nowMs;
  }

  /** Runs inference at most every `PERCEPTION_INTERVAL_MS`. */
  private pumpPerception(nowMs: number): void {
    if (!this.perception || !this.cameraAvailable || !this.camera.hasFrame) return;
    if (nowMs - this.lastPerceptionMs < PERCEPTION_INTERVAL_MS) return;

    const dt = this.lastPerceptionMs === 0 ? 1 / 30 : (nowMs - this.lastPerceptionMs) / 1000;
    this.lastPerceptionMs = nowMs;

    let result;
    try {
      result = this.perception.process(this.camera.video, nowMs);
    } catch (err) {
      // One bad inference must not kill the loop; drop perception instead.
      console.error('[aether] perception threw, disabling it', err);
      this.perceptionStatus = { kind: 'unavailable', reason: 'Inference failed.' };
      this.perception.close();
      this.perception = null;
      return;
    }
    if (!result) return;

    this.inferenceMs = result.latencyMs;
    this.lastHands = result.hands;
    this.engine.push_hands(result.hands, dt);
    this.engine.push_pose(result.pose, dt);

    if (result.mask) {
      const { data, width, height } = result.mask;
      if (width * height <= this.views.mask.length) {
        this.views.mask.set(data.subarray(0, width * height));
        this.engine.push_mask(width, height, true, dt);
      }
    }
  }

  private buildFrame(stats: Float32Array): RenderFrame {
    // Motion energy is unbounded; a soft knee maps it into the [0, 1] the
    // renderer wants without clipping the common case to full brightness.
    const motion = stats[STAT.MOTION_ENERGY] ?? 0;
    const intensity = Math.min(1, Math.sqrt(Math.max(0, motion)) * 0.35);

    return {
      dye: this.views.dye,
      particles: this.particleView(),
      particleCount: this.engine.particle_count(),
      debug: this.mode === 'debug' ? this.readDebug() : null,
      video: this.cameraAvailable ? this.camera.video : null,
      hands: stats[STAT.HANDS_PRESENT] > 0 ? this.lastHands : null,
      time: this.frames / 60,
      intensity,
      mode: this.mode,
      showCamera: this.showCamera && this.cameraAvailable,
    };
  }

  private readDebug(): Uint8Array {
    // debug_ptr() re-encodes the texture, so the view is built after the call.
    const ptr = this.engine.debug_ptr();
    return new Uint8Array(this.memory.buffer, ptr, this.engine.dye_len());
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this.perception?.close();
    this.camera.stop();
  }

  // ------------------------------------------------------------ test hooks

  /**
   * Replaces the perception source at runtime. The headless test uses this to
   * feed scripted landmarks, which is what makes the gesture pipeline testable
   * without a real hand in front of a real camera.
   */
  setPerception(source: PerceptionSource | null): void {
    this.perception?.close();
    this.perception = source;
    this.perceptionStatus = source?.status ?? { kind: 'unavailable', reason: 'detached' };
    this.engine.clear_perception();
    this.lastHands = null;
    this.lastPerceptionMs = 0;
  }

  get diagnostics() {
    return {
      frames: this.frames,
      fps: this.fps,
      stepMs: this.stepMs,
      renderMs: this.renderMs,
      inferenceMs: this.inferenceMs,
      cameraAvailable: this.cameraAvailable,
      perception: this.perceptionStatus,
      stats: Array.from(this.engine.stats()),
      spells: [this.engine.spell_name(0), this.engine.spell_name(1)] as [string, string],
      luminance: this.renderer.sampleLuminance(),
      particleCount: this.engine.particle_count(),
      mode: this.mode,
    };
  }

  get rawEngine(): AetherEngine {
    return this.engine;
  }

  forceMode(mode: ViewMode): void {
    this.mode = mode;
  }

  /**
   * Resizes the particle pool through the same path the HUD uses. Exposed to
   * tests because this is the one operation that reallocates engine buffers and
   * so detaches every typed-array view — the interesting thing to verify is
   * that the next frame still works.
   */
  setParticleCount(n: number): void {
    this.engine.set_particle_count(n);
    this.views = this.makeViews();
  }
}

/** Surface exposed on `window.__aether` for the Playwright suite. */
export interface AetherTestHooks {
  app: App;
  diagnostics(): App['diagnostics'];
  setPerception(source: PerceptionSource | null): void;
  setParam(key: string, value: number): boolean;
  setParticleCount(n: number): void;
  forceMode(mode: ViewMode): void;
  reset(): void;
}

declare global {
  interface Window {
    __aether?: AetherTestHooks;
  }
}

async function boot(): Promise<void> {
  const bootEl = document.getElementById('boot');
  const statusEl = document.getElementById('boot-status');
  const setStatus = (text: string) => {
    if (statusEl) statusEl.textContent = text;
  };

  const fail = (message: string, err?: unknown) => {
    console.error(`[aether] ${message}`, err);
    if (statusEl) statusEl.textContent = message;
    bootEl?.classList.add('failed');
  };

  try {
    setStatus('loading the engine…');
    const wasm = await init();
    const engine = new AetherEngine(SEED);
    assertLayout(engine.layout());

    const canvas = document.getElementById('stage');
    if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#stage canvas missing');

    setStatus('starting the renderer…');
    const renderer = new Renderer(canvas);

    const app = new App(engine, wasm.memory, renderer);
    window.__aether = {
      app,
      diagnostics: () => app.diagnostics,
      setPerception: (source) => app.setPerception(source),
      setParam: (key, value) => engine.set_param(key, value),
      setParticleCount: (n) => app.setParticleCount(n),
      forceMode: (mode) => app.forceMode(mode),
      reset: () => engine.reset(),
    };

    await app.start(setStatus);
    bootEl?.classList.add('done');
  } catch (err) {
    fail(
      err instanceof Error ? `Aether failed to start: ${err.message}` : 'Aether failed to start.',
      err,
    );
  }
}

void boot();
