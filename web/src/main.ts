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
import {
  DEFAULT_SPAWN_RATE,
  OVERDRIVE_PARTICLES,
  OVERDRIVE_SPAWN_RATE,
  OverdriveBanner,
} from './overdrive';
import { MediaPipePerception } from './perception';
import { Renderer } from './render/renderer';
import { PARTICLE_STRIDE, STAT, assertLayout } from './constants';
import type { PerceptionSource, PerceptionStatus, RenderFrame, ViewMode } from './types';

/** Engine seed; fixed so a session is reproducible given the same input. */
const SEED = 0xa37e5eed;

/** Floor on the interval between inferences, in ms. */
const PERCEPTION_MIN_INTERVAL_MS = 1000 / 30;

/**
 * Ceiling on the interval, in ms. Past this, gestures are too stale to feel
 * connected to the hand.
 *
 * Reaching it is not a stable operating point but a verdict: if the duty
 * target below cannot be met *within* this interval, inference is too
 * expensive for this device, full stop. Capping the interval and carrying on
 * would silently break the duty guarantee — measured at 61% of wall time on a
 * 1224 ms inference, which is what `PERCEPTION_GIVE_UP_STRIKES` exists to
 * catch.
 */
const PERCEPTION_MAX_INTERVAL_MS = 2000;

/**
 * Consecutive inferences too slow to fit the duty budget before perception is
 * switched off for the session.
 *
 * More than one because the first call includes MediaPipe's own warm-up and is
 * not representative, and a single slow frame should not cost the user hand
 * tracking for the rest of the session. When it does trip, the app keeps
 * working on the Rust optical-flow path and the HUD says why — which is a far
 * better outcome than a technically-alive gesture layer eating two thirds of
 * every frame.
 */
const PERCEPTION_GIVE_UP_STRIKES = 3;

/**
 * Share of wall-clock time inference is allowed to consume.
 *
 * `PerceptionSource.process` is synchronous — MediaPipe's video API has no
 * async form — so its cost lands directly in the frame it runs on. On a real
 * GPU that is 10-20 ms and invisible at a 30 Hz cadence. On software
 * rasterisation it measured **753 ms**, which at a fixed 30 Hz cadence means
 * every frame tries to run an inference that takes 22 frames, and the render
 * loop collapses to inference speed.
 *
 * So the cadence is derived from measured latency rather than assumed: spend at
 * most this fraction of the time in inference and leave the rest to the
 * simulation. Aether stays responsive through the Rust optical-flow path, which
 * is exactly the case that path exists for; gestures just update more slowly.
 */
const PERCEPTION_DUTY = 0.3;

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
  private readonly overdrive: OverdriveBanner;
  /** Particle count to return to when overdrive is switched off. */
  private particlesBeforeOverdrive = 0;
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
  /** Current inference cadence, adapted from measured cost. */
  private perceptionIntervalMs = PERCEPTION_MIN_INTERVAL_MS;
  /** Smoothed wall time one `process` call costs the render loop. */
  private inferenceCostMs = 0;
  /** Consecutive inferences too slow to fit the duty budget. */
  private perceptionStrikes = 0;
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
      onOverdrive: (on) => this.setOverdrive(on),
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
    // After the HUD, which owns and rewrites #hud's markup.
    this.overdrive = new OverdriveBanner(document.getElementById('hud')!);
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

  /**
   * Brings the app up, then returns — without waiting for the vision models.
   *
   * Loading them means fetching a 12 MB WASM runtime plus 14 MB of model
   * bundles and letting MediaPipe warm up, which took ~25 s on software
   * rasterisation. Awaiting that before the first frame means the user watches
   * a boot screen for half a minute while a fully working simulation sits
   * behind it: optical flow, the fluid, particles and the ambient drive all
   * need nothing from MediaPipe. So the loop starts as soon as the camera is
   * up, and perception attaches itself whenever it is ready.
   */
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
      // `?perception=off` runs the app on the model-free path only. The headless
      // suite uses it for everything that is not specifically about MediaPipe:
      // on software rasterisation one inference costs ~770 ms, which would make
      // every unrelated assertion wait on a model it does not care about.
      const disabled = new URLSearchParams(location.search).get('perception') === 'off';
      if (disabled) {
        this.perceptionStatus = {
          kind: 'unavailable',
          reason: 'Disabled by ?perception=off — optical flow only.',
        };
      } else {
        this.perceptionStatus = { kind: 'loading' };
        void this.attachPerception();
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

    // Two different quantities, and conflating them is how a frame-rate readout
    // ends up unable to report the problem it exists to report. `simDt` is
    // clamped so one long stall does not blow the integrator apart; `realDt` is
    // what actually elapsed, and is the only honest input to an fps number.
    const realDt = Math.max(1e-4, (nowMs - this.lastFrameMs) / 1000);
    const simDt = Math.min(0.05, Math.max(1 / 480, realDt));
    this.lastFrameMs = nowMs;
    this.fps = this.fps * FPS_SMOOTHING + (1 / realDt) * (1 - FPS_SMOOTHING);
    this.frames++;

    if (this.views.buffer !== this.memory.buffer || this.views.dye.length === 0) {
      this.views = this.makeViews();
    }

    this.pumpCamera(nowMs);
    this.pumpPerception(nowMs);

    const stepStart = performance.now();
    this.engine.step(simDt);
    this.stepMs = performance.now() - stepStart;

    const stats = this.engine.stats();
    const renderStart = performance.now();
    this.renderer.render(this.buildFrame(stats));
    this.renderMs = performance.now() - renderStart;

    this.overdrive.update(stats, this.fps, this.stepMs);
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

  /** Runs inference at a cadence derived from its own measured cost. */
  private pumpPerception(nowMs: number): void {
    if (!this.perception || !this.cameraAvailable || !this.camera.hasFrame) return;
    if (nowMs - this.lastPerceptionMs < this.perceptionIntervalMs) return;

    const dt = this.lastPerceptionMs === 0 ? 1 / 30 : (nowMs - this.lastPerceptionMs) / 1000;
    this.lastPerceptionMs = nowMs;

    let result;
    const start = performance.now();
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

    // Measured here rather than trusting the source's own `latencyMs`: what
    // matters for pacing is the wall time this call cost *the render loop*,
    // including anything the source does around the inference itself.
    const cost = performance.now() - start;
    if (result) {
      this.inferenceCostMs = this.inferenceCostMs * 0.8 + cost * 0.2;
      const wanted = this.inferenceCostMs / PERCEPTION_DUTY;
      this.perceptionIntervalMs = Math.min(
        PERCEPTION_MAX_INTERVAL_MS,
        Math.max(PERCEPTION_MIN_INTERVAL_MS, wanted),
      );

      if (wanted > PERCEPTION_MAX_INTERVAL_MS) {
        this.perceptionStrikes++;
        if (this.perceptionStrikes >= PERCEPTION_GIVE_UP_STRIKES) {
          const ms = Math.round(this.inferenceCostMs);
          const reason =
            `Inference takes ${ms} ms on this device — too slow to run without ` +
            `stalling the simulation. Running on motion detection only.`;
          console.warn(`[aether] ${reason}`);
          this.perception.close();
          this.perception = null;
          this.perceptionStatus = { kind: 'unavailable', reason };
          this.lastHands = null;
          this.engine.clear_perception();
          return;
        }
      } else {
        this.perceptionStrikes = 0;
      }
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
   * Loads the vision models off the critical path and attaches them.
   *
   * Never rejects: a failure here is a capability downgrade, not an error. The
   * Rust optical-flow path keeps driving the fluid, so the app stays fully
   * responsive and the HUD explains what is missing.
   */
  private async attachPerception(): Promise<void> {
    const perception = new MediaPipePerception();
    try {
      await perception.init();
      // A scripted source may have been injected while the models loaded
      // (the headless suite does exactly this), and it must win — otherwise a
      // slow load silently overwrites the test's perception mid-run.
      if (this.perception !== null) {
        perception.close();
        return;
      }
      this.perception = perception;
      this.perceptionStatus = perception.status;
    } catch (err) {
      const reason = `Vision models unavailable: ${String(err)}`;
      console.warn(`[aether] ${reason}`);
      this.perceptionStatus = { kind: 'unavailable', reason };
      perception.close();
    }
  }

  /**
   * Replaces the perception source at runtime. The headless suite uses this to
   * feed scripted landmarks, which is what makes the gesture pipeline testable
   * without a real hand in front of a real camera. Passing null detaches
   * perception entirely, leaving the model-free optical-flow path in charge.
   */
  setPerception(source: PerceptionSource | null): void {
    this.perception?.close();
    this.perception = source;
    this.perceptionStatus = source?.status ?? { kind: 'unavailable', reason: 'detached' };
    this.engine.clear_perception();
    this.lastHands = null;
    this.lastPerceptionMs = 0;
    // A scripted source has nothing to do with the real one's cost, so the
    // adaptive cadence has to start over or a slow MediaPipe load would leave
    // an injected test source throttled to once every two seconds.
    this.perceptionIntervalMs = PERCEPTION_MIN_INTERVAL_MS;
    this.inferenceCostMs = 0;
    this.perceptionStrikes = 0;
  }

  /**
   * A cheap snapshot, safe to poll every frame.
   *
   * Luminance is deliberately NOT in here. It reads back the framebuffer, and
   * with `preserveDrawingBuffer` that forces a pipeline flush — which on
   * software rasterisation cost ~250 ms. Playwright's `waitForFunction` polls
   * on every animation frame, so including it made the frame-rate test destroy
   * the frame rate it was measuring: it read 3.9 fps where the app was really
   * running at 16.5. Instrumentation that perturbs what it measures is worse
   * than none. Call `luminance()` explicitly when you want it.
   */
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
      particleCount: this.engine.particle_count(),
      mode: this.mode,
      /** Effective inference cadence in Hz, after adaptive throttling. */
      perceptionHz: 1000 / this.perceptionIntervalMs,
      inferenceCostMs: this.inferenceCostMs,
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

  /**
   * Overdrive: the full 1M pool plus a spawn rate that fills it. Off restores
   * the pool size the user had and the default spawn rate. Reallocation
   * detaches views, hence the rebuild.
   */
  setOverdrive(on: boolean): void {
    if (on === this.overdrive.active) return;
    if (on) {
      this.particlesBeforeOverdrive = this.engine.particle_count();
      this.engine.set_particle_count(OVERDRIVE_PARTICLES);
      this.engine.set_param('spawn_rate', OVERDRIVE_SPAWN_RATE);
    } else {
      this.engine.set_particle_count(this.particlesBeforeOverdrive);
      this.engine.set_param('spawn_rate', DEFAULT_SPAWN_RATE);
    }
    this.views = this.makeViews();
    this.overdrive.setActive(on);
  }

  /**
   * Mean luminance of the last rendered frame. Expensive — see `diagnostics`.
   */
  luminance(): number {
    return this.renderer.sampleLuminance();
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
  /** Framebuffer readback; do not poll this on every frame. */
  luminance(): number;
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
      luminance: () => app.luminance(),
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
