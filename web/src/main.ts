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

import type { AetherEngine } from './wasm/aether';
import { Camera, CameraError } from './camera';
import { ensureCrossOriginIsolation } from './cross-origin-isolation';
import { loadEngine, type EngineTier } from './engine-loader';
import { Hud } from './hud';
import {
  DEFAULT_SPAWN_RATE,
  OVERDRIVE_PARTICLES,
  OVERDRIVE_SPAWN_RATE,
  OverdriveBanner,
} from './overdrive';
import { PerformanceGovernor } from './performance-governor';
import { QaRecorder, clearQaSession, readQaSession, type QaSession } from './qa-recorder';
import { MediaPipePerception } from './perception';
import { WorkerPerception } from './perception-worker-client';
import { createRenderer } from './gpu';
import { PARTICLE_OP_STRIDE, PARTICLE_STRIDE, STAT, assertLayout, assertOpLayout } from './constants';
import { FLUID_H, FLUID_W } from './constants';
import type {
  GpuSimFrame,
  PerceptionSource,
  PerceptionStatus,
  RenderFrame,
  SceneRenderer,
  ViewMode,
} from './types';

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

/**
 * `Params::default().pressure_iters` from `crates/aether-core/src/config.rs`.
 * The solver's iteration count is not HUD-exposed, so this is what the adaptive
 * quality ladder treats as 100%.
 */
const DEFAULT_PRESSURE_ITERS = 28;

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
  /** The engine's gesture-sequence book. Static for the session. */
  private readonly comboBook: string;
  /** The engine's two-hand duet book. Static for the session. */
  private readonly duetBook: string;
  private readonly memory: WebAssembly.Memory;
  private readonly renderer: SceneRenderer;
  private readonly hud: Hud;
  /** Which engine build is running and on how many threads. */
  readonly tier: EngineTier;
  private readonly overdrive: OverdriveBanner;
  /** Adaptive quality; keeps the frame rate up on slower devices. */
  private readonly governor: PerformanceGovernor;
  /**
   * Records the session for later inspection. Only a real tab runs the loop, so
   * this is the only trace a QA pass leaves behind.
   */
  private readonly qa = new QaRecorder();
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
  /**
   * True when the source runs the models on a worker, so `process` is cheap and
   * self-pacing and the render loop should drain it every frame.
   */
  private perceptionOffThread = false;
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
  /** Camera pump: grabbing the video frame and handing it to perception. */
  private cameraMs = 0;
  /** HUD DOM update, which touches layout and is not free at 60 Hz. */
  private hudMs = 0;
  /** Everything the callback did, so the measured rows can be checked to sum. */
  private frameMs = 0;
  /**
   * The gap the callback did not spend: wall time between frames minus the work
   * of the previous one. `step`/`render`/`infer` summing far below the frame
   * period means the cost is here — GPU work the driver finishes after the GL
   * calls return, compositing, or another task blocking the loop — and without
   * this row that time is invisible and the HUD appears to contradict the frame
   * rate it sits next to.
   */
  private outsideMs = 0;
  private frames = 0;

  constructor(
    engine: AetherEngine,
    memory: WebAssembly.Memory,
    renderer: SceneRenderer,
    tier: EngineTier,
  ) {
    this.engine = engine;
    this.comboBook = engine.combo_book();
    this.duetBook = engine.duet_book();
    this.memory = memory;
    this.renderer = renderer;
    this.tier = tier;
    // The WebGPU backend simulates the pool itself: the engine stops advancing
    // its particles and starts logging what the spells did to them instead.
    if (renderer.backend === 'webgpu') {
      assertOpLayout(engine.particle_op_layout());
      engine.set_gpu_particles(true);
    }
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
        // The user's number is the new 100% for the quality ladder.
        this.governor.rebase(DEFAULT_PRESSURE_ITERS, n);
      },
      onOverdrive: (on) => this.setOverdrive(on),
      onViewMode: (mode) => {
        this.mode = mode;
      },
      onToggleCamera: (on) => {
        this.showCamera = on;
      },
      onPractice: (on) => this.engine.set_practice(on),
      onReset: () => {
        this.engine.reset();
        this.views = this.makeViews();
      },
    });
    this.governor = new PerformanceGovernor(
      {
        setPressureIters: (iters) => {
          this.engine.set_param('pressure_iters', iters);
        },
        setParticleCount: (count) => {
          this.engine.set_particle_count(count);
          this.views = this.makeViews();
        },
        setRenderScale: (scale) => this.renderer.setQualityScale(scale),
      },
      DEFAULT_PRESSURE_ITERS,
      engine.particle_count(),
    );
    // After the HUD, which owns and rewrites #hud's markup.
    this.overdrive = new OverdriveBanner(document.getElementById('hud')!, tier);
    this.hud.setEngineTier(tier);
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

    const frameStart = performance.now();
    this.outsideMs = Math.max(0, realDt * 1000 - this.frameMs);

    this.pumpCamera(nowMs);
    this.pumpPerception(nowMs);
    this.cameraMs = performance.now() - frameStart;

    const stepStart = performance.now();
    this.engine.step(simDt);
    this.stepMs = performance.now() - stepStart;

    const stats = this.engine.stats();
    const renderStart = performance.now();
    this.renderer.render(this.buildFrame(stats));
    this.renderMs = performance.now() - renderStart;

    // The GPU owns the pool, so its own count is the only true one. It lags a
    // frame behind the readback, which the HUD's stats row can live with.
    if (this.engine.gpu_particles()) {
      this.engine.set_particles_alive(this.renderer.particlesAlive?.() ?? 0);
    }

    this.governor.update(this.fps);
    this.overdrive.update(stats, this.fps, this.stepMs);
    const hudStart = performance.now();
    this.hud.update({
      fps: this.fps,
      stepMs: this.stepMs,
      renderMs: this.renderMs,
      inferenceMs: this.inferenceMs,
      stats,
      spells: [this.engine.spell_name(0), this.engine.spell_name(1)],
      comboBook: this.comboBook,
      comboProgress: this.engine.combo_progress(),
      duetBook: this.duetBook,
      duetProgress: this.engine.duet_progress(),
      hands: stats[STAT.HANDS_PRESENT] > 0 ? this.lastHands : null,
      perception: this.perceptionStatus,
    });
    this.hudMs = performance.now() - hudStart;
    this.frameMs = performance.now() - frameStart;

    // Last in the frame: the recorder reads the values this frame just produced,
    // and it rate-limits itself to 2 Hz internally.
    this.qa.sample(this.diagnostics);
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

    // Off-thread perception paces itself: the worker takes one frame at a time,
    // and `process` only decodes a bitmap and hands back whatever has already
    // come back. Throttling it here would do nothing but hold a finished result
    // for up to a full interval before the engine sees it — pure added latency,
    // which is what made gestures feel late. The interval gate stays for the
    // inline source, where every call really does run the models in this frame.
    if (
      !this.perceptionOffThread &&
      nowMs - this.lastPerceptionMs < this.perceptionIntervalMs
    ) {
      return;
    }

    // Time since the last result actually reached the engine — not since the
    // last attempt — because that is the interval the gesture velocities are
    // differentiated over.
    const dt = this.lastPerceptionMs === 0 ? 1 / 30 : (nowMs - this.lastPerceptionMs) / 1000;

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
    if (result && !this.perceptionOffThread) {
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
    this.lastPerceptionMs = nowMs;

    // `cost`, not `result.latencyMs`: the HUD row sits in the frame budget, so
    // it must report what the render loop actually paid. With perception on a
    // worker the model still takes ~26 ms, but the loop pays only the bitmap
    // decode — charging it the full inference would show a blown budget on a
    // frame that comfortably made 60 Hz.
    this.inferenceMs = cost;
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
      rush: this.engine.rush_state(),
      sim: this.engine.gpu_particles() ? this.gpuSim() : null,
    };
  }

  /**
   * Fluid, obstacle and op log for the GPU pool, as views over WASM memory.
   *
   * Rebuilt every frame on purpose: the pointers move whenever the engine
   * reallocates, and a retained view would silently read freed memory.
   */
  private gpuSim(): GpuSimFrame {
    const buffer = this.views.buffer;
    const cells = FLUID_W * FLUID_H;
    const info = this.engine.obstacle_info();
    const obstacleCells = Math.max(1, Math.floor(info[0])) * Math.max(1, Math.floor(info[1]));
    return {
      velU: new Float32Array(buffer, this.engine.velocity_u_ptr(), cells),
      velV: new Float32Array(buffer, this.engine.velocity_v_ptr(), cells),
      gridW: FLUID_W,
      gridH: FLUID_H,
      obstacle: new Float32Array(buffer, this.engine.obstacle_ptr(), obstacleCells),
      obstacleInfo: info,
      ops: new Float32Array(
        buffer,
        this.engine.particle_ops_ptr(),
        this.engine.particle_ops_len() * PARTICLE_OP_STRIDE,
      ),
      opCount: this.engine.particle_ops_len(),
      params: this.engine.particle_step_params(),
      active: this.engine.particle_count(),
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
  /**
   * Prefers perception on a worker, falling back to the inline source.
   *
   * Off-thread is strictly better — a 26 ms inference stops being a 26 ms hole
   * in the render loop — but it needs module workers and `createImageBitmap`,
   * and the worker can fail to start for reasons the page cannot inspect. So
   * the inline path stays as the fallback rather than the default: same models,
   * same results, just paid for out of the frame budget.
   */
  private async buildPerception(): Promise<PerceptionSource> {
    const offloaded = new WorkerPerception();
    try {
      await offloaded.init();
      return offloaded;
    } catch (err) {
      console.warn('[aether] perception worker unusable, running inference inline', err);
      offloaded.close();
      return new MediaPipePerception();
    }
  }

  private async attachPerception(): Promise<void> {
    const perception = await this.buildPerception();
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
      this.perceptionOffThread = perception instanceof WorkerPerception;
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
    this.perceptionOffThread = source instanceof WorkerPerception;
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
      /** Camera pump, inclusive of the bitmap decode charged to `inferenceMs`. */
      cameraMs: this.cameraMs,
      hudMs: this.hudMs,
      frameMs: this.frameMs,
      outsideMs: this.outsideMs,
      cameraAvailable: this.cameraAvailable,
      perception: this.perceptionStatus,
      stats: Array.from(this.engine.stats()),
      spells: [this.engine.spell_name(0), this.engine.spell_name(1)] as [string, string],
      particleCount: this.engine.particle_count(),
      mode: this.mode,
      engine: this.tier,
      /** Effective inference cadence in Hz, after adaptive throttling. */
      perceptionHz: 1000 / this.perceptionIntervalMs,
      inferenceCostMs: this.inferenceCostMs,
      /** Adaptive quality rung; 0 is full quality. */
      qualityTier: this.governor.level,
      /** Which graphics API drew the last frame, and who owns the pool. */
      renderBackend: this.renderer.backend,
    };
  }

  /** The session record so far, without waiting for the next storage write. */
  qaSession(): QaSession {
    return this.qa.summary();
  }

  /** Flushes the record immediately — e.g. before closing the tab. */
  flushQaSession(): void {
    this.qa.persist();
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
    // Overdrive is a ceiling demo: quality must not be pulled out from under it,
    // and switching back restores the pool the user had, which is the new base.
    this.governor.setPaused(on);
    if (!on) this.governor.rebase(DEFAULT_PRESSURE_ITERS, this.engine.particle_count());
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
  /** The live session record from this tab. */
  qaSession(): QaSession;
  /** The last session persisted on this origin, from any tab. */
  lastQaSession(): QaSession | null;
  /** Discards the persisted record. */
  clearQaSession(): void;
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
    // Must precede loadEngine: it decides the tier from `crossOriginIsolated`,
    // and this is what can still turn that true.
    await ensureCrossOriginIsolation();

    const loaded = await loadEngine(setStatus);
    const engine = new loaded.module.AetherEngine(SEED);
    assertLayout(engine.layout());

    const canvas = document.getElementById('stage');
    if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#stage canvas missing');

    setStatus('starting the renderer…');
    const { renderer } = await createRenderer(canvas);

    const app = new App(engine, loaded.memory, renderer, loaded.tier);
    window.__aether = {
      app,
      diagnostics: () => app.diagnostics,
      setPerception: (source) => app.setPerception(source),
      setParam: (key, value) => engine.set_param(key, value),
      setParticleCount: (n) => app.setParticleCount(n),
      forceMode: (mode) => app.forceMode(mode),
      luminance: () => app.luminance(),
      reset: () => engine.reset(),
      qaSession: () => app.qaSession(),
      lastQaSession: () => readQaSession(),
      clearQaSession,
    };

    // A tab is usually closed rather than idled out, and the 2 Hz writer may be
    // up to two seconds behind when that happens.
    window.addEventListener('pagehide', () => app.flushQaSession());

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
