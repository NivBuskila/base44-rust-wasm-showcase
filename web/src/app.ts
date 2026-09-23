/**
 * The app object and the frame loop.
 *
 * Three clocks run at different rates and must not be conflated:
 * - **render** (~60 Hz): `engine.step` + draw.
 * - **camera** (~30 Hz): luma readback for optical flow, only when the video
 *   has actually advanced.
 * - **perception** (~30 Hz, best effort): MediaPipe inference, which can take
 *   longer than a frame and must never block the loop (see `perception-pump.ts`).
 *
 * Everything degrades rather than fails. No camera, no models, or a WebGL2
 * context loss each drop one capability and leave the rest running — the
 * engine's ambient mode keeps the screen alive with nothing connected at all.
 */

import type { AetherEngine } from './wasm/aether';
import { Camera, CameraError } from './camera';
import { DEFAULT_PRESSURE_ITERS, EnginePool } from './engine-pool';
import { EngineViews } from './engine-views';
import { FrameClock } from './frame-clock';
import { FrameTimings } from './frame-timings';
import type { EngineTier } from './engine-loader';
import { Hud } from './hud';
import { OverdriveBanner } from './overdrive';
import { PerceptionPump } from './perception-pump';
import { PerformanceGovernor } from './performance-governor';
import { QaRecorder, type QaSession } from './qa-recorder';
import { STAT, assertOpLayout } from './constants';
import type { PerceptionSource, RenderFrame, SceneRenderer, ViewMode } from './types';

export class App {
  private readonly engine: AetherEngine;
  /** The engine's gesture-sequence book. Static for the session. */
  private readonly comboBook: string;
  /** The engine's two-hand duet book. Static for the session. */
  private readonly duetBook: string;
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
  /** Reused callback: diagnostics are built only when the recorder takes a sample. */
  private readonly qaDiagnostics = () => this.diagnostics;
  /** Every pool resize goes through here; see `engine-pool.ts`. */
  private readonly pool: EnginePool;
  private readonly camera = new Camera();
  private readonly perception: PerceptionPump;

  private readonly views: EngineViews;

  private running = false;
  private rafId = 0;
  /** The two deltas, the smoothed fps and the gap outside the callback. */
  private readonly clock = new FrameClock();
  private lastCameraTime = -1;
  private lastCameraMs = 0;
  private cameraAvailable = false;

  private mode: ViewMode = 'aether';
  private showCamera = true;

  /** What each stage of the callback cost; see `frame-timings.ts`. */
  private readonly timings = new FrameTimings();

  constructor(
    engine: AetherEngine,
    memory: WebAssembly.Memory,
    renderer: SceneRenderer,
    tier: EngineTier,
  ) {
    this.engine = engine;
    this.comboBook = engine.combo_book();
    this.duetBook = engine.duet_book();
    this.renderer = renderer;
    this.tier = tier;
    // The WebGPU backend simulates the pool itself: the engine stops advancing
    // its particles and starts logging what the spells did to them instead.
    if (renderer.backend === 'webgpu') {
      assertOpLayout(engine.particle_op_layout());
      engine.set_gpu_particles(true);
    }
    this.views = new EngineViews(engine, memory);
    this.pool = new EnginePool(engine, this.views, () => this.governor);
    this.perception = new PerceptionPump({
      engine,
      frame: () =>
        this.cameraAvailable && this.camera.hasFrame ? this.camera.video : null,
      mask: () => this.views.mask,
    });
    this.hud = new Hud(document.getElementById('hud')!, {
      onParam: (key, value) => {
        if (!this.engine.set_param(key, value)) {
          console.warn(`[aether] unknown engine parameter: ${key}`);
        }
      },
      onParticleCount: (n) => this.pool.setCount(n),
      onOverdrive: (on) => this.setOverdrive(on),
      onViewMode: (mode) => {
        this.mode = mode;
      },
      onToggleCamera: (on) => {
        this.showCamera = on;
      },
      onPractice: (on) => this.engine.set_practice(on),
      onReset: () => this.pool.reset(),
    });
    this.governor = new PerformanceGovernor(
      {
        setPressureIters: (iters) => {
          this.engine.set_param('pressure_iters', iters);
        },
        setParticleCount: (count) => this.pool.resize(count),
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
    // Let the worker load the vision graphs while camera permission and the
    // first video frame are pending, instead of waiting for both to finish.
    const disabled = new URLSearchParams(location.search).get('perception') === 'off';
    if (!disabled) void this.perception.attach();

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
      this.perception.close();
      this.perception.status = { kind: 'unavailable', reason: message };
    }

    // `?perception=off` keeps model loading out of the headless optical-flow
    // suite, where a software inference would stall unrelated assertions.
    if (this.cameraAvailable && disabled) {
      this.perception.status = {
        kind: 'unavailable',
        reason: 'Disabled by ?perception=off — optical flow only.',
      };
    }

    this.renderer.setVideo(this.cameraAvailable ? this.camera.video : null);
    this.running = true;
    this.clock.reset(performance.now());
    this.rafId = requestAnimationFrame(this.frame);
  }

  private readonly frame = (nowMs: number): void => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this.frame);

    // See `frame-clock.ts`: `simDt` is clamped for the integrator, `realDt` is
    // what actually elapsed and is the only honest input to an fps number.
    const { simDt } = this.clock.tick(nowMs);

    this.views.refresh();

    const t = this.timings;
    t.begin(this.clock.outside(t.frameMs));

    t.measure('cameraMs', () => {
      // Start the worker's asynchronous bitmap decode before the synchronous
      // luma readback; both inputs still reach the engine before step().
      this.perception.pump(nowMs);
      this.pumpCamera(nowMs);
    });
    t.measure('stepMs', () => this.engine.step(simDt));

    const stats = this.engine.stats();
    t.measure('renderMs', () => this.renderer.render(this.buildFrame(stats)));

    // The GPU owns the pool, so its own count is the only true one. It lags a
    // frame behind the readback, which the HUD's stats row can live with.
    if (this.engine.gpu_particles()) {
      this.engine.set_particles_alive(this.renderer.particlesAlive?.() ?? 0);
    }

    this.governor.update(this.clock.fps);
    this.overdrive.update(stats, this.clock.fps, t.stepMs);
    t.measure('hudMs', () =>
      this.hud.update({
        fps: this.clock.fps,
        stepMs: t.stepMs,
        renderMs: t.renderMs,
        inferenceMs: this.perception.inferenceMs,
        stats,
        spells: [this.engine.spell_name(0), this.engine.spell_name(1)],
        comboBook: this.comboBook,
        comboProgress: this.engine.combo_progress(),
        duetBook: this.duetBook,
        duetProgress: this.engine.duet_progress(),
        hands: stats[STAT.HANDS_PRESENT] > 0 ? this.perception.hands : null,
        perception: this.perception.status,
      }),
    );
    t.end();

    // Last in the frame: the recorder reads the values this frame just produced,
    // and it rate-limits itself to 2 Hz internally.
    this.qa.sample(this.qaDiagnostics);
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

  private buildFrame(stats: Float32Array): RenderFrame {
    // Motion energy is unbounded; a soft knee maps it into the [0, 1] the
    // renderer wants without clipping the common case to full brightness.
    const motion = stats[STAT.MOTION_ENERGY] ?? 0;
    const intensity = Math.min(1, Math.sqrt(Math.max(0, motion)) * 0.35);

    return {
      dye: this.views.dye,
      particles: this.views.particles(),
      particleCount: this.engine.particle_count(),
      debug: this.mode === 'debug' ? this.views.debug() : null,
      video: this.cameraAvailable ? this.camera.video : null,
      hands: stats[STAT.HANDS_PRESENT] > 0 ? this.perception.hands : null,
      time: this.clock.frames / 60,
      intensity,
      mode: this.mode,
      showCamera: this.showCamera && this.cameraAvailable,
      rush: this.engine.rush_state(),
      sim: this.engine.gpu_particles() ? this.views.gpuSim() : null,
    };
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
    this.perception.close();
    this.camera.stop();
  }

  // ------------------------------------------------------------ test hooks

  /** See `PerceptionPump.setSource`. */
  setPerception(source: PerceptionSource | null): void {
    this.perception.setSource(source);
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
      frames: this.clock.frames,
      fps: this.clock.fps,
      inferenceMs: this.perception.inferenceMs,
      // `cameraMs` is inclusive of the bitmap decode charged to `inferenceMs`.
      ...this.timings.rows,
      cameraAvailable: this.cameraAvailable,
      perception: this.perception.status,
      stats: Array.from(this.engine.stats()),
      spells: [this.engine.spell_name(0), this.engine.spell_name(1)] as [string, string],
      particleCount: this.engine.particle_count(),
      mode: this.mode,
      engine: this.tier,
      /** Measured rate at which inference results reach the engine. */
      perceptionHz: this.perception.actualHz,
      /** Inline inference budget cadence (not the measured result rate). */
      perceptionBudgetHz: this.perception.hz,
      inferenceCostMs: this.perception.costMs,
      /** Adaptive quality rung; 0 is full quality. */
      qualityTier: this.governor.level,
      /** Which graphics API drew the last frame, and who owns the pool. */
      renderBackend: this.renderer.backend,
    };
  }

  /** The intro has opened: first-run HUD moments may start now. */
  stage(): void {
    this.hud.stage();
  }

  /** Hands the engine sees this frame; cheap enough to poll at 20 Hz. */
  get handsPresent(): number {
    return this.engine.stats()[STAT.HANDS_PRESENT] ?? 0;
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
    this.pool.resize(n);
  }

  /** Overdrive: the full pool and a spawn rate that fills it. */
  setOverdrive(on: boolean): void {
    if (on === this.overdrive.active) return;
    this.pool.setOverdrive(on);
    this.overdrive.setActive(on);
  }

  /**
   * Mean luminance of the last rendered frame. Expensive — see `diagnostics`.
   */
  luminance(): number {
    return this.renderer.sampleLuminance();
  }
}
