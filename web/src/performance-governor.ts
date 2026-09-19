/**
 * Adaptive quality: keeps the loop at the refresh rate instead of at "whatever
 * this machine manages".
 *
 * The three costs in a frame are the solver (`engine.step`), the draw
 * (particles + bloom at scene resolution) and, when a camera is attached,
 * gesture inference. The first two are the ones a device can be too slow for,
 * and both have knobs that trade detail for time — pressure iterations, pool
 * size, internal resolution. Fixed settings mean the same numbers have to work
 * on a discrete GPU and on a software rasteriser, so they are chosen for the
 * weakest target and the strong one is left idle.
 *
 * So they move at runtime. The governor watches the smoothed frame rate and
 * walks a short ladder of quality tiers: down quickly when frames are being
 * missed (a stutter is the one thing the user always notices), up slowly and
 * only from sustained headroom, so it settles instead of oscillating between
 * two tiers. Tier 0 is exactly the app's own defaults, so a machine that keeps
 * up never sees the governor at all.
 *
 * Deliberately not governed: overdrive. That mode exists to show the engine's
 * ceiling at one million particles, and quietly shrinking the pool to protect
 * the frame rate would erase the thing being demonstrated.
 */

/** What the governor is allowed to touch. */
export interface QualityKnobs {
  /** Jacobi iterations in the pressure projection. */
  setPressureIters(iters: number): void;
  /** Particle pool size. */
  setParticleCount(count: number): void;
  /** Multiplier on the renderer's internal resolution. */
  setRenderScale(scale: number): void;
}

/**
 * One rung of the ladder, as fractions of the user's own settings.
 *
 * Resolution comes down first and hardest: it is the cheapest to give up
 * (the composite scales the scene back to canvas size, and at 0.85 that is
 * invisible in motion) and it cuts every full-screen pass at once. Pressure
 * iterations follow — far below the default the projection starts leaking
 * divergence and the vortices go soft, hence the absolute floor in `apply`.
 * The pool shrinks last, because particle density is what the app looks like.
 */
interface Tier {
  render: number;
  pressure: number;
  particles: number;
}

const TIERS: readonly Tier[] = [
  { render: 1.0, pressure: 1.0, particles: 1.0 },
  { render: 0.85, pressure: 0.75, particles: 1.0 },
  { render: 0.72, pressure: 0.6, particles: 0.7 },
  { render: 0.6, pressure: 0.5, particles: 0.45 },
];

/** Below this smoothed fps the current tier is not holding. */
const DOWN_FPS = 50;
/** Above this there is room for the tier above. */
const UP_FPS = 58;
/** Frames under `DOWN_FPS` before dropping a tier (~0.5 s at 60 Hz). */
const DOWN_FRAMES = 30;
/** Frames over `UP_FPS` before climbing one (~4 s), so it settles. */
const UP_FRAMES = 240;
/** Frames ignored after a change, while the fps average catches up. */
const SETTLE_FRAMES = 45;

export class PerformanceGovernor {
  private readonly knobs: QualityKnobs;

  /** The user's settings, which tier 0 must reproduce exactly. */
  private basePressure: number;
  private baseParticles: number;

  private tier = 0;
  private below = 0;
  private above = 0;
  private settle = SETTLE_FRAMES;
  private paused = false;

  constructor(knobs: QualityKnobs, basePressure: number, baseParticles: number) {
    this.knobs = knobs;
    this.basePressure = basePressure;
    this.baseParticles = baseParticles;
  }

  /** Which rung is in effect; 0 means the user's own settings. */
  get level(): number {
    return this.tier;
  }

  /**
   * Re-reads the settings the tiers are relative to.
   *
   * Called when the user changes the pool or the solver from the HUD: their new
   * number is the new 100%, and the current tier is re-applied on top of it so
   * a slow device does not jump back to full cost on a slider drag.
   */
  rebase(pressure: number, particles: number): void {
    this.basePressure = pressure;
    this.baseParticles = particles;
    this.settle = SETTLE_FRAMES;
    // The caller has just set the knobs to 100%; a device already down the
    // ladder must not run at full cost until the next tier change.
    if (this.tier !== 0) this.apply(this.tier);
  }

  /**
   * Suspends adaptation and returns to the user's settings.
   *
   * Used for overdrive, where the point is the ceiling, not the frame rate.
   */
  setPaused(paused: boolean): void {
    if (paused === this.paused) return;
    this.paused = paused;
    if (paused && this.tier !== 0) this.apply(0);
    this.below = 0;
    this.above = 0;
    this.settle = SETTLE_FRAMES;
  }

  /** Once per frame, with the loop's smoothed frame rate. */
  update(fps: number): void {
    if (this.paused) return;
    // The first frames of a session are dominated by shader compilation and
    // the first WASM step; judging quality on them drops two tiers for nothing.
    if (this.settle > 0) {
      this.settle--;
      return;
    }
    if (!Number.isFinite(fps) || fps <= 0) return;

    if (fps < DOWN_FPS) {
      this.above = 0;
      if (++this.below >= DOWN_FRAMES && this.tier < TIERS.length - 1) {
        this.apply(this.tier + 1);
      }
      return;
    }

    this.below = 0;
    if (fps > UP_FPS) {
      if (++this.above >= UP_FRAMES && this.tier > 0) this.apply(this.tier - 1);
    } else {
      this.above = 0;
    }
  }

  private apply(tier: number): void {
    const next = TIERS[tier]!;
    this.tier = tier;
    this.below = 0;
    this.above = 0;
    this.settle = SETTLE_FRAMES;

    this.knobs.setRenderScale(next.render);
    this.knobs.setPressureIters(Math.max(10, Math.round(this.basePressure * next.pressure)));
    this.knobs.setParticleCount(Math.max(2000, Math.round(this.baseParticles * next.particles)));
  }
}
