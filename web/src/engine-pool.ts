/**
 * Every write that resizes the particle pool, in one place.
 *
 * Split out of `app.ts` because all three callers (the HUD slider, the overdrive
 * toggle and the test hook) have to obey the same two rules, and forgetting
 * either is a crash or a silently wrong quality ladder:
 *
 * - resizing the pool can reallocate inside WASM, which detaches every
 *   typed-array view — so `EngineViews.rebuild` follows every write;
 * - the number the user chose is the new 100% for the adaptive ladder, so the
 *   governor is rebased against it rather than against the boot default.
 *
 * Overdrive is a ceiling demo: the governor is paused while it runs so quality
 * cannot be pulled out from under it, and switching back restores the pool the
 * user had.
 */

import type { EngineViews } from "./engine-views";
import type { PerformanceGovernor } from "./performance-governor";
import {
  DEFAULT_SPAWN_RATE,
  OVERDRIVE_PARTICLES,
  OVERDRIVE_SPAWN_RATE,
} from "./overdrive";
import type { AetherEngine } from "./wasm/aether";

/**
 * `Params::default().pressure_iters` from `crates/aether-core/src/config.rs`.
 * The solver's iteration count is not HUD-exposed, so this is what the adaptive
 * quality ladder treats as 100%.
 */
export const DEFAULT_PRESSURE_ITERS = 28;

export class EnginePool {
  /** Particle count to return to when overdrive is switched off. */
  private beforeOverdrive = 0;
  private overdriveOn = false;

  constructor(
    private readonly engine: AetherEngine,
    private readonly views: EngineViews,
    private readonly governor: () => PerformanceGovernor,
  ) {}

  get count(): number {
    return this.engine.particle_count();
  }

  /** The HUD slider's write: resize, re-view, and rebase the ladder. */
  setCount(n: number): void {
    this.engine.set_particle_count(n);
    this.views.rebuild();
    this.governor().rebase(DEFAULT_PRESSURE_ITERS, n);
  }

  /** The test hook's write: resize and re-view, leaving the ladder alone. */
  resize(n: number): void {
    this.engine.set_particle_count(n);
    this.views.rebuild();
  }

  /** The full 1M pool plus a spawn rate that fills it, or back again. */
  setOverdrive(on: boolean): void {
    this.overdriveOn = on;
    if (on) {
      this.beforeOverdrive = this.engine.particle_count();
      this.engine.set_particle_count(OVERDRIVE_PARTICLES);
      this.engine.set_param("spawn_rate", OVERDRIVE_SPAWN_RATE);
    } else {
      this.engine.set_particle_count(this.beforeOverdrive);
      this.engine.set_param("spawn_rate", DEFAULT_SPAWN_RATE);
    }
    this.views.rebuild();
    this.governor().setPaused(on);
    if (!on)
      this.governor().rebase(DEFAULT_PRESSURE_ITERS, this.engine.particle_count());
  }

  get overdriveActive(): boolean {
    return this.overdriveOn;
  }

  /** Engine reset shares the reallocation rule, so it lives here too. */
  reset(): void {
    this.engine.reset();
    this.views.rebuild();
  }
}
