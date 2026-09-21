/**
 * The frame loop's clock: the two deltas, the smoothed frame rate and the
 * unspent gap between callbacks. Pure and DOM-free, pinned by
 * `frame-clock.test.ts`.
 *
 * `simDt` and `realDt` are two different quantities, and conflating them is how
 * a frame-rate readout ends up unable to report the problem it exists to
 * report: `simDt` is clamped so one long stall does not blow the integrator
 * apart, while `realDt` is what actually elapsed and is the only honest input to
 * an fps number.
 */

/** Rolling window for the fps readout. */
const FPS_SMOOTHING = 0.9;

/** Integrator clamps: never below 1/480 s, never above 50 ms. */
const MIN_SIM_DT = 1 / 480;
const MAX_SIM_DT = 0.05;

export interface FrameTick {
  /** Wall-clock seconds since the previous callback. */
  realDt: number;
  /** Clamped step for the engine. */
  simDt: number;
}

export class FrameClock {
  private lastMs = 0;
  private realDt = 0;
  private smoothedFps = 0;
  private count = 0;

  /** Call once at `start`, so the first delta is not the page's whole age. */
  reset(nowMs: number): void {
    this.lastMs = nowMs;
  }

  tick(nowMs: number): FrameTick {
    const realDt = Math.max(1e-4, (nowMs - this.lastMs) / 1000);
    this.lastMs = nowMs;
    this.realDt = realDt;
    this.smoothedFps =
      this.smoothedFps * FPS_SMOOTHING + (1 / realDt) * (1 - FPS_SMOOTHING);
    this.count++;
    return {
      realDt,
      simDt: Math.min(MAX_SIM_DT, Math.max(MIN_SIM_DT, realDt)),
    };
  }

  get fps(): number {
    return this.smoothedFps;
  }

  get frames(): number {
    return this.count;
  }

  /**
   * The gap this frame's callback did not spend: wall time since the previous
   * callback minus the work that one did. `step`/`render`/`infer` summing far
   * below the frame period means the cost is here — GPU work the driver finishes
   * after the GL calls return, compositing, or another task blocking the loop —
   * and without this row that time is invisible and the HUD appears to
   * contradict the frame rate it sits next to.
   */
  outside(previousFrameMs: number): number {
    return Math.max(0, this.realDt * 1000 - previousFrameMs);
  }
}
