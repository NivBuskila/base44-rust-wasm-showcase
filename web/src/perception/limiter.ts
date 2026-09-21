/**
 * The duty-cycle limiter, split out of `perception.ts` as pure arithmetic.
 *
 * It exists for one reason: `tasks-vision` is synchronous, so on the main thread
 * an expensive pass stalls the render loop and rationing is the only way to keep
 * the app alive. Up to about a frame and a half the loop's own 30 Hz gate is the
 * binding constraint and rationing would only throw away landmarks the caller
 * asked for — so below `BUDGET_MS` it stays out of the way entirely.
 *
 * Inside `perception.worker.ts` none of that applies (the worker blocks nobody
 * and its client keeps one inference in flight), which is why it is constructed
 * disabled there: left on, it inserts idle time between landmarks — at a 60 ms
 * pass ~16 Hz down to ~6 Hz — felt directly as hands lagging behind the fluid.
 */

/** Inference cost, in ms, above which perception starts rationing itself. */
export const BUDGET_MS = 50;

/** Share of wall-clock time inference may consume once past `BUDGET_MS`. */
export const MAX_DUTY = 0.35;

/** Gain of the inference-cost estimate; ~4 inferences to track a change. */
export const COST_GAIN = 0.25;

export class DutyLimiter {
  /** Smoothed inference cost driving the limiter, in ms. */
  costMs = 0;
  private nextRunMs = 0;

  constructor(private readonly enabled: boolean) {}

  /** False while the limiter is still holding inference back. */
  ready(nowMs: number): boolean {
    return nowMs >= this.nextRunMs;
  }

  /**
   * Charges one inference and schedules the next.
   *
   * The first pass through each graph also compiles shaders and allocates
   * textures; it routinely costs twenty times the steady state, so it updates
   * the schedule but never the estimate — folding it in would ration the whole
   * first ten seconds of the session for no reason. `finishedMs` is the moment
   * the main thread became free again, so the gap is idle time, not a period.
   */
  charge(latencyMs: number, finishedMs: number, firstInference: boolean): void {
    if (!firstInference) this.costMs += (latencyMs - this.costMs) * COST_GAIN;
    this.nextRunMs = finishedMs + this.gapMs;
  }

  /** Idle time owed after an inference costing `costMs`. */
  get gapMs(): number {
    if (!this.enabled) return 0;
    return this.costMs <= BUDGET_MS ? 0 : this.costMs * (1 / MAX_DUTY - 1);
  }

  reset(): void {
    this.costMs = 0;
    this.nextRunMs = 0;
  }
}
