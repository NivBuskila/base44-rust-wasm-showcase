/**
 * The panel's frame-time reading, without the DOM.
 *
 * `Hud.update` runs exactly once per rendered frame and is handed that frame's
 * `requestAnimationFrame` timestamp, so the interval between samples *is* the
 * frame period — and unlike `stats.fps` it is not floored by the loop's own
 * 50 ms dt clamp, which reports a hard stall as a tidy 20 fps. It must be the
 * frame's timestamp, not `performance.now()` at the call: the call comes after
 * the engine step, so its time drifts with the step's cost, and that jitter
 * read a steady 60 Hz loop as the high 40s.
 *
 * Each paint reads two numbers from the same intervals:
 * - **the delivered rate, for the digits**: frames per second actually shown
 *   over about the last second. Not the worst frame: the rAF timestamps sit on
 *   the vsync grid, so any one interval is a whole number of refreshes, and the
 *   worst of a 100 ms window is one refresh or two. Shown as fps, that turns a
 *   loop missing one refresh in twelve (57 fps delivered) into a flat 30;
 * - **the worst frame, for the sparkline**: peak-hold per paint, not the sample
 *   that landed on the tick, because a 40 ms hitch every 300 ms is precisely
 *   what a frame-budget chart must not hide.
 *
 * `stats.fps` is only an honest fallback once it has warmed up: it is an EMA
 * seeded at zero, so the first frame of a 60 Hz session reports ~6 fps.
 * Charting that invents a 166 ms frame that never happened, and because the
 * sparkline scales to the worst sample in its window that phantom flattens the
 * next nine seconds of real history into two pixels. With nothing measured and
 * nothing trustworthy to fall back on, `read` returns zeros and the chip keeps
 * its placeholder instead of guessing.
 */

/**
 * Longer than this between two `update` calls is the tab having been
 * backgrounded, not a slow frame; charting it would peg the sparkline for nine
 * seconds over something the user never saw.
 */
const FRAME_GAP_MAX_MS = 500;

/** `update` calls before `stats.fps` is believable. */
const FPS_WARMUP_FRAMES = 20;

/** How much recent history the delivered rate averages over. */
const RATE_WINDOW_MS = 1000;

/** Intervals the rate window can hold: a second at 500 Hz. */
const RATE_CAPACITY = 512;

/** One paint's reading, in ms per frame; both 0 when there is nothing honest to show. */
export interface FrameReading {
  /** Mean interval over about the last second: 1000 / the delivered fps. */
  meanMs: number;
  /** The worst interval since the previous read. */
  worstMs: number;
}

export class FrameTimer {
  private peak = 0;
  private lastMs = 0;
  /** `update` calls so far, capped at `FPS_WARMUP_FRAMES`. */
  private updates = 0;
  /** Recent intervals, oldest at `head`, summing to about `RATE_WINDOW_MS`. */
  private readonly gaps = new Float64Array(RATE_CAPACITY);
  private head = 0;
  private count = 0;
  private sum = 0;

  /** Call once per `update`, ahead of the paint gate. */
  sample(now: number): void {
    const gap = this.lastMs > 0 ? now - this.lastMs : 0;
    this.lastMs = now;
    if (this.updates < FPS_WARMUP_FRAMES) this.updates++;
    if (gap <= 0) return;
    if (gap > FRAME_GAP_MAX_MS) {
      // Back from the background: the second before it is not this one.
      this.head = this.count = this.sum = 0;
      return;
    }
    if (gap > this.peak) this.peak = gap;
    this.push(gap);
  }

  /**
   * This paint's reading, and resets the worst-frame hold. `fps` is the loop's
   * own smoothed rate, used only when nothing was measurable.
   */
  read(fps: number): FrameReading {
    const worst = this.peak;
    this.peak = 0;
    if (worst > 0.001) return { meanMs: this.sum / this.count, worstMs: worst };
    if (this.updates < FPS_WARMUP_FRAMES || !(fps > 0.001)) return { meanMs: 0, worstMs: 0 };
    const ms = Math.min(1000, 1000 / fps);
    return { meanMs: ms, worstMs: ms };
  }

  private push(gap: number): void {
    if (this.count === RATE_CAPACITY) this.dropOldest();
    this.gaps[(this.head + this.count) % RATE_CAPACITY] = gap;
    this.count++;
    this.sum += gap;
    // Keep the newest intervals that still cover the window.
    while (this.count > 1 && this.sum - this.gaps[this.head] >= RATE_WINDOW_MS) this.dropOldest();
  }

  private dropOldest(): void {
    this.sum -= this.gaps[this.head];
    this.head = (this.head + 1) % RATE_CAPACITY;
    this.count--;
  }
}
