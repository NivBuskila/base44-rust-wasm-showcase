/**
 * The panel's frame-time reading, without the DOM.
 *
 * `Hud.update` runs exactly once per rendered frame, so the interval between
 * calls *is* the frame period — and unlike `stats.fps` it is not floored by the
 * loop's own 50 ms dt clamp, which reports a hard stall as a tidy 20 fps.
 *
 * Two rules live here, both load-bearing for a readout that must not lie:
 * - **peak-hold, not the sample that landed on the paint tick**: a 40 ms hitch
 *   every 300 ms is precisely what a frame-budget readout must not hide;
 * - **`stats.fps` is only an honest fallback once it has warmed up**: it is an
 *   EMA seeded at zero, so the first frame of a 60 Hz session reports ~6 fps.
 *   Charting that invents a 166 ms frame that never happened, and because the
 *   sparkline scales to the worst sample in its window that phantom flattens
 *   the next nine seconds of real history into two pixels. With nothing
 *   measured and nothing trustworthy to fall back on, `read` returns 0 and the
 *   chip keeps its placeholder instead of guessing.
 */

/**
 * Longer than this between two `update` calls is the tab having been
 * backgrounded, not a slow frame; charting it would peg the sparkline for nine
 * seconds over something the user never saw.
 */
const FRAME_GAP_MAX_MS = 500;

/** `update` calls before `stats.fps` is believable. */
const FPS_WARMUP_FRAMES = 20;

export class FrameTimer {
  private peak = 0;
  private lastMs = 0;
  /** `update` calls so far, capped at `FPS_WARMUP_FRAMES`. */
  private updates = 0;

  /** Call once per `update`, ahead of the paint gate. */
  sample(now: number): void {
    const gap = this.lastMs > 0 ? now - this.lastMs : 0;
    this.lastMs = now;
    if (gap > 0 && gap <= FRAME_GAP_MAX_MS && gap > this.peak) this.peak = gap;
    if (this.updates < FPS_WARMUP_FRAMES) this.updates++;
  }

  /**
   * The worst frame since the last call, in ms, and resets the hold. `fps` is
   * the loop's own smoothed rate, used only when nothing was measurable.
   * Returns 0 when there is nothing honest to report.
   */
  read(fps: number): number {
    const peak = this.peak;
    this.peak = 0;
    if (peak > 0.001) return peak;
    if (this.updates < FPS_WARMUP_FRAMES) return 0;
    return fps > 0.001 ? Math.min(1000, 1000 / fps) : 0;
  }
}
