/**
 * The loop's stopwatch, split out of `app.ts` beside `frame-clock.ts`.
 *
 * `FrameClock` owns the deltas the simulation and the fps number are made of;
 * this owns the *costs* — how long each stage of one callback took — which is a
 * different question and the one the HUD meters and the QA session read.
 *
 * Two of the rows are easy to misread, so they are named here rather than in the
 * caller: `frameMs` is the whole callback, and `outsideMs` is the wall-clock gap
 * it did not spend. When the measured stages sum far under the frame period,
 * `outsideMs` is the row that matters — that time is GPU work completing after
 * the draw calls returned, compositing, or another task holding the loop.
 */

export interface FrameTimingRows {
  stepMs: number;
  renderMs: number;
  cameraMs: number;
  hudMs: number;
  frameMs: number;
  outsideMs: number;
}

/** Stage keys `measure` may write; the two derived rows are set by the loop. */
type Stage = "stepMs" | "renderMs" | "cameraMs" | "hudMs";

export class FrameTimings {
  /** `engine.step`. */
  stepMs = 0;
  /** The draw, whichever backend owns it. */
  renderMs = 0;
  /** Camera pump: grabbing the video frame and handing it to perception. */
  cameraMs = 0;
  /** HUD DOM update, which touches layout and is not free at 60 Hz. */
  hudMs = 0;
  /** Everything the callback did, so the measured rows can be checked to sum. */
  frameMs = 0;
  /** See `FrameClock.outside`. */
  outsideMs = 0;

  private frameStart = 0;

  /** Opens the frame; `outsideMs` is what the clock reported for the gap. */
  begin(outsideMs: number): void {
    this.frameStart = performance.now();
    this.outsideMs = outsideMs;
  }

  /** Runs `work` and charges its wall-clock cost to `stage`. */
  measure<T>(stage: Stage, work: () => T): T {
    const start = performance.now();
    const out = work();
    this[stage] = performance.now() - start;
    return out;
  }

  /** Closes the frame, charging everything since `begin` to `frameMs`. */
  end(): void {
    this.frameMs = performance.now() - this.frameStart;
  }

  /** The rows as data, for `diagnostics` and the HUD. */
  get rows(): FrameTimingRows {
    return {
      stepMs: this.stepMs,
      renderMs: this.renderMs,
      cameraMs: this.cameraMs,
      hudMs: this.hudMs,
      frameMs: this.frameMs,
      outsideMs: this.outsideMs,
    };
  }
}
