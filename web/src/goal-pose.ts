/**
 * The target hand on the tutorial card.
 *
 * Most lessons are a shape, so the target is one still skeleton. `release` is a
 * *movement* — squeeze, then open — and a still drawing of an open hand cannot
 * say that, so it plays as a loop morphing between the fist and the open hand.
 * Both poses are fitted together, so the hand opens instead of resizing.
 */

import { fit, HandSkeleton } from './hand-skeleton';
import { poseLandmarks } from './pose-landmarks';

/** Lessons taught as a motion: the two poses the loop runs between. */
const SEQUENCES: Record<string, readonly [string, string]> = {
  release: ['attract', 'release'],
};
/** One there-and-back cycle. */
const CYCLE_MS = 1800;

/** Ease in-out, so the hand settles at each end instead of ticking. */
function ease(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
}

export class GoalPose {
  private from: number[][] = [];
  private to: number[][] = [];
  private raf = 0;

  constructor(private readonly skeleton: HandSkeleton) {}

  /** Draws the taught pose — still, or looping when the lesson is a motion. */
  show(spell: string): void {
    this.stop();
    const seq = SEQUENCES[spell];
    if (!seq) {
      const pts = poseLandmarks(spell);
      if (pts) this.skeleton.draw(fit(pts));
      return;
    }
    const a = poseLandmarks(seq[0]);
    const b = poseLandmarks(seq[1]);
    if (!a || !b) return;
    const both = fit([...a, ...b]);
    this.from = both.slice(0, a.length);
    this.to = both.slice(a.length);
    const start = performance.now();
    const tick = (now: number): void => {
      const phase = ((now - start) % CYCLE_MS) / CYCLE_MS;
      const t = ease(phase < 0.5 ? phase * 2 : (1 - phase) * 2);
      this.skeleton.draw(
        this.from.map(([x, y], i) => [x + (this.to[i][0] - x) * t, y + (this.to[i][1] - y) * t]),
      );
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  /** Stops the loop, so a closed card costs nothing. */
  stop(): void {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
  }
}
