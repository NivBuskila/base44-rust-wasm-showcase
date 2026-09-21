/**
 * Sticky slot assignment, split out of the packing index.
 *
 * MediaPipe's result order is not stable, so writing `landmarks[i]` into slot
 * `i` swaps hand identities mid-gesture and the latched spells jump between
 * hands. Slots are assigned by a 2x2 cost match instead: handedness dominates,
 * palm distance breaks the tie (and decides outright when both hands report the
 * same handedness, which happens).
 */

import { HANDS } from '../../constants';
import type { HandDetection } from './detections';

/** How long a vacated slot keeps its identity, in ms. */
const TRACK_TTL_MS = 600;

/**
 * Palm distance, in normalised screen units, that a handedness disagreement is
 * worth during matching.
 *
 * A palm moves ~0.02-0.10 between 30 Hz inferences even when swiping hard, so
 * this makes handedness the primary key: position only overrides the label once
 * the nearest track is a quarter of the screen away, which in practice means
 * the label is the thing that is wrong. Scaled by the classifier's own
 * confidence, so a hesitant label does not outvote geometry.
 */
const MISMATCH_COST = 0.25;

/**
 * Cost charged for taking an unoccupied slot. Any live track nearer than this
 * wins, so hands prefer to keep the slot they had; beyond it, a stale track is
 * not evidence and the canonical mapping (slot 0 = left) takes over.
 */
const EMPTY_SLOT_COST = 0.5;

/**
 * Sticky slot assignment for up to `HANDS` detections.
 *
 * Solved by brute force: with two slots there are two permutations, so the
 * optimal match is two additions and a compare — a general assignment solver
 * would be more code and more cycles for the same answer.
 */
export class HandSlots {
  private readonly track = Array.from({ length: HANDS }, () => ({
    live: false,
    handedness: 0,
    x: 0,
    y: 0,
    seenMs: 0,
  }));
  private readonly out = new Int32Array(HANDS);

  /**
   * Maps each slot to a detection index, or -1 when the slot is empty this
   * frame. Commits the chosen match as the next frame's tracking state.
   */
  assign(dets: readonly HandDetection[], count: number, nowMs: number): Int32Array {
    const out = this.out;
    out.fill(-1);

    // A count of zero falls through with both slots empty and, deliberately,
    // both tracks untouched: the recogniser drops a frame constantly, and a
    // dropout that reshuffled the slots would be worse than the dropout.
    if (count === 1) {
      out[this.cost(dets[0], 0, nowMs) <= this.cost(dets[0], 1, nowMs) ? 0 : 1] = 0;
    } else if (count >= 2) {
      const straight = this.cost(dets[0], 0, nowMs) + this.cost(dets[1], 1, nowMs);
      const swapped = this.cost(dets[0], 1, nowMs) + this.cost(dets[1], 0, nowMs);
      if (straight <= swapped) {
        out[0] = 0;
        out[1] = 1;
      } else {
        out[0] = 1;
        out[1] = 0;
      }
    }

    for (let slot = 0; slot < HANDS; slot++) {
      const i = out[slot];
      if (i < 0) continue;
      const det = dets[i];
      const t = this.track[slot];
      t.live = true;
      t.handedness = det.handedness;
      t.x = det.palmX;
      t.y = det.palmY;
      t.seenMs = nowMs;
    }
    return out;
  }

  /** Cost of putting `det` in `slot`: palm distance plus a handedness penalty. */
  private cost(det: HandDetection, slot: number, nowMs: number): number {
    const t = this.track[slot];
    if (!t.live || nowMs - t.seenMs > TRACK_TTL_MS) {
      // Nothing to compare against, so the only preference left is the
      // canonical mapping: slot 0 holds the left hand.
      return EMPTY_SLOT_COST + (det.handedness === slot ? 0 : MISMATCH_COST * det.handScore);
    }
    const d = Math.hypot(det.palmX - t.x, det.palmY - t.y);
    return d + (det.handedness === t.handedness ? 0 : MISMATCH_COST * det.handScore);
  }

  reset(): void {
    for (const t of this.track) t.live = false;
  }
}

