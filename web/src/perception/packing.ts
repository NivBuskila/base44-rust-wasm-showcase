/**
 * The DOM-free, MediaPipe-free half of perception: reading a recogniser result
 * into slots and packing it into the buffers the engine reads.
 *
 * Everything here is pure or owns nothing but a reusable buffer, so it can be
 * reasoned about (and tested) without a camera, a model or a GPU.
 *
 * Two things are subtle enough to state up front, because each is a bug that
 * looks like something else:
 *
 * 1. **Mirroring is not uniform.** The engine works in the mirrored view the
 *    user sees (`Camera.readLuma(mirror = true)`), but `push_mask` mirrors the
 *    silhouette itself. So landmarks are flipped here (`x -> 1 - x`) and the
 *    mask is passed through untouched. Flipping both makes gestures work while
 *    body collision appears on the wrong side.
 * 2. **Slot assignment must be sticky.** MediaPipe's result order is not
 *    stable, so writing `landmarks[i]` into slot `i` swaps hand identities
 *    mid-gesture and the latched spells jump between hands. Slots are assigned
 *    by a 2x2 cost match: handedness dominates, palm distance breaks the tie
 *    (and decides outright when both hands report the same handedness, which
 *    happens).
 */

import {
  HAND_LANDMARKS,
  HAND_STRIDE,
  HANDS,
  LM,
  MASK_IN_MAX_DIM,
  POSE_LANDMARKS,
  gestureId,
} from '../constants';
import type { MaskFrame } from '../types';
import type { GestureRecognizerResult, NormalizedLandmark } from '@mediapipe/tasks-vision';

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
 * Joints averaged into the palm centre used for matching. `gesture.rs` uses
 * five; `LM` exposes four of them, which is plenty for a nearest-neighbour key
 * that never feeds the simulation.
 */
const PALM_JOINTS: readonly number[] = [LM.WRIST, LM.INDEX_MCP, LM.MIDDLE_MCP, LM.PINKY_MCP];

/** The fields of `GestureRecognizerResult` this module actually reads. */
export type HandResultLike = Pick<GestureRecognizerResult, 'landmarks' | 'handedness' | 'gestures'>;

/** One hand as this module sees it: mirrored, validated, classifier-labelled. */
export interface HandDetection {
  /** 1 for a right hand *in the mirrored view*, 0 for a left one. */
  handedness: number;
  /** Confidence of the handedness label, `[0, 1]`. */
  handScore: number;
  /** Canned gesture id per `gestureId()`; 0 means "no gesture". */
  gesture: number;
  /** Confidence of the gesture label, `[0, 1]`. */
  score: number;
  /** Mirrored palm centre, normalised. Matching key, not simulation input. */
  palmX: number;
  palmY: number;
  /** Mirrored landmarks, `HAND_LANDMARKS * (x, y, z)`. */
  landmarks: Float32Array;
}

/** A reusable detection pool; `readDetections` never allocates. */
export function emptyDetections(): HandDetection[] {
  return Array.from({ length: HANDS }, () => ({
    handedness: 0,
    handScore: 0,
    gesture: 0,
    score: 0,
    palmX: 0,
    palmY: 0,
    landmarks: new Float32Array(HAND_LANDMARKS * 3),
  }));
}

/** Finite-or-default, then clamped to `[0, 1]`. NaN reads as `fallback`. */
function unit(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : fallback;
}

/**
 * Reads a recogniser result into `out`, mirrored, and returns how many hands
 * survived validation.
 *
 * A hand with a non-finite coordinate is dropped whole rather than patched:
 * the engine rejects such a buffer anyway, and half a hand would drag the
 * temporal filter towards a garbage pose.
 */
export function readDetections(result: HandResultLike, out: HandDetection[]): number {
  const limit = Math.min(result.landmarks.length, out.length);
  let count = 0;

  for (let i = 0; i < limit; i++) {
    const lm = result.landmarks[i];
    if (!lm || lm.length < HAND_LANDMARKS) continue;

    const det = out[count];
    const dst = det.landmarks;
    let ok = true;
    for (let j = 0; j < HAND_LANDMARKS; j++) {
      const p = lm[j];
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
        ok = false;
        break;
      }
      // The single place mirroring happens for landmarks. z is a depth offset,
      // unaffected by a horizontal flip.
      dst[j * 3] = 1 - p.x;
      dst[j * 3 + 1] = p.y;
      dst[j * 3 + 2] = Number.isFinite(p.z) ? p.z : 0;
    }
    if (!ok) continue;

    const hand = result.handedness[i]?.[0];
    // MediaPipe labels the hand as if the image were *not* mirrored, so the
    // view the user sees reads it inverted: their right hand is labelled Left.
    det.handedness = hand?.categoryName === 'Left' ? 1 : 0;
    det.handScore = unit(hand?.score, 0);

    const gesture = result.gestures[i]?.[0];
    det.gesture = gesture ? gestureId(gesture.categoryName) : 0;
    det.score = unit(gesture?.score, 0);

    let px = 0;
    let py = 0;
    for (const j of PALM_JOINTS) {
      px += dst[j * 3];
      py += dst[j * 3 + 1];
    }
    det.palmX = px / PALM_JOINTS.length;
    det.palmY = py / PALM_JOINTS.length;

    count++;
  }
  return count;
}

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

/** Writes the packed hand buffer: `[present, handedness, gestureId, score, ...xyz]`. */
export function packHands(
  buf: Float32Array,
  dets: readonly HandDetection[],
  slots: Int32Array,
): void {
  for (let slot = 0; slot < HANDS; slot++) {
    const base = slot * HAND_STRIDE;
    const i = slots[slot];
    if (i < 0) {
      // Zero the whole slot, not just `present`: the renderer's landmark
      // overlay reads the coordinates straight out of this buffer.
      buf.fill(0, base, base + HAND_STRIDE);
      continue;
    }
    const det = dets[i];
    buf[base] = 1;
    buf[base + 1] = det.handedness;
    buf[base + 2] = det.gesture;
    buf[base + 3] = det.score;
    buf.set(det.landmarks, base + 4);
  }
}

/** Writes the packed pose buffer: `[present, ...33 * (x, y, z, visibility)]`. */
export function packPose(
  buf: Float32Array,
  landmarks: readonly NormalizedLandmark[] | undefined,
): boolean {
  if (!landmarks || landmarks.length < POSE_LANDMARKS) {
    buf.fill(0);
    return false;
  }
  buf[0] = 1;
  for (let i = 0; i < POSE_LANDMARKS; i++) {
    const o = 1 + i * 4;
    const p = landmarks[i];
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      // Visibility 0 is how the engine is told to ignore a joint; a NaN
      // coordinate with a high visibility would be read as a real position.
      buf[o] = 0;
      buf[o + 1] = 0;
      buf[o + 2] = 0;
      buf[o + 3] = 0;
      continue;
    }
    buf[o] = 1 - p.x;
    buf[o + 1] = p.y;
    buf[o + 2] = Number.isFinite(p.z) ? p.z : 0;
    buf[o + 3] = unit(p.visibility, 0);
  }
  return true;
}

/**
 * Box-downscales a segmentation mask to fit `MASK_IN_MAX_DIM`.
 *
 * `app.ts` silently drops anything larger than the engine's staging buffer, so
 * an oversized mask would mean no body collision at all. Averaging rather than
 * point-sampling matters here: the mask is a coverage field the engine
 * thresholds, and dropping 15 of every 16 pixels makes a thin limb flicker in
 * and out of existence between frames.
 *
 * The output is *not* mirrored — `push_mask` does that itself.
 */
export class MaskScaler {
  private buf = new Float32Array(0);
  private readonly frame: MaskFrame = { data: this.buf, width: 0, height: 0 };

  /** Returns a frame that is valid until the next call, or null if unusable. */
  fit(data: Float32Array, width: number, height: number): MaskFrame | null {
    const w = Math.floor(width);
    const h = Math.floor(height);
    if (!(w > 0) || !(h > 0) || data.length < w * h) return null;

    if (w <= MASK_IN_MAX_DIM && h <= MASK_IN_MAX_DIM) {
      // Pass the model's own buffer through; the caller copies it into WASM
      // before the next inference can touch it, and `BodyMask::stage` already
      // clamps non-finite confidences.
      this.frame.data = data;
      this.frame.width = w;
      this.frame.height = h;
      return this.frame;
    }

    const scale = MASK_IN_MAX_DIM / Math.max(w, h);
    const outW = Math.max(1, Math.min(MASK_IN_MAX_DIM, Math.floor(w * scale)));
    const outH = Math.max(1, Math.min(MASK_IN_MAX_DIM, Math.floor(h * scale)));
    if (this.buf.length < outW * outH) this.buf = new Float32Array(outW * outH);
    const out = this.buf;

    for (let oy = 0; oy < outH; oy++) {
      const y0 = Math.floor((oy * h) / outH);
      const y1 = Math.max(y0 + 1, Math.floor(((oy + 1) * h) / outH));
      for (let ox = 0; ox < outW; ox++) {
        const x0 = Math.floor((ox * w) / outW);
        const x1 = Math.max(x0 + 1, Math.floor(((ox + 1) * w) / outW));
        let sum = 0;
        for (let y = y0; y < y1; y++) {
          const row = y * w;
          for (let x = x0; x < x1; x++) {
            // Written as nested comparisons rather than a clamp so that a NaN
            // confidence collapses to 0 instead of poisoning the average.
            const v = data[row + x];
            sum += v > 0 ? (v < 1 ? v : 1) : 0;
          }
        }
        out[oy * outW + ox] = sum / ((y1 - y0) * (x1 - x0));
      }
    }

    this.frame.data = out;
    this.frame.width = outW;
    this.frame.height = outH;
    return this.frame;
  }
}

/**
 * Next timestamp to feed the graph, strictly greater than `prev`.
 *
 * MediaPipe rejects a timestamp it has already seen, and `performance.now()`
 * can return the same value twice inside one millisecond. The bump is a whole
 * millisecond rather than an epsilon because the graph quantises to
 * microseconds, so a smaller nudge can round back onto the previous tick.
 */
export function nextTimestamp(prev: number, now: number): number {
  const t = Number.isFinite(now) ? now : prev + 1;
  return t > prev ? t : prev + 1;
}
