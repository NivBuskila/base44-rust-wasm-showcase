/**
 * Reading a recogniser result into the reusable detection pool — the one place
 * hand landmarks are mirrored.
 *
 * The engine works in the mirrored view the user sees (`Camera.readLuma(mirror =
 * true)`), but `push_mask` mirrors the silhouette itself. So landmarks are
 * flipped here (`x -> 1 - x`) and the mask is passed through untouched by
 * `mask.ts`. Flipping both makes gestures work while body collision appears on
 * the wrong side.
 */

import type { GestureRecognizerResult } from '@mediapipe/tasks-vision';

import { HAND_LANDMARKS, HANDS, LM, gestureId } from '../../constants';

/**
 * Joints averaged into the palm centre used for matching. `gesture.rs` uses
 * five; `LM` exposes four of them, which is plenty for a nearest-neighbour key
 * that never feeds the simulation.
 */
const PALM_JOINTS: readonly number[] = [
  LM.WRIST,
  LM.INDEX_MCP,
  LM.MIDDLE_MCP,
  LM.PINKY_MCP,
];

/** The fields of `GestureRecognizerResult` this module actually reads. */
export type HandResultLike = Pick<
  GestureRecognizerResult,
  'landmarks' | 'handedness' | 'gestures'
>;

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
export function unit(value: number | undefined, fallback: number): number {
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

