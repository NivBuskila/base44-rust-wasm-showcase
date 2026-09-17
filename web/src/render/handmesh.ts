/**
 * Turns the packed hand buffer into overlay geometry.
 *
 * This is the only per-frame CPU work in the renderer, and it is bounded at
 * 126 vertices — two hands of 21 bones and 21 joints — so it costs nothing next
 * to the particle upload it shares a buffer path with.
 */

import { HAND_LANDMARKS, HAND_STRIDE, HANDS, LM } from '../constants';

/** MediaPipe's hand skeleton, as index pairs into the 21 landmarks. */
const BONES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

/** Fingertips are the business end of the hand, so they glow hardest. */
const TIPS: readonly number[] = [LM.THUMB_TIP, LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP];

/** Floats per overlay vertex: `x, y, glow`. */
export const OVERLAY_STRIDE = 3;

/** Worst case: both hands, every bone as two vertices, every joint. */
export const OVERLAY_CAPACITY = HANDS * (BONES.length * 2 + HAND_LANDMARKS);

/** Vertex ranges in the buffer: lines first, then points. */
export interface OverlayMesh {
  lineVertices: number;
  pointVertices: number;
}

/**
 * Writes bone segments followed by joint points into `out`.
 *
 * Landmarks arrive in the same normalised, already-mirrored `[0, 1]` view space
 * as the dye and the particles — the engine's whole convention is the flipped
 * image the user sees — so the overlay needs no flip of its own.
 *
 * Non-finite or wildly out-of-range landmarks are dropped rather than clamped:
 * clamping garbage to the edge draws a bone across the whole screen, which
 * reads as a renderer bug, while a missing bone reads as tracking noise.
 */
export function buildHandMesh(hands: Float32Array, out: Float32Array): OverlayMesh {
  const cap = Math.floor(out.length / OVERLAY_STRIDE);
  let n = 0;

  const push = (x: number, y: number, glow: number): void => {
    if (n >= cap) return;
    const o = n * OVERLAY_STRIDE;
    out[o] = x;
    out[o + 1] = y;
    out[o + 2] = glow;
    n++;
  };

  // NaN fails every comparison, so this rejects non-finite landmarks too.
  const usable = (i: number): boolean =>
    hands[i] >= -0.5 && hands[i] <= 1.5 && hands[i + 1] >= -0.5 && hands[i + 1] <= 1.5;

  const present: Array<{ lm: number; gain: number }> = [];
  for (let h = 0; h < HANDS; h++) {
    const base = h * HAND_STRIDE;
    if (base + HAND_STRIDE > hands.length) break;
    if (!(hands[base] > 0.5)) continue;
    // A confident canned gesture lights the hand up and an unrecognised pose
    // stays dim, which is what makes "the engine understood me" legible.
    const score = Math.min(1, Math.max(0, hands[base + 3] || 0));
    present.push({ lm: base + 4, gain: 0.55 + 0.45 * score });
  }

  for (const hand of present) {
    for (const [a, b] of BONES) {
      const ia = hand.lm + a * 3;
      const ib = hand.lm + b * 3;
      if (!usable(ia) || !usable(ib)) continue;
      push(hands[ia], hands[ia + 1], hand.gain * 0.5);
      push(hands[ib], hands[ib + 1], hand.gain * 0.5);
    }
  }
  // Both endpoints of a bone are pushed together, so the line range is always
  // even and the joint range that follows starts on a fresh segment.
  const lineVertices = n;

  for (const hand of present) {
    for (let k = 0; k < HAND_LANDMARKS; k++) {
      const i = hand.lm + k * 3;
      if (!usable(i)) continue;
      push(hands[i], hands[i + 1], hand.gain * (TIPS.includes(k) ? 1.0 : 0.45));
    }
  }

  return { lineVertices, pointVertices: n - lineVertices };
}
