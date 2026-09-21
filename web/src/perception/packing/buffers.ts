/**
 * Packing the validated detections into the flat buffers the engine reads, plus
 * the graph's timestamp rule. Pure: no MediaPipe types beyond a landmark shape,
 * no DOM, no state.
 */

import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

import { HAND_STRIDE, HANDS, POSE_LANDMARKS } from '../../constants';
import type { HandDetection } from './detections';
import { unit } from './detections';

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
