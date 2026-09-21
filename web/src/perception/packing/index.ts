/**
 * The DOM-free, MediaPipe-free half of perception: reading a recogniser result
 * into slots and packing it into the buffers the engine reads.
 *
 * Everything here is pure or owns nothing but a reusable buffer, so it can be
 * reasoned about (and tested) without a camera, a model or a GPU. The split is
 * by stage, and the two subtleties each live with the code that implements them:
 *
 * - `detections.ts` reads and validates one result, and is the only place hand
 *   landmarks are mirrored (`mask.ts` deliberately does not mirror, because
 *   `push_mask` does it itself);
 * - `slots.ts` is the sticky 2x2 assignment that keeps a hand in the slot it
 *   had, so a latched spell never jumps hands;
 * - `buffers.ts` writes the packed hand/pose buffers and the graph timestamp;
 * - `mask.ts` box-downscales the silhouette to the engine's staging size.
 */

export type { HandDetection, HandResultLike } from './detections';
export { emptyDetections, readDetections } from './detections';
export { HandSlots } from './slots';
export { nextTimestamp, packHands, packPose } from './buffers';
export { MaskScaler } from './mask';
