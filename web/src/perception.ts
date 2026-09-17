/**
 * MediaPipe Tasks-Vision perception.
 *
 * STUB — `init` rejects, which `main.ts` treats as a capability downgrade
 * rather than a failure: the app keeps running on the Rust optical-flow path.
 * The real implementation loads `GestureRecognizer` (hand landmarks *and*
 * canned gesture labels in one model) and `PoseLandmarker` with
 * `outputSegmentationMasks: true` (body joints *and* the silhouette in one),
 * then packs both into the buffers `constants.ts` documents.
 */

import { HAND_BUFFER, POSE_STRIDE } from './constants';
import type { PerceptionFrame, PerceptionSource, PerceptionStatus } from './types';

export class MediaPipePerception implements PerceptionSource {
  private state: PerceptionStatus = { kind: 'loading' };
  private readonly hands = new Float32Array(HAND_BUFFER);
  private readonly pose = new Float32Array(POSE_STRIDE);

  get status(): PerceptionStatus {
    return this.state;
  }

  async init(): Promise<void> {
    this.state = { kind: 'unavailable', reason: 'Perception not implemented yet.' };
    throw new Error('MediaPipePerception is not implemented yet.');
  }

  process(_video: HTMLVideoElement, _timestampMs: number): PerceptionFrame | null {
    return null;
  }

  close(): void {
    this.hands.fill(0);
    this.pose.fill(0);
  }
}
