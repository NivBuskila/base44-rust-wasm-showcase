/**
 * Messages between the main thread and `perception.worker.ts`.
 *
 * Shared so both ends fail to compile together rather than drifting into a
 * silent mismatch, which in a worker surfaces only as perception never
 * producing a frame.
 */

import type { PerceptionStatus } from './types';

/** Main thread -> worker. */
export type ToWorker =
  | { type: 'init' }
  /**
   * One decoded camera frame to run the models on.
   *
   * `maskBuf` hands back the buffer of the previous result so the worker can
   * refill it instead of allocating: the mask is up to 320x320 floats, and at
   * 30 Hz a fresh one per inference is ~12 MB/s of garbage.
   */
  | { type: 'frame'; bitmap: ImageBitmap; timestampMs: number; maskBuf: ArrayBuffer | null }
  | { type: 'close' };

/** Worker -> main thread. */
export type FromWorker =
  | { type: 'status'; status: PerceptionStatus }
  /** Inference produced landmarks. Buffers are transferred, not copied. */
  | {
      type: 'result';
      hands: ArrayBuffer;
      pose: ArrayBuffer;
      mask: ArrayBuffer | null;
      maskWidth: number;
      maskHeight: number;
      latencyMs: number;
      anyHand: boolean;
    }
  /**
   * The frame produced nothing (limiter, or a failed pass). Sent anyway so the
   * main thread can clear its in-flight flag and submit the next frame; without
   * it, one skipped inference stalls perception permanently.
   */
  | { type: 'skip'; maskBuf: ArrayBuffer | null };
