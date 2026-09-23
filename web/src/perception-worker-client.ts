/**
 * Main-thread half of off-thread perception.
 *
 * Implements `PerceptionSource` so `main.ts` cannot tell the difference, but
 * where `MediaPipePerception.process` runs the models inline, this one submits
 * the frame and returns the *previous* result. `process` therefore stays
 * synchronous and costs the render loop only a bitmap decode.
 *
 * One inference is in flight at a time. Queueing more would not add
 * information — the models are slower than the camera, so a queued frame is
 * already stale by the time it runs — and would grow an unbounded backlog of
 * GPU-backed bitmaps.
 */

import type { FromWorker, ToWorker } from './perception-protocol';
import type { MaskFrame, PerceptionFrame, PerceptionSource, PerceptionStatus } from './types';

/**
 * Width the camera frame is decoded down to before inference, in pixels.
 *
 * The camera runs at 1280x720 but MediaPipe's hand and pose graphs resize their
 * input to a couple of hundred pixels on the way in, so the extra detail is
 * thrown away after being paid for twice: once in the main thread's decode and
 * once in the worker's own rescale. Decoding straight to this width cuts both,
 * which is latency off the front of every gesture — and it stays comfortably
 * above the models' own input size, so the landmarks do not move.
 */
const INFERENCE_WIDTH = 480;

export class WorkerPerception implements PerceptionSource {
  private readonly worker: Worker;
  private state: PerceptionStatus = { kind: 'loading' };

  private booting: Promise<void> | null = null;
  private closed = false;

  /** True from submitting a frame until the worker answers about it. */
  private inFlight = false;

  /** True while `createImageBitmap` is still resolving. */
  private decoding = false;
  private lastVideoTime = -1;
  private decodeMs = 0;
  private submittedAt = 0;

  /** Result waiting to be handed to the caller; null once consumed. */
  private pending: PerceptionFrame | null = null;

  private mask: MaskFrame = { data: new Float32Array(0), width: 0, height: 0 };

  /**
   * Mask buffer of the result currently in `pending`.
   *
   * It cannot be lent back while the caller might still read it, so it only
   * becomes `spareMask` once `process` has handed the frame over.
   */
  private resultMask: ArrayBuffer | null = null;

  /** Mask buffer free to lend back to the worker with the next frame. */
  private spareMask: ArrayBuffer | null = null;

  constructor() {
    this.worker = new Worker(new URL('./perception.worker.ts', import.meta.url), {
      type: 'module',
      name: 'aether-perception',
    });
    this.worker.onmessage = (event: MessageEvent<FromWorker>) => this.receive(event.data);
    this.worker.onerror = (event) => {
      // A worker that cannot even parse its module never answers `init`, so
      // without this the boot would hang on the loading status forever.
      this.state = {
        kind: 'unavailable',
        reason: `Perception worker failed: ${event.message || 'unknown error'}`,
      };
      this.inFlight = false;
    };
  }

  get status(): PerceptionStatus {
    return this.state;
  }

  /** Resolves once the worker reports a terminal status; rejects if unusable. */
  init(): Promise<void> {
    this.booting ??= new Promise<void>((resolve, reject) => {
      const settle = () => {
        if (this.state.kind === 'loading') return false;
        if (this.state.kind === 'ready') resolve();
        else reject(new Error(this.state.reason));
        return true;
      };

      this.onStatusChange = settle;
      this.post({ type: 'init' });
    });
    return this.booting;
  }

  /** Set by `init` to notice the status the worker reports back. */
  private onStatusChange: (() => boolean) | null = null;

  process(video: HTMLVideoElement, timestampMs: number): PerceptionFrame | null {
    this.submit(video, timestampMs);

    // Handing the same result out twice would make the engine re-apply a
    // gesture it has already seen, and its release timers would never fire.
    const frame = this.pending;
    this.pending = null;
    if (frame) {
      // The caller reads the mask synchronously and is done with it by the next
      // render frame, which is the earliest `submit` can lend the buffer out.
      this.spareMask = this.resultMask;
      this.resultMask = null;
    }
    return frame;
  }

  /**
   * Decodes and sends the current video frame if the worker is free.
   *
   * `createImageBitmap` is asynchronous, so the decode may still be running
   * when the next render frame calls in; `decoding` keeps that from queueing a
   * second one.
   */
  private submit(video: HTMLVideoElement, timestampMs: number): void {
    if (this.closed || this.inFlight || this.decoding) return;
    if (this.state.kind !== 'ready') return;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;

    // Same freshness rule as the inline path: the graphs reject a repeated
    // timestamp, and re-running on identical pixels buys nothing.
    if (video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;

    this.decoding = true;
    const decodeStarted = performance.now();
    const scale = Math.min(1, INFERENCE_WIDTH / video.videoWidth);
    createImageBitmap(video, {
      resizeWidth: Math.round(video.videoWidth * scale),
      resizeHeight: Math.round(video.videoHeight * scale),
      // `low` is a box filter, which is what a downscale for a CNN wants; the
      // sharper kernels cost more and buy the model nothing.
      resizeQuality: 'low',
    })
      .then((bitmap) => {
        this.decoding = false;
        if (this.closed) {
          bitmap.close();
          return;
        }
        this.decodeMs = performance.now() - decodeStarted;
        this.inFlight = true;
        this.submittedAt = performance.now();
        const spare = this.spareMask;
        this.spareMask = null;
        this.post({ type: 'frame', bitmap, timestampMs, maskBuf: spare }, [
          bitmap,
          ...(spare ? [spare] : []),
        ]);
      })
      .catch((err) => {
        this.decoding = false;
        // A decode failure is a dropped frame, not a dead source: it happens
        // when the track ends or the tab is hidden mid-call.
        console.warn('[aether] could not decode a camera frame', err);
      });
  }

  private receive(message: FromWorker): void {
    if (message.type === 'status') {
      this.state = message.status;
      if (this.onStatusChange?.()) this.onStatusChange = null;
      return;
    }

    this.inFlight = false;

    if (message.type === 'skip') {
      if (message.maskBuf) this.spareMask = message.maskBuf;
      return;
    }

    const hands = new Float32Array(message.hands);
    const pose = new Float32Array(message.pose);

    let mask: MaskFrame | null = null;
    if (message.mask) {
      const size = message.maskWidth * message.maskHeight;
      this.mask = {
        data: new Float32Array(message.mask, 0, size),
        width: message.maskWidth,
        height: message.maskHeight,
      };
      mask = this.mask;
      this.resultMask = message.mask;
    }

    this.pending = {
      hands,
      pose,
      mask,
      latencyMs: message.latencyMs,
      anyHand: message.anyHand,
      decodeMs: this.decodeMs,
      workerRoundTripMs: performance.now() - this.submittedAt,
    };
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    if (this.state.kind === 'ready') {
      this.state = { kind: 'unavailable', reason: 'Perception was closed.' };
    }
    // Asks the worker to release the two WASM graphs before it goes away;
    // `terminate` alone would drop them without closing their GL contexts.
    this.post({ type: 'close' });
    this.pending = null;
  }

  private post(message: ToWorker, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }
}
