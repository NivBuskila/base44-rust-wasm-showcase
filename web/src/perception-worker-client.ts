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
/** Model downloads can be slow, but a silent worker must not hang forever. */
const INIT_TIMEOUT_MS = 60_000;
const FRAME_TIMEOUT_MS = 5_000;
const MAX_DECODE_FAILURES = 3;

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
  private initTimer: ReturnType<typeof setTimeout> | null = null;
  private decodeFailures = 0;
  private rejectInit: ((reason: Error) => void) | null = null;

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
      this.fail(`Perception worker failed: ${event.message || 'unknown error'}`);
    };
    this.worker.onmessageerror = () => this.fail('Perception worker sent an unreadable message.');
  }

  get status(): PerceptionStatus {
    return this.state;
  }

  /** Resolves on readiness; rejects on errors, cancellation or a silent boot. */
  init(): Promise<void> {
    if (this.closed) return Promise.reject(new Error('Perception was closed.'));
    this.booting ??= new Promise<void>((resolve, reject) => {
      this.rejectInit = reject;
      this.resolveInit = resolve;
      this.initTimer = setTimeout(() => this.fail('Perception worker timed out loading models.'), INIT_TIMEOUT_MS);
      try {
        this.post({ type: 'init' });
      } catch (err) {
        this.fail(`Perception worker could not start: ${String(err)}`);
      }
    });
    return this.booting;
  }

  private resolveInit: (() => void) | null = null;

  private finishInit(): void {
    if (this.initTimer !== null) clearTimeout(this.initTimer);
    this.initTimer = null;
    this.resolveInit = null;
    this.rejectInit = null;
  }

  private fail(reason: string): void {
    if (this.closed) return;
    this.state = { kind: 'unavailable', reason };
    this.rejectInit?.(new Error(reason));
    this.finishInit();
    this.inFlight = false;
    this.worker.terminate();
  }

  process(video: HTMLVideoElement, timestampMs: number): PerceptionFrame | null {
    if (this.inFlight && performance.now() - this.submittedAt > FRAME_TIMEOUT_MS) {
      this.fail('Perception worker stopped responding to frames.');
    }
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
        if (this.closed || this.state.kind !== 'ready') {
          bitmap.close();
          return;
        }
        this.decodeFailures = 0;
        this.decodeMs = performance.now() - decodeStarted;
        this.inFlight = true;
        this.submittedAt = performance.now();
        const spare = this.spareMask;
        this.spareMask = null;
        try {
          this.post({ type: 'frame', bitmap, timestampMs, maskBuf: spare }, [
            bitmap,
            ...(spare ? [spare] : []),
          ]);
        } catch (err) {
          bitmap.close();
          this.fail(`Perception worker rejected a frame: ${String(err)}`);
        }
      })
      .catch((err) => {
        this.decoding = false;
        if (this.closed || this.state.kind !== 'ready') return;
        // A transient decode failure can happen when a tab is hidden. Repeated
        // failures with a live video need the inline path instead.
        if (++this.decodeFailures >= MAX_DECODE_FAILURES) {
          this.fail(`Camera frame decode failed repeatedly: ${String(err)}`);
        } else {
          console.warn('[aether] could not decode a camera frame', err);
        }
      });
  }

  private receive(message: FromWorker): void {
    if (this.closed || this.state.kind === 'unavailable') return;
    if (message.type === 'status') {
      if (message.status.kind === 'unavailable') {
        this.fail(message.status.reason);
      } else if (message.status.kind === 'ready') {
        this.state = message.status;
        this.resolveInit?.();
        this.finishInit();
      }
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
    this.state = { kind: 'unavailable', reason: 'Perception was closed.' };
    this.rejectInit?.(new Error(this.state.reason));
    this.finishInit();
    // Termination also works when the worker is stuck in a synchronous pass;
    // the browser releases its graphs and GL context with the worker.
    this.worker.terminate();
    this.pending = null;
  }

  private post(message: ToWorker, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }
}
