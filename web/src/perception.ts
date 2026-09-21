/**
 * MediaPipe Tasks-Vision perception: hands, pose and silhouette.
 *
 * Two models, chosen so one inference pass per frame yields everything the
 * engine consumes:
 * - `GestureRecognizer` gives hand landmarks *and* the canned gesture label,
 *   so there is no second classifier to run over the landmarks.
 * - `PoseLandmarker` with `outputSegmentationMasks` gives body joints *and* the
 *   silhouette, so body collision costs no extra model.
 *
 * This file owns the *session*: loading the graphs, running them, and the
 * duty-cycle limiter. The result-shaping half (mirroring, sticky slots, the
 * packed buffers, the mask downscale) is `perception/packing.ts`, and asset
 * resolution is `perception/assets.ts`.
 *
 * One thing about the call site is worth stating up front: **`process` is
 * called from the render loop.** It is synchronous, never throws, and returns
 * early before touching MediaPipe when the video frame has not advanced — the
 * graph rejects a repeated timestamp.
 *
 * The MediaPipe glue is imported dynamically: the app only constructs this
 * class when a camera exists, so an ambient-mode session never pays for the
 * chunk.
 */

import { HAND_BUFFER, POSE_STRIDE } from './constants';
import { describe } from './perception/assets';
import { closeGraphs, loadGraphs } from './perception/graphs';
import { DutyLimiter } from './perception/limiter';
import {
  HandSlots,
  MaskScaler,
  emptyDetections,
  nextTimestamp,
  packHands,
  packPose,
  readDetections,
} from './perception/packing';
import type { MaskFrame, PerceptionFrame, PerceptionSource, PerceptionStatus } from './types';
import type { GestureRecognizer, PoseLandmarker } from '@mediapipe/tasks-vision';

/** Consecutive inference failures before perception gives up for good. */
const MAX_FAILURES = 10;

/** Live numbers about the loaded models, for the HUD and the headless suite. */
export interface PerceptionDiagnostics {
  delegate: 'GPU' | 'CPU' | null;
  /** Wall-clock ms spent in `init`, including the model downloads. */
  loadMs: number;
  handModel: string;
  poseModel: string;
  /** Inferences that produced a frame. */
  inferences: number;
  /** Latency of the most recent inference, in ms. */
  lastLatencyMs: number;
  /** Mask size as the model emitted it, before any downscale. */
  maskWidth: number;
  maskHeight: number;
  /** Smoothed inference cost driving the duty-cycle limiter, in ms. */
  costMs: number;
  /** Idle time the limiter is currently inserting between inferences, in ms. */
  gapMs: number;
  failures: number;
}

export class MediaPipePerception implements PerceptionSource {
  /**
   * When false, the duty-cycle limiter is disabled.
   *
   * The limiter exists for one reason: inference is synchronous, so on the main
   * thread an expensive pass stalls the render loop and rationing is the only
   * way to keep the app alive. Inside `perception.worker.ts` none of that
   * applies — the worker blocks nobody, and its client already keeps exactly
   * one inference in flight. Rationing there only inserts idle time between
   * landmarks: at a 60 ms pass it drops gestures from ~16 Hz to ~6 Hz, felt
   * directly as hands lagging behind the fluid they are supposed to push.
   */
  private readonly limiter: DutyLimiter;

  constructor(options: { rationInference?: boolean } = {}) {
    this.limiter = new DutyLimiter(options.rationInference ?? true);
  }

  private state: PerceptionStatus = { kind: 'loading' };
  private readonly hands = new Float32Array(HAND_BUFFER);
  private readonly pose = new Float32Array(POSE_STRIDE);

  private recognizer: GestureRecognizer | null = null;
  private landmarker: PoseLandmarker | null = null;

  private readonly dets = emptyDetections();
  private readonly slots = new HandSlots();
  private readonly scaler = new MaskScaler();

  private readonly frame: PerceptionFrame = {
    hands: this.hands,
    pose: this.pose,
    mask: null,
    latencyMs: 0,
    anyHand: false,
  };

  private booting: Promise<void> | null = null;
  private closed = false;
  private lastVideoTime = -1;
  private lastStamp = 0;
  private failures = 0;

  private delegate: 'GPU' | 'CPU' | null = null;
  private loadMs = 0;
  private handModel = '';
  private poseModel = '';
  private inferences = 0;
  private maskWidth = 0;
  private maskHeight = 0;

  get status(): PerceptionStatus {
    return this.state;
  }

  get diagnostics(): PerceptionDiagnostics {
    return {
      delegate: this.delegate,
      loadMs: this.loadMs,
      handModel: this.handModel,
      poseModel: this.poseModel,
      inferences: this.inferences,
      lastLatencyMs: this.frame.latencyMs,
      maskWidth: this.maskWidth,
      maskHeight: this.maskHeight,
      costMs: this.limiter.costMs,
      gapMs: this.limiter.gapMs,
      failures: this.failures,
    };
  }

  /** Loads the runtime and both models. Idempotent; rejects if nothing loads. */
  init(): Promise<void> {
    this.booting ??= this.load();
    return this.booting;
  }

  private async load(): Promise<void> {
    const started = performance.now();
    try {
      const graphs = await loadGraphs();
      this.handModel = graphs.handModel;
      this.poseModel = graphs.poseModel;
      this.delegate = graphs.delegate;
      this.loadMs = performance.now() - started;
      // A `close()` can land while 14 MB of models are still in flight. Adopting
      // the tasks anyway would resurrect a source the caller has already
      // released — inference would resume, and the two WASM graphs and their GL
      // contexts would never be freed, since nothing holds a reference to close
      // them a second time.
      if (this.closed) {
        closeGraphs(graphs.recognizer, graphs.landmarker);
        throw new Error('Perception was closed while it was loading.');
      }
      this.recognizer = graphs.recognizer;
      this.landmarker = graphs.landmarker;
      this.state = { kind: 'ready', delegate: graphs.delegate };
      return;
    } catch (err) {
      this.loadMs = performance.now() - started;
      const reason = describe(err);
      this.state = { kind: 'unavailable', reason };
      throw new Error(reason);
    }
  }

  /**
   * Runs both models on the current video frame.
   *
   * Returns null — cheaply, before touching MediaPipe — when perception is not
   * ready, the duty-cycle limiter is holding it back, the video has no pixels
   * yet, or `currentTime` has not advanced. The last case is the common one:
   * the render loop runs faster than the camera, and re-submitting a frame is
   * both wasted work and a graph error.
   */
  process(video: HTMLVideoElement, timestampMs: number): PerceptionFrame | null {
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return null;

    const videoTime = video.currentTime;
    if (videoTime === this.lastVideoTime) return null;
    this.lastVideoTime = videoTime;

    return this.processSource(video, timestampMs);
  }

  /**
   * Runs both models on an already-decoded frame.
   *
   * Split out of `process` so `perception.worker.ts` can feed the `ImageBitmap`
   * it was handed instead of a video element it does not have. The freshness
   * check stays in `process`, because only whoever owns the video can tell
   * whether `currentTime` advanced.
   */
  processSource(
    source: ImageBitmap | HTMLVideoElement,
    timestampMs: number,
  ): PerceptionFrame | null {
    const recognizer = this.recognizer;
    const landmarker = this.landmarker;
    if (!recognizer || !landmarker) return null;
    if (!this.limiter.ready(performance.now())) return null;

    const stamp = nextTimestamp(this.lastStamp, timestampMs);
    this.lastStamp = stamp;

    // Spans the mask readback as well as the two models: the GPU->CPU transfer
    // is part of what the frame pays, and hiding it from the HUD would make a
    // frame-time regression impossible to attribute.
    const started = performance.now();
    let count = 0;
    let mask: MaskFrame | null = null;
    try {
      const handResult = recognizer.recognizeForVideo(source, stamp);
      const poseResult = landmarker.detectForVideo(source, stamp);
      try {
        count = readDetections(handResult, this.dets);
        // An empty pose buffer is a result, not a non-result: the engine needs
        // it to run its release timers instead of coasting on stale landmarks.
        packPose(this.pose, poseResult.landmarks[0]);

        const raw = poseResult.segmentationMasks?.[0];
        this.maskWidth = 0;
        this.maskHeight = 0;
        if (raw) {
          this.maskWidth = raw.width;
          this.maskHeight = raw.height;
          // One transfer per inference: `getAsFloat32Array` drags the mask off
          // the GPU, and calling it twice doubles the most expensive part of
          // the frame.
          mask = this.scaler.fit(raw.getAsFloat32Array(), raw.width, raw.height);
        }
      } finally {
        // Closes every mask the result holds. MediaPipe is explicit that a mask
        // left open leaks its GPU texture, and at 30 Hz that is a tab crash.
        poseResult.close();
      }
    } catch (err) {
      this.onFailure(err);
      return null;
    }

    packHands(this.hands, this.dets, this.slots.assign(this.dets, count, stamp));

    const finished = performance.now();
    const latency = finished - started;
    this.limiter.charge(latency, finished, this.inferences === 0);

    this.failures = 0;
    this.inferences++;
    this.frame.mask = mask;
    this.frame.latencyMs = latency;
    this.frame.anyHand = count > 0;
    return this.frame;
  }

  /**
   * A failing inference is survivable — a dropped frame — until it is not.
   * Some delegate losses (a GPU context reset) fail every call from then on, so
   * after `MAX_FAILURES` in a row perception shuts itself down and leaves the
   * optical-flow path to drive the fluid.
   */
  private onFailure(err: unknown): void {
    this.failures++;
    if (this.failures === 1) console.warn('[aether] inference failed', err);
    if (this.failures >= MAX_FAILURES) {
      const reason = `inference failed ${this.failures} times in a row: ${describe(err)}`;
      console.error(`[aether] ${reason}`);
      this.state = { kind: 'unavailable', reason };
      this.release();
    }
  }

  /** Closes both tasks and clears every buffer. Safe to call twice. */
  close(): void {
    this.closed = true;
    if (this.state.kind === 'ready') {
      this.state = { kind: 'unavailable', reason: 'Perception was closed.' };
    }
    this.release();
  }

  private release(): void {
    closeGraphs(this.recognizer, this.landmarker);
    this.recognizer = null;
    this.landmarker = null;
    this.slots.reset();
    this.limiter.reset();
    this.hands.fill(0);
    this.pose.fill(0);
    this.frame.mask = null;
    this.frame.anyHand = false;
    this.lastVideoTime = -1;
  }
}
