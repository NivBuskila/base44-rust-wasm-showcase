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
 * Three things here are subtle enough to be worth stating up front, because
 * each one is a bug that looks like something else:
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
 * 3. **`process` is called from the render loop.** It is synchronous, never
 *    throws, and returns early before touching MediaPipe when the video frame
 *    has not advanced — the graph rejects a repeated timestamp.
 *
 * The MediaPipe glue is imported dynamically: `main.ts` only constructs this
 * class when a camera exists, so an ambient-mode session never pays for the
 * chunk.
 */

import {
  HAND_BUFFER,
  HAND_LANDMARKS,
  HAND_STRIDE,
  HANDS,
  LM,
  MASK_IN_MAX_DIM,
  POSE_LANDMARKS,
  POSE_STRIDE,
  gestureId,
} from './constants';
import type { MaskFrame, PerceptionFrame, PerceptionSource, PerceptionStatus } from './types';
import type {
  GestureRecognizer,
  GestureRecognizerResult,
  NormalizedLandmark,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';

/** Vendored WASM runtime, served from our own origin by `sync-mp-assets.mjs`. */
const WASM_PATH = '/mp-wasm';

/**
 * Public copy of the same runtime, used when the vendored one is not there.
 *
 * `sync-mp-assets.mjs` copies it out of node_modules on install, so it is
 * missing whenever that has not run: a partial install, a network that blocks
 * the npm registry, or any deployment of `dist/` that did not carry the 24 MB
 * along. The models already fall back this way and the runtime did not, which
 * meant perception died with a 404 while a byte-identical copy sat one
 * hostname away. The version is pinned to the installed package so the
 * fallback can never be a different build than the loader expects.
 */
const WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';

/** Google's public model host, used when the local copy is absent. */
const CDN = 'https://storage.googleapis.com/mediapipe-models';

/** Local path first, CDN second: works offline after one `npm run fetch:models`. */
const HAND_MODEL = {
  local: '/models/gesture_recognizer.task',
  remote: `${CDN}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`,
};
const POSE_MODEL = {
  local: '/models/pose_landmarker_lite.task',
  remote: `${CDN}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
};

/**
 * Smallest plausible `.task` bundle. A dev server that answers unknown paths
 * with `index.html` returns 200 for a missing model, so the probe checks the
 * size as well: the real bundles are 5-8 MB.
 */
const MIN_MODEL_BYTES = 1_000_000;

/** Consecutive inference failures before perception gives up for good. */
const MAX_FAILURES = 10;

/**
 * Inference cost, in ms, above which perception starts rationing itself.
 *
 * `tasks-vision` is synchronous: a pass blocks the main thread, and with it the
 * render loop. Up to about a frame and a half the render loop's own 30 Hz gate
 * is the binding constraint and rationing would only throw away landmarks the
 * caller asked for — so the limiter stays out of the way. Past it, calling as
 * often as the loop asks pins the whole app at a few fps: the fluid stops, the
 * particles stop, and the gesture being made is lost anyway.
 */
const BUDGET_MS = 50;

/** Share of wall-clock time inference may consume once past `BUDGET_MS`. */
const MAX_DUTY = 0.35;

/** Gain of the inference-cost estimate; ~4 inferences to track a change. */
const COST_GAIN = 0.25;

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
 * `main.ts` silently drops anything larger than the engine's staging buffer,
 * so an oversized mask would mean no body collision at all. Averaging rather
 * than point-sampling matters here: the mask is a coverage field the engine
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

function describe(err: unknown): string {
  if (err instanceof Error) return err.message;
  const text = String(err);
  return text.length > 200 ? `${text.slice(0, 200)}…` : text;
}

/**
 * Probes for a locally served model and falls back to the CDN, so the app
 * works out of the box and offline after one `npm run fetch:models`.
 *
 * `HEAD` rather than `GET` so the 8 MB body is never pulled twice. A 200 is not
 * proof on its own: a dev server with an SPA fallback answers a missing model
 * with `index.html` and status 200, which is why the content type is checked
 * too. The size is only checked when the server declares one — rejecting a
 * response for lacking `content-length` would fail the local path on servers
 * that stream it.
 */
export async function resolveModel(
  model: { local: string; remote: string },
  probe: typeof fetch = fetch,
): Promise<string> {
  try {
    const res = await probe(model.local, { method: 'HEAD', cache: 'no-store' });
    const declared = res.headers.get('content-length');
    const length = declared === null ? Number.POSITIVE_INFINITY : Number(declared);
    const type = res.headers.get('content-type') ?? '';
    if (res.ok && length >= MIN_MODEL_BYTES && !type.includes('html')) return model.local;
  } catch {
    // A blocked or failed probe is not an error; the CDN is the answer.
  }
  return model.remote;
}

/**
 * Probes for the vendored WASM runtime and falls back to jsdelivr.
 *
 * Mirrors [`resolveModel`], including why a 200 is not proof: a dev server with
 * an SPA fallback answers a missing file with `index.html`. The loader script
 * is the thing probed because `FilesetResolver` fetches it first, so if it is
 * there the binary beside it is too.
 */
export async function resolveWasmPath(probe: typeof fetch = fetch): Promise<string> {
  try {
    const res = await probe(`${WASM_PATH}/vision_wasm_internal.js`, {
      method: 'HEAD',
      cache: 'no-store',
    });
    const type = res.headers.get('content-type') ?? '';
    if (res.ok && !type.includes('html')) return WASM_PATH;
  } catch {
    // A blocked or failed probe is not an error; the CDN is the answer.
  }
  return WASM_CDN;
}

/**
 * Demotes the runtime's `INFO:` chatter for the duration of a load, and
 * returns the undo.
 *
 * MediaPipe pipes its stderr to `console.error`, and building the graphs writes
 * `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` — not an error by
 * any reading, but enough to fail the "no console errors" bar the headless
 * suite holds this app to. Only lines the library itself marks INFO are
 * demoted; everything else goes straight through, so a real failure inside the
 * WASM runtime is still loud.
 */
function muteInfoLogs(): () => void {
  const original = console.error;
  const filtered = (...args: unknown[]): void => {
    if (typeof args[0] === 'string' && args[0].startsWith('INFO:')) console.info(...args);
    else original.apply(console, args);
  };
  console.error = filtered;
  return () => {
    // Only undo our own wrapper: another module may have wrapped console while
    // 14 MB of models were in flight, and clobbering it would be worse.
    if (console.error === filtered) console.error = original;
  };
}

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
  private readonly rationInference: boolean;

  constructor(options: { rationInference?: boolean } = {}) {
    this.rationInference = options.rationInference ?? true;
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
  private costMs = 0;
  private nextRunMs = 0;

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
      costMs: this.costMs,
      gapMs: this.gap(),
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
    const unmute = muteInfoLogs();
    let reason: string;
    try {
      const vision = await import('@mediapipe/tasks-vision');
      const fileset = await vision.FilesetResolver.forVisionTasks(await resolveWasmPath());
      const [handModel, poseModel] = await Promise.all([
        resolveModel(HAND_MODEL),
        resolveModel(POSE_MODEL),
      ]);
      this.handModel = handModel;
      this.poseModel = poseModel;

      const failures: string[] = [];
      // The GPU delegate fails on machines with no usable WebGL2, and the
      // failure surfaces as a rejected task construction rather than a flag to
      // query — so trying it and falling back *is* the capability check.
      for (const delegate of ['GPU', 'CPU'] as const) {
        const built = await Promise.allSettled([
          vision.GestureRecognizer.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: handModel, delegate },
            runningMode: 'VIDEO',
            numHands: HANDS,
          }),
          vision.PoseLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: poseModel, delegate },
            runningMode: 'VIDEO',
            numPoses: 1,
            outputSegmentationMasks: true,
          }),
        ]);
        const [recognizer, landmarker] = built;
        if (recognizer.status === 'fulfilled' && landmarker.status === 'fulfilled') {
          this.recognizer = recognizer.value;
          this.landmarker = landmarker.value;
          this.delegate = delegate;
          this.loadMs = performance.now() - started;
          // A `close()` can land while 14 MB of models are still in flight.
          // Adopting the tasks anyway would resurrect a source the caller has
          // already released — inference would resume, and the two WASM graphs
          // and their GL contexts would never be freed, since nothing holds a
          // reference to close them a second time.
          if (this.closed) {
            this.release();
            throw new Error('Perception was closed while it was loading.');
          }
          this.state = { kind: 'ready', delegate };
          return;
        }
        // One half can succeed while the other fails; that half owns a WASM
        // graph and a WebGL context, and leaking them would poison the retry.
        const why: string[] = [];
        for (const task of built) {
          if (task.status === 'fulfilled') task.value.close();
          else why.push(describe(task.reason));
        }
        failures.push(`${delegate}: ${why.join(' / ')}`);
        console.warn(`[aether] ${delegate} delegate unavailable: ${why.join(' / ')}`);
      }
      reason = `no usable delegate (${failures.join('; ')})`;
    } catch (err) {
      reason = describe(err);
    } finally {
      unmute();
    }

    this.loadMs = performance.now() - started;
    this.state = { kind: 'unavailable', reason };
    throw new Error(reason);
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
    if (performance.now() < this.nextRunMs) return null;

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
    // The first pass through each graph also compiles shaders and allocates
    // textures; it routinely costs twenty times the steady state, and folding
    // it into the estimate would ration the whole first ten seconds of the
    // session for no reason.
    if (this.inferences > 0) this.costMs += (latency - this.costMs) * COST_GAIN;
    // Measured from the moment the main thread is free again, so the gap is
    // idle time rather than a total period.
    this.nextRunMs = finished + this.gap();

    this.failures = 0;
    this.inferences++;
    this.frame.mask = mask;
    this.frame.latencyMs = latency;
    this.frame.anyHand = count > 0;
    return this.frame;
  }

  /** Idle time the duty-cycle limiter owes after an inference costing `costMs`. */
  private gap(): number {
    if (!this.rationInference) return 0;
    return this.costMs <= BUDGET_MS ? 0 : this.costMs * (1 / MAX_DUTY - 1);
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
    for (const task of [this.recognizer, this.landmarker]) {
      try {
        task?.close();
      } catch (err) {
        // Teardown runs on the page-unload path; a throw here would take the
        // rest of the cleanup with it.
        console.warn('[aether] closing a vision task failed', err);
      }
    }
    this.recognizer = null;
    this.landmarker = null;
    this.slots.reset();
    this.costMs = 0;
    this.nextRunMs = 0;
    this.hands.fill(0);
    this.pose.fill(0);
    this.frame.mask = null;
    this.frame.anyHand = false;
    this.lastVideoTime = -1;
  }
}
