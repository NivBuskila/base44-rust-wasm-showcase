/**
 * Perception pacing: how often the vision models are allowed to run, and what
 * happens when they are too slow for this device.
 *
 * Owns the perception source's whole lifetime (loading it off the critical path,
 * the worker→inline fallback, giving up on it) so the frame loop only has to
 * call `pump` once per frame and read `hands`/`status`.
 */

import type { AetherEngine } from './wasm/aether';
import { MediaPipePerception } from './perception';
import { WorkerPerception } from './perception-worker-client';
import type { PerceptionSource, PerceptionStatus } from './types';

/** Floor on the interval between inferences, in ms. */
const PERCEPTION_MIN_INTERVAL_MS = 1000 / 30;

/**
 * Ceiling on the interval, in ms. Past this, gestures are too stale to feel
 * connected to the hand.
 *
 * Reaching it is not a stable operating point but a verdict: if the duty
 * target below cannot be met *within* this interval, inference is too
 * expensive for this device, full stop. Capping the interval and carrying on
 * would silently break the duty guarantee — measured at 61% of wall time on a
 * 1224 ms inference, which is what `PERCEPTION_GIVE_UP_STRIKES` exists to
 * catch.
 */
const PERCEPTION_MAX_INTERVAL_MS = 2000;

/**
 * Consecutive inferences too slow to fit the duty budget before perception is
 * switched off for the session.
 *
 * More than one because the first call includes MediaPipe's own warm-up and is
 * not representative, and a single slow frame should not cost the user hand
 * tracking for the rest of the session. When it does trip, the app keeps
 * working on the Rust optical-flow path and the HUD says why — which is a far
 * better outcome than a technically-alive gesture layer eating two thirds of
 * every frame.
 */
const PERCEPTION_GIVE_UP_STRIKES = 3;

/**
 * Share of wall-clock time inference is allowed to consume.
 *
 * `PerceptionSource.process` is synchronous — MediaPipe's video API has no
 * async form — so its cost lands directly in the frame it runs on. On a real
 * GPU that is 10-20 ms and invisible at a 30 Hz cadence. On software
 * rasterisation it measured **753 ms**, which at a fixed 30 Hz cadence means
 * every frame tries to run an inference that takes 22 frames, and the render
 * loop collapses to inference speed.
 *
 * So the cadence is derived from measured latency rather than assumed: spend at
 * most this fraction of the time in inference and leave the rest to the
 * simulation. Aether stays responsive through the Rust optical-flow path, which
 * is exactly the case that path exists for; gestures just update more slowly.
 */
const PERCEPTION_DUTY = 0.3;

export interface PerceptionPumpDeps {
  engine: AetherEngine;
  /** The video to infer from, or null when there is no usable camera frame. */
  frame(): HTMLVideoElement | null;
  /** The engine's segmentation-mask view, rebuilt by the owner as needed. */
  mask(): Float32Array;
}

export class PerceptionPump {
  private source: PerceptionSource | null = null;
  status: PerceptionStatus = { kind: 'loading' };
  /**
   * The most recent packed hand buffer, kept for the renderer's landmark
   * overlay. Held by reference — perception reuses its own arrays, so this
   * tracks the live values rather than a snapshot, which is what the overlay
   * wants anyway.
   */
  hands: Float32Array | null = null;
  /** Wall time the last `process` call cost the render loop. */
  inferenceMs = 0;
  /** Current inference cadence, adapted from measured cost. */
  intervalMs = PERCEPTION_MIN_INTERVAL_MS;
  /** Smoothed wall time one `process` call costs the render loop. */
  costMs = 0;

  private lastMs = 0;
  private lastResultMs = 0;
  private resultHz = 0;
  /**
   * True when the source runs the models on a worker, so `process` is cheap and
   * self-pacing and the render loop should drain it every frame.
   */
  private offThread = false;
  /** Consecutive inferences too slow to fit the duty budget. */
  private strikes = 0;

  constructor(private readonly deps: PerceptionPumpDeps) {}

  get attached(): boolean {
    return this.source !== null;
  }

  /** Effective inline inference budget cadence, not the delivered result rate. */
  get hz(): number {
    return 1000 / this.intervalMs;
  }

  /** Measured rate at which completed results reached the engine. */
  get actualHz(): number {
    if (this.status.kind !== 'ready' || performance.now() - this.lastResultMs > 2000) return 0;
    return this.resultHz;
  }

  /** Runs inference at a cadence derived from its own measured cost. */
  pump(nowMs: number): void {
    const video = this.deps.frame();
    if (!this.source || !video) return;

    // Off-thread perception paces itself: the worker takes one frame at a time,
    // and `process` only decodes a bitmap and hands back whatever has already
    // come back. Throttling it here would do nothing but hold a finished result
    // for up to a full interval before the engine sees it — pure added latency,
    // which is what made gestures feel late. The interval gate stays for the
    // inline source, where every call really does run the models in this frame.
    if (!this.offThread && nowMs - this.lastMs < this.intervalMs) return;

    // Time since the last result actually reached the engine — not since the
    // last attempt — because that is the interval the gesture velocities are
    // differentiated over.
    const dt = this.lastMs === 0 ? 1 / 30 : (nowMs - this.lastMs) / 1000;

    let result;
    const start = performance.now();
    try {
      result = this.source.process(video, nowMs);
    } catch (err) {
      // One bad inference must not kill the loop; drop perception instead.
      console.error('[aether] perception threw, disabling it', err);
      this.status = { kind: 'unavailable', reason: 'Inference failed.' };
      this.source.close();
      this.source = null;
      return;
    }

    // Measured here rather than trusting the source's own `latencyMs`: what
    // matters for pacing is the wall time this call cost *the render loop*,
    // including anything the source does around the inference itself.
    const cost = performance.now() - start;
    if (result && !this.offThread && !this.adaptCadence(cost)) return;
    if (!result) return;
    this.lastMs = nowMs;
    if (this.lastResultMs > 0) {
      const gap = nowMs - this.lastResultMs;
      if (gap > 0) {
        const rate = 1000 / gap;
        this.resultHz = this.resultHz === 0 || gap > 2000
          ? rate
          : this.resultHz * 0.8 + rate * 0.2;
      }
    }
    this.lastResultMs = nowMs;

    // `cost`, not `result.latencyMs`: the HUD row sits in the frame budget, so
    // it must report what the render loop actually paid. With perception on a
    // worker the model still takes ~26 ms, but the loop pays only the bitmap
    // decode — charging it the full inference would show a blown budget on a
    // frame that comfortably made 60 Hz.
    this.inferenceMs = cost;
    this.hands = result.hands;
    this.deps.engine.push_hands(result.hands, dt);
    this.deps.engine.push_pose(result.pose, dt);

    if (result.mask) {
      const { data, width, height } = result.mask;
      const mask = this.deps.mask();
      if (width * height <= mask.length) {
        mask.set(data.subarray(0, width * height));
        this.deps.engine.push_mask(width, height, true, dt);
      }
    }
  }

  /**
   * Folds the measured cost into the cadence. Returns false when perception was
   * given up on, in which case the caller must stop using this frame's result.
   */
  private adaptCadence(cost: number): boolean {
    this.costMs = this.costMs * 0.8 + cost * 0.2;
    const wanted = this.costMs / PERCEPTION_DUTY;
    this.intervalMs = Math.min(
      PERCEPTION_MAX_INTERVAL_MS,
      Math.max(PERCEPTION_MIN_INTERVAL_MS, wanted),
    );

    if (wanted <= PERCEPTION_MAX_INTERVAL_MS) {
      this.strikes = 0;
      return true;
    }
    this.strikes++;
    if (this.strikes < PERCEPTION_GIVE_UP_STRIKES) return true;

    const ms = Math.round(this.costMs);
    const reason =
      `Inference takes ${ms} ms on this device — too slow to run without ` +
      `stalling the simulation. Running on motion detection only.`;
    console.warn(`[aether] ${reason}`);
    this.source?.close();
    this.source = null;
    this.status = { kind: 'unavailable', reason };
    this.hands = null;
    this.deps.engine.clear_perception();
    return false;
  }

  /**
   * Prefers perception on a worker, falling back to the inline source.
   *
   * Off-thread is strictly better — a 26 ms inference stops being a 26 ms hole
   * in the render loop — but it needs module workers and `createImageBitmap`,
   * and the worker can fail to start for reasons the page cannot inspect. So
   * the inline path stays as the fallback rather than the default: same models,
   * same results, just paid for out of the frame budget.
   */
  private async build(): Promise<PerceptionSource> {
    const offloaded = new WorkerPerception();
    try {
      await offloaded.init();
      return offloaded;
    } catch (err) {
      console.warn('[aether] perception worker unusable, running inference inline', err);
      offloaded.close();
      return new MediaPipePerception();
    }
  }

  /**
   * Loads the vision models off the critical path and attaches them.
   *
   * Never rejects: a failure here is a capability downgrade, not an error. The
   * Rust optical-flow path keeps driving the fluid, so the app stays fully
   * responsive and the HUD explains what is missing.
   */
  async attach(): Promise<void> {
    const source = await this.build();
    try {
      await source.init();
      // A scripted source may have been injected while the models loaded
      // (the headless suite does exactly this), and it must win — otherwise a
      // slow load silently overwrites the test's perception mid-run.
      if (this.source !== null) {
        source.close();
        return;
      }
      this.source = source;
      this.offThread = source instanceof WorkerPerception;
      this.status = source.status;
    } catch (err) {
      const reason = `Vision models unavailable: ${String(err)}`;
      console.warn(`[aether] ${reason}`);
      this.status = { kind: 'unavailable', reason };
      source.close();
    }
  }

  /**
   * Replaces the perception source at runtime. The headless suite uses this to
   * feed scripted landmarks, which is what makes the gesture pipeline testable
   * without a real hand in front of a real camera. Passing null detaches
   * perception entirely, leaving the model-free optical-flow path in charge.
   */
  setSource(source: PerceptionSource | null): void {
    this.source?.close();
    this.source = source;
    this.status = source?.status ?? { kind: 'unavailable', reason: 'detached' };
    this.deps.engine.clear_perception();
    this.hands = null;
    this.lastMs = 0;
    this.lastResultMs = 0;
    this.resultHz = 0;
    this.offThread = source instanceof WorkerPerception;
    // A scripted source has nothing to do with the real one's cost, so the
    // adaptive cadence has to start over or a slow MediaPipe load would leave
    // an injected test source throttled to once every two seconds.
    this.intervalMs = PERCEPTION_MIN_INTERVAL_MS;
    this.costMs = 0;
    this.strikes = 0;
  }

  close(): void {
    this.source?.close();
  }
}
