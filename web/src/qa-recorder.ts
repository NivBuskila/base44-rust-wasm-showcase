/**
 * Session recorder — the bridge from a human QA session back to the tooling.
 *
 * The real engine only runs in a standalone top-level tab: the embedded preview
 * iframe is not cross-origin isolated and never fires `requestAnimationFrame`,
 * so the frame rate, the quality governor and the camera path can only be
 * observed by a person driving a real tab. That observation used to be lost the
 * moment the tab closed.
 *
 * This records it. The recorder samples `App.diagnostics` at a low rate and
 * keeps a rolling summary in `localStorage`, which is scoped per **origin**
 * rather than per tab — so a session driven in a standalone tab is readable
 * afterwards from any other context on the same origin, including the embedded
 * preview.
 *
 * Design constraints, both taken from the diagnostics comment in `main.ts`:
 * instrumentation must not perturb what it measures, so this samples at 2 Hz,
 * touches nothing that forces a GL flush (never `luminance()`), and writes to
 * storage on an interval rather than per frame. And it keeps a bounded summary,
 * not a full trace: a long session must not grow without limit.
 */

/** Storage key holding the most recent session summary. */
export const QA_SESSION_KEY = 'aether.qa.session';

/** Dev-server drop box; see `web/qa-session-endpoint.ts`. */
const QA_SESSION_ENDPOINT = '/__qa/session';

/** Sampling period. 2 Hz is enough for fps percentiles over a human session. */
const SAMPLE_MS = 500;

/** How often the summary is persisted. Writes are synchronous, so not often. */
const PERSIST_MS = 2000;

/**
 * Samples kept for percentiles: 1800 at 2 Hz is 15 minutes, after which the
 * oldest drop, because the interesting part of a QA session is what the tester
 * was doing recently.
 */
const MAX_SAMPLES = 1800;

/** Cap on recorded events, so a flapping governor cannot grow the record. */
const MAX_EVENTS = 60;

/** Sampled frame rates below this are what a tester perceives as stutter. */
const SLOW_FRAME_FPS = 50;

import { gpuErrorLog } from './gpu/error-log';
import type { EngineTier } from './engine-loader';
import type { PerceptionStatus, ViewMode } from './types';

/**
 * Flattens a perception status into one comparable label, so a transition can
 * be detected and stored without keeping the whole object per sample. The
 * delegate matters (GPU vs CPU is a large performance difference) and so does
 * the failure reason, so both are folded into the label.
 */
function perceptionLabel(status: PerceptionStatus): string {
  switch (status.kind) {
    case 'ready':
      return `ready:${status.delegate}`;
    case 'unavailable':
      return `unavailable:${status.reason}`;
    default:
      return status.kind;
  }
}

/** The diagnostics fields this recorder consumes. */
export interface QaDiagnostics {
  fps: number;
  stepMs: number;
  renderMs: number;
  inferenceMs: number;
  cameraMs: number;
  hudMs: number;
  frameMs: number;
  outsideMs: number;
  cameraAvailable: boolean;
  perception: PerceptionStatus;
  particleCount: number;
  mode: ViewMode;
  engine: EngineTier;
  perceptionHz: number;
  qualityTier: number;
}

/** A notable transition, with the session time it happened at. */
interface QaEvent {
  atMs: number;
  kind: 'quality' | 'perception' | 'mode' | 'particles';
  from: string;
  to: string;
}

/** What a reader gets back out of storage. */
export interface QaSession {
  startedAt: string;
  durationMs: number;
  samples: number;
  engine: string;
  /** The engine loader's own explanation for that tier. */
  engineReason: string;
  /** Whether the document could allocate `SharedArrayBuffer` at all. */
  crossOriginIsolated: boolean;
  cameraAvailable: boolean;
  fps: { min: number; p5: number; median: number; mean: number; max: number };
  slowFrameShare: number;
  stepMs: { median: number; p95: number };
  renderMs: { median: number; p95: number };
  inferenceMs: { median: number; p95: number };
  cameraMs: { median: number; p95: number };
  hudMs: { median: number; p95: number };
  frameMs: { median: number; p95: number };
  /** Frame time the callback did not account for — see `App.outsideMs`. */
  outsideMs: { median: number; p95: number };
  perceptionHz: { median: number; min: number };
  qualityTier: { start: number; worst: number; end: number };
  particleCount: { min: number; max: number; end: number };
  /** WebGPU errors this tab hit, each with its repeat count. Empty is healthy. */
  gpuErrors: string[];
  events: QaEvent[];
}

/**
 * Whether this document is framed. Cross-origin framing makes `window.top`
 * throw on access, which is itself the answer, so the throw is treated as
 * embedded rather than propagated.
 */
function isEmbedded(): boolean {
  try {
    return window.top !== window.self;
  } catch {
    return true;
  }
}

function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

const round = (n: number, digits = 1): number => {
  const k = 10 ** digits;
  return Math.round(n * k) / k;
};

function msStats(values: readonly number[]): { median: number; p95: number } {
  const sorted = [...values].sort((a, b) => a - b);
  return { median: round(quantile(sorted, 0.5), 2), p95: round(quantile(sorted, 0.95), 2) };
}

function push(target: number[], value: number): void {
  if (!Number.isFinite(value)) return;
  target.push(value);
  if (target.length > MAX_SAMPLES) target.shift();
}

/**
 * Records a session. Construct once and call {@link sample} from the frame
 * loop — it rate-limits itself, so calling it every frame is correct and cheap.
 */
export class QaRecorder {
  private readonly startedAt = new Date().toISOString();
  private readonly t0 = performance.now();
  private lastSample = 0;
  private lastPersist = 0;

  private readonly fps: number[] = [];
  private readonly stepMs: number[] = [];
  private readonly renderMs: number[] = [];
  private readonly inferenceMs: number[] = [];
  private readonly cameraMs: number[] = [];
  private readonly hudMs: number[] = [];
  private readonly frameMs: number[] = [];
  private readonly outsideMs: number[] = [];
  private readonly perceptionHz: number[] = [];
  private readonly events: QaEvent[] = [];

  private previous: QaDiagnostics | null = null;
  private tierStart = 0;
  private tierWorst = 0;
  private particlesMin = Number.POSITIVE_INFINITY;
  private particlesMax = 0;
  private engine = 'unknown';
  private engineReason = 'unknown';
  private isolated = false;
  private cameraSeen = false;

  sample(diag: QaDiagnostics): void {
    const now = performance.now();
    if (now - this.lastSample < SAMPLE_MS) return;
    this.lastSample = now;

    // The first second is boot: the smoothed fps is still climbing out of its
    // seed value and would drag every percentile down.
    if (now - this.t0 > 1000 && Number.isFinite(diag.fps) && diag.fps > 0) {
      push(this.fps, diag.fps);
      push(this.stepMs, diag.stepMs);
      push(this.renderMs, diag.renderMs);
      push(this.inferenceMs, diag.inferenceMs);
      push(this.cameraMs, diag.cameraMs);
      push(this.hudMs, diag.hudMs);
      push(this.frameMs, diag.frameMs);
      push(this.outsideMs, diag.outsideMs);
      push(this.perceptionHz, diag.perceptionHz);
    }

    if (this.previous === null) this.tierStart = diag.qualityTier;
    this.tierWorst = Math.max(this.tierWorst, diag.qualityTier);
    this.particlesMin = Math.min(this.particlesMin, diag.particleCount);
    this.particlesMax = Math.max(this.particlesMax, diag.particleCount);
    // `threads:8` vs `single:1` is the single most important fact about a
    // session's frame rate, so the tier is flattened to a label with its width.
    this.engine = `${diag.engine.name}:${diag.engine.threads}`;
    // Why that tier was chosen — a `single:1` session is only actionable when
    // the record also says which gate rejected the threaded build.
    this.engineReason = diag.engine.reason;
    this.isolated = Boolean(globalThis.crossOriginIsolated);
    this.cameraSeen = this.cameraSeen || diag.cameraAvailable;

    this.recordTransitions(diag, now - this.t0);
    this.previous = diag;

    if (now - this.lastPersist >= PERSIST_MS) {
      this.lastPersist = now;
      this.persist();
    }
  }

  /** The summary as it stands — also what {@link persist} writes. */
  summary(): QaSession {
    const fps = [...this.fps].sort((a, b) => a - b);
    const hz = [...this.perceptionHz].sort((a, b) => a - b);
    const slow = this.fps.filter((f) => f < SLOW_FRAME_FPS).length;
    const last = this.previous;
    return {
      startedAt: this.startedAt,
      durationMs: Math.round(performance.now() - this.t0),
      samples: fps.length,
      engine: this.engine,
      engineReason: this.engineReason,
      crossOriginIsolated: this.isolated,
      cameraAvailable: this.cameraSeen,
      fps: {
        min: round(quantile(fps, 0)),
        p5: round(quantile(fps, 0.05)),
        median: round(quantile(fps, 0.5)),
        mean: round(fps.reduce((a, b) => a + b, 0) / (fps.length || 1)),
        max: round(quantile(fps, 1)),
      },
      slowFrameShare: round(slow / (this.fps.length || 1), 3),
      stepMs: msStats(this.stepMs),
      renderMs: msStats(this.renderMs),
      inferenceMs: msStats(this.inferenceMs),
      cameraMs: msStats(this.cameraMs),
      hudMs: msStats(this.hudMs),
      frameMs: msStats(this.frameMs),
      outsideMs: msStats(this.outsideMs),
      perceptionHz: { median: round(quantile(hz, 0.5)), min: round(quantile(hz, 0)) },
      qualityTier: {
        start: this.tierStart,
        worst: this.tierWorst,
        end: last?.qualityTier ?? this.tierStart,
      },
      particleCount: {
        min: Number.isFinite(this.particlesMin) ? this.particlesMin : 0,
        max: this.particlesMax,
        end: last?.particleCount ?? 0,
      },
      gpuErrors: gpuErrorLog(),
      events: this.events,
    };
  }

  /**
   * Writes the summary. Storage failures are ignored: this is diagnostics.
   *
   * Only a top-level tab may write. The embedded preview iframe does run the
   * frame loop — at a throttled rate and always single-threaded, since it never
   * gets `SharedArrayBuffer` — so it produces a record full of real samples that
   * describes the iframe rather than the QA pass, and `localStorage` is shared
   * per origin, so that record overwrites the standalone-tab session this exists
   * to capture. Sample count cannot tell the two apart; only the frame context
   * can. A record with no samples is still never written.
   */
  persist(): void {
    if (this.fps.length === 0) return;
    if (isEmbedded()) return;
    this.upload();
    try {
      localStorage.setItem(QA_SESSION_KEY, JSON.stringify(this.summary()));
    } catch {
      /* private mode or quota — never break the session over telemetry */
    }
  }

  /**
   * Ships the summary to the dev server, which is the only bridge that actually
   * reaches outside this tab: `localStorage` is partitioned by top-level site, so
   * a record written here is invisible to the embedded preview even though the
   * origin matches. `sendBeacon` because this also runs from `pagehide`, where a
   * `fetch` is not guaranteed to be flushed; failure is ignored, as with storage.
   */
  private upload(): void {
    // The endpoint is a dev-server plugin; a static host answers 405.
    if (!import.meta.env.DEV) return;
    try {
      const body = new Blob([JSON.stringify(this.summary())], { type: 'application/json' });
      navigator.sendBeacon(QA_SESSION_ENDPOINT, body);
    } catch {
      /* dev-server absent (a production build) — never break the session */
    }
  }

  private recordTransitions(diag: QaDiagnostics, atMs: number): void {
    const prev = this.previous;
    if (!prev) return;
    const add = (kind: QaEvent['kind'], from: string, to: string) => {
      if (this.events.length >= MAX_EVENTS) return;
      this.events.push({ atMs: Math.round(atMs), kind, from, to });
    };
    if (prev.qualityTier !== diag.qualityTier) {
      add('quality', String(prev.qualityTier), String(diag.qualityTier));
    }
    const [was, now] = [perceptionLabel(prev.perception), perceptionLabel(diag.perception)];
    if (was !== now) add('perception', was, now);
    if (prev.mode !== diag.mode) add('mode', prev.mode, diag.mode);
    if (prev.particleCount !== diag.particleCount) {
      add('particles', String(prev.particleCount), String(diag.particleCount));
    }
  }
}

/** Reads the last recorded session, from any context on the same origin. */
export function readQaSession(): QaSession | null {
  try {
    const raw = localStorage.getItem(QA_SESSION_KEY);
    return raw ? (JSON.parse(raw) as QaSession) : null;
  } catch {
    return null;
  }
}

/** Clears the record so the next session starts from a clean slate. */
export function clearQaSession(): void {
  try {
    localStorage.removeItem(QA_SESSION_KEY);
  } catch {
    /* ignore */
  }
}
