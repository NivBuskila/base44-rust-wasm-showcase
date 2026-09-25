// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  QA_SESSION_KEY,
  QaRecorder,
  clearQaSession,
  readQaSession,
  type QaDiagnostics,
} from './qa-recorder';

// The recorder's own cadence, mirrored so a change to it is a visible one.
const SAMPLE_MS = 500;
const BOOT_MS = 1000;
const MAX_EVENTS = 60;

function diag(over: Partial<QaDiagnostics> = {}): QaDiagnostics {
  return {
    fps: 60,
    stepMs: 4,
    renderMs: 3,
    inferenceMs: 20,
    decodeMs: 0,
    modelMs: 0,
    workerOverheadMs: 0,
    firstHandAtMs: null,
    cameraMs: 1,
    hudMs: 0.1,
    frameMs: 8,
    outsideMs: 2,
    cameraAvailable: false,
    perception: { kind: 'loading' },
    particleCount: 120_000,
    mode: 'aether',
    engine: { name: 'threads', threads: 8, reason: 'cross-origin isolated' },
    perceptionHz: 30,
    qualityTier: 0,
    ...over,
  };
}

/** A hand-cranked clock for `performance.now`, so sampling is deterministic. */
function clock() {
  let now = 0;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  return {
    /** Jumps past boot and lands on a sample boundary. */
    warm: () => {
      now = BOOT_MS + SAMPLE_MS;
    },
    tick: (ms = SAMPLE_MS) => {
      now += ms;
    },
  };
}

describe('QaRecorder', () => {
  beforeEach(() => {
    localStorage.clear();
    // jsdom has no `sendBeacon`; the recorder must survive both its absence
    // (see the try/catch there) and its presence.
    Object.defineProperty(navigator, 'sendBeacon', {
      configurable: true,
      value: vi.fn(() => true),
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('samples at its own cadence and ignores the boot second', () => {
    const c = clock();
    const rec = new QaRecorder();
    for (let i = 0; i < 20; i++) {
      rec.sample(diag({ fps: 60 }));
      c.tick(50); // called every frame; must not record every frame
    }
    expect(rec.summary().samples).toBe(0);

    c.warm();
    rec.sample(diag({ fps: 60 }));
    rec.sample(diag({ fps: 60 })); // same instant: rate-limited away
    expect(rec.summary().samples).toBe(1);
    c.tick(SAMPLE_MS - 1);
    rec.sample(diag({ fps: 60 }));
    expect(rec.summary().samples).toBe(1);
    c.tick(1);
    rec.sample(diag({ fps: 60 }));
    expect(rec.summary().samples).toBe(2);
  });

  it('only reads live diagnostics when it takes a sample', () => {
    const c = clock();
    const rec = new QaRecorder();
    const read = vi.fn(() => diag());
    for (let i = 0; i < 10; i++) {
      rec.sample(read);
      c.tick(10);
    }
    expect(read).not.toHaveBeenCalled();
    c.warm();
    rec.sample(read);
    expect(read).toHaveBeenCalledTimes(1);
    rec.sample(read);
    expect(read).toHaveBeenCalledTimes(1);
    c.tick(SAMPLE_MS);
    rec.sample(read);
    expect(read).toHaveBeenCalledTimes(2);
  });

  it('summarises frame rate with percentiles and the share of slow samples', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    const fps = [60, 60, 60, 60, 60, 60, 60, 60, 30, 45];
    for (const f of fps) {
      rec.sample(diag({ fps: f, stepMs: f === 30 ? 20 : 4 }));
      c.tick();
    }
    const s = rec.summary();
    expect(s.samples).toBe(10);
    expect(s.fps.min).toBe(30);
    expect(s.fps.max).toBe(60);
    expect(s.fps.median).toBe(60);
    expect(s.fps.mean).toBe(55.5);
    // 30 and 45 are both under the 50 fps stutter line.
    expect(s.slowFrameShare).toBe(0.2);
    expect(s.stepMs.median).toBe(4);
    expect(s.stepMs.p95).toBe(20);
    expect(s.engine).toBe('threads:8');
    expect(s.engineReason).toBe('cross-origin isolated');
  });

  it('does not report a configured cadence while models are still loading', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag({ perceptionHz: 30, perception: { kind: 'loading' } }));
    expect(rec.summary().perceptionHz).toEqual({ median: 0, min: 0 });
    c.tick();
    rec.sample(diag({ perceptionHz: 12, perception: { kind: 'ready', delegate: 'GPU' } }));
    expect(rec.summary().perceptionHz).toEqual({ median: 12, min: 12 });
  });

  it('records first hand arrival and separates decode, model and worker overhead', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag({ firstHandAtMs: null }));
    c.tick();
    rec.sample(diag({ firstHandAtMs: 1800, decodeMs: 4, modelMs: 22, workerOverheadMs: 3 }));
    c.tick();
    rec.sample(diag({ firstHandAtMs: 1900, decodeMs: 6, modelMs: 24, workerOverheadMs: 5 }));
    const s = rec.summary();
    expect(s.firstHandMs).toBe(1800);
    expect(s.decodeMs.p95).toBe(6);
    expect(s.modelMs.p95).toBe(24);
    expect(s.workerOverheadMs.p95).toBe(5);
  });

  it('reports only the last 30 seconds and marks old samples stale', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag({ fps: 20, frameMs: 30 }));
    c.tick(30_500);
    rec.sample(diag({ fps: 60, frameMs: 8, perception: { kind: 'ready', delegate: 'GPU' }, perceptionHz: 18 }));
    const s = rec.summary();
    expect(s.samples).toBe(2);
    expect(s.recent.samples).toBe(1);
    expect(s.recent.fps.median).toBe(60);
    expect(s.recent.frameMs.p95).toBe(8);
    expect(s.recent.perceptionHz.median).toBe(18);
    expect(s.recent.lastSampleAtMs).toBe(32_000);
    expect(s.recent.lastSampleAgeMs).toBe(0);
    c.tick(31_000);
    expect(rec.summary().recent.samples).toBe(0);
    expect(rec.summary().recent.lastSampleAgeMs).toBeNull();
  });

  it('drops non-finite timings without dropping the sample', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag({ fps: 60, stepMs: NaN }));
    c.tick();
    rec.sample(diag({ fps: 60, stepMs: 5 }));
    const s = rec.summary();
    expect(s.samples).toBe(2);
    expect(s.stepMs.median).toBe(5);
  });

  it('records each kind of transition once, with the session time', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag());
    c.tick();
    rec.sample(
      diag({
        qualityTier: 2,
        perception: { kind: 'ready', delegate: 'GPU' },
        mode: 'debug',
        particleCount: 50_000,
      }),
    );
    c.tick();
    // Nothing changed: no new events.
    rec.sample(
      diag({
        qualityTier: 2,
        perception: { kind: 'ready', delegate: 'GPU' },
        mode: 'debug',
        particleCount: 50_000,
      }),
    );
    const s = rec.summary();
    expect(s.events.map((e) => [e.kind, e.from, e.to])).toEqual([
      ['quality', '0', '2'],
      ['perception', 'loading', 'ready:GPU'],
      ['mode', 'aether', 'debug'],
      ['particles', '120000', '50000'],
    ]);
    expect(s.events.every((e) => e.atMs === BOOT_MS + 2 * SAMPLE_MS)).toBe(true);
    expect(s.qualityTier).toEqual({ start: 0, worst: 2, end: 2 });
    expect(s.particleCount).toEqual({ min: 50_000, max: 120_000, end: 50_000 });
  });

  it('caps the event log so a flapping governor cannot grow the record', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    for (let i = 0; i < MAX_EVENTS + 20; i++) {
      rec.sample(diag({ qualityTier: i % 2 }));
      c.tick();
    }
    expect(rec.summary().events).toHaveLength(MAX_EVENTS);
  });

  it('never persists an empty record', () => {
    clock();
    const rec = new QaRecorder();
    rec.persist();
    expect(localStorage.getItem(QA_SESSION_KEY)).toBeNull();
    expect(navigator.sendBeacon).not.toHaveBeenCalled();
  });

  it('persists from a top-level tab to storage and to the dev-server drop box', () => {
    const c = clock();
    const rec = new QaRecorder();
    c.warm();
    rec.sample(diag({ fps: 58 }));
    const summarise = vi.spyOn(rec, 'summary');
    rec.persist();
    expect(summarise).toHaveBeenCalledTimes(1);

    expect(navigator.sendBeacon).toHaveBeenCalledWith('/__qa/session', expect.any(Blob));
    const stored = readQaSession();
    expect(stored?.samples).toBe(1);
    expect(stored?.fps.median).toBe(58);

    clearQaSession();
    expect(readQaSession()).toBeNull();
  });

  it('refuses to persist from an embedded frame', () => {
    const c = clock();
    c.warm();
    const top = vi.spyOn(window, 'top', 'get').mockReturnValue({} as Window);
    const rec = new QaRecorder();
    rec.sample(diag());
    rec.persist();
    top.mockRestore();
    expect(localStorage.getItem(QA_SESSION_KEY)).toBeNull();
    expect(navigator.sendBeacon).not.toHaveBeenCalled();
  });

  it('reads back null for a corrupt record instead of throwing', () => {
    localStorage.setItem(QA_SESSION_KEY, '{not json');
    expect(readQaSession()).toBeNull();
  });
});
