import { describe, expect, it } from 'vitest';

import { FrameTimer } from './hud-frame-timer';
import { fmtSmall, presence, spellAt } from './hud-readouts';

describe('presence', () => {
  it('reads a lone right hand as slot 1, not slot 0', () => {
    expect(presence(1, ['idle', 'attract'])).toEqual([false, true]);
  });

  it('assumes slot 0 for one hand with nothing latched', () => {
    expect(presence(1, ['idle', 'idle'])).toEqual([true, false]);
  });

  it('trusts a latched spell over a missing count', () => {
    expect(presence(0, ['push', 'attract'])).toEqual([true, true]);
  });

  it('reports no hands when nothing is seen or cast', () => {
    expect(presence(0, undefined)).toEqual([false, false]);
  });
});

describe('spellAt', () => {
  it('falls back to idle for anything unsent', () => {
    expect(spellAt(undefined, 0)).toBe('idle');
    expect(spellAt(['', 'push'], 0)).toBe('idle');
    expect(spellAt(['', 'push'], 1)).toBe('push');
  });
});

describe('fmtSmall', () => {
  it('keeps tiny values legible', () => {
    expect(fmtSmall(0)).toBe('0');
    expect(fmtSmall(0.00042)).toBe('4.2e−4');
    expect(fmtSmall(0.125)).toBe('0.1250');
  });
});

describe('FrameTimer', () => {
  const vsync = 1000 / 60;

  it('holds the worst frame in the window for the sparkline, then resets', () => {
    const t = new FrameTimer();
    let now = 0;
    for (const gap of [16, 40, 16]) {
      now += gap;
      t.sample(now);
    }
    expect(t.read(60).worstMs).toBeCloseTo(40, 5);
    now += 16;
    t.sample(now);
    expect(t.read(60).worstMs).toBeCloseTo(16, 5);
  });

  it('reads a steady 60 Hz display as 60 when fed frame timestamps, however long each frame\'s work', () => {
    // Work inside each frame varies from 4 to 12 ms, as the engine step does.
    const work = [4, 12, 5, 11, 4, 12];
    const byFrame = new FrameTimer();
    const byCall = new FrameTimer();
    work.forEach((ms, i) => {
      byFrame.sample(i * vsync);
      byCall.sample(i * vsync + ms);
    });
    const frame = byFrame.read(60);
    expect(1000 / frame.meanMs).toBeCloseTo(60, 5);
    expect(1000 / frame.worstMs).toBeCloseTo(60, 5);
    // Timing the calls instead would report the jitter as a slow frame.
    expect(1000 / byCall.read(60).worstMs).toBeLessThan(45);
  });

  it('shows the delivered rate, not half the refresh rate, when one frame in twelve misses', () => {
    const t = new FrameTimer();
    let now = 0;
    const reads: { fps: number; worst: number }[] = [];
    let lastPaint = 0;
    // Two seconds of a 60 Hz loop that misses one refresh every twelfth frame.
    for (let i = 1; i <= 110; i++) {
      now += i % 12 === 0 ? 2 * vsync : vsync;
      t.sample(now);
      if (now - lastPaint >= 100) {
        lastPaint = now;
        const r = t.read(60);
        reads.push({ fps: 1000 / r.meanMs, worst: r.worstMs });
      }
    }
    const settled = reads.slice(-5);
    // 12 frames shown over 13 refreshes: 55.4 fps delivered.
    for (const r of settled) expect(r.fps).toBeGreaterThan(54);
    for (const r of settled) expect(r.fps).toBeLessThan(57);
    // The miss is still on the chart.
    expect(Math.max(...reads.map((r) => r.worst))).toBeCloseTo(2 * vsync, 5);
  });

  it('follows a real drop to 30 within about a second', () => {
    const t = new FrameTimer();
    let now = 0;
    for (let i = 0; i < 120; i++) t.sample((now += vsync));
    t.read(60);
    for (let i = 0; i < 36; i++) t.sample((now += 2 * vsync));
    expect(1000 / t.read(60).meanMs).toBeCloseTo(30, 0);
  });

  it('drops a backgrounded gap and the second before it', () => {
    const t = new FrameTimer();
    t.sample(0);
    t.sample(5_000);
    // Nothing measurable, and the loop's own average is not warmed up yet.
    expect(t.read(60)).toEqual({ meanMs: 0, worstMs: 0 });

    let now = 5_000;
    for (let i = 0; i < 30; i++) t.sample((now += 2 * vsync));
    t.sample((now += 5_000));
    for (let i = 0; i < 10; i++) t.sample((now += vsync));
    // Only the frames since coming back count.
    expect(1000 / t.read(60).meanMs).toBeCloseTo(60, 5);
  });

  it('falls back to the loop average once warmed up', () => {
    const t = new FrameTimer();
    for (let i = 1; i <= 25; i++) t.sample(i * 16);
    t.read(60);
    expect(t.read(50)).toEqual({ meanMs: 20, worstMs: 20 });
  });
});
