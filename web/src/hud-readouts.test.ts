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
  it('holds the worst frame in the window and then resets', () => {
    const t = new FrameTimer();
    let now = 0;
    for (const gap of [16, 40, 16]) {
      now += gap;
      t.sample(now);
    }
    expect(t.read(60)).toBeCloseTo(40, 5);
    now += 16;
    t.sample(now);
    expect(t.read(60)).toBeCloseTo(16, 5);
  });

  it('drops a backgrounded gap', () => {
    const t = new FrameTimer();
    t.sample(0);
    t.sample(5_000);
    // Nothing measurable, and the loop's own average is not warmed up yet.
    expect(t.read(60)).toBe(0);
  });

  it('falls back to the loop average once warmed up', () => {
    const t = new FrameTimer();
    for (let i = 1; i <= 25; i++) t.sample(i * 16);
    t.read(60);
    expect(t.read(50)).toBeCloseTo(20, 5);
  });
});
