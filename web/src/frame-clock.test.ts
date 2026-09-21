import { describe, expect, it } from 'vitest';

import { FrameClock } from './frame-clock';

describe('FrameClock', () => {
  it('reports the real delta and clamps the sim step', () => {
    const clock = new FrameClock();
    clock.reset(1000);
    const a = clock.tick(1016);
    expect(a.realDt).toBeCloseTo(0.016, 6);
    expect(a.simDt).toBeCloseTo(0.016, 6);

    // A one-second stall: honest realDt, clamped step.
    const b = clock.tick(2016);
    expect(b.realDt).toBeCloseTo(1, 6);
    expect(b.simDt).toBeCloseTo(0.05, 6);

    // A repeated timestamp must never produce a zero or negative step.
    const c = clock.tick(2016);
    expect(c.realDt).toBeGreaterThan(0);
    expect(c.simDt).toBeCloseTo(1 / 480, 6);
  });

  it('counts frames and smooths towards the steady rate', () => {
    const clock = new FrameClock();
    clock.reset(0);
    for (let i = 1; i <= 200; i++) clock.tick(i * 16.6667);
    expect(clock.frames).toBe(200);
    expect(clock.fps).toBeGreaterThan(55);
    expect(clock.fps).toBeLessThan(65);
  });

  it('charges the unspent wall time to the gap outside the callback', () => {
    const clock = new FrameClock();
    clock.reset(0);
    clock.tick(16);
    expect(clock.outside(6)).toBeCloseTo(10, 6);
    // Work longer than the frame period leaves no gap, never a negative one.
    expect(clock.outside(40)).toBe(0);
  });
});
