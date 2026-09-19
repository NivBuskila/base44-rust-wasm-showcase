import { describe, expect, it } from 'vitest';

import { MAX_PARTICLES } from './constants';
import { frameTone } from './hud-meter';
import { PARTICLE_DETENTS, countToDetent, detentToCount, thousands } from './hud-spec';

describe('particle slider mapping', () => {
  it('spans the whole pool from the floor to the maximum', () => {
    expect(detentToCount(0)).toBe(1000);
    expect(detentToCount(PARTICLE_DETENTS)).toBe(MAX_PARTICLES);
  });

  it('is monotonic and quantised to readable steps', () => {
    let last = 0;
    for (let d = 0; d <= PARTICLE_DETENTS; d++) {
      const n = detentToCount(d);
      expect(n).toBeGreaterThanOrEqual(last);
      const quantum = n < 10_000 ? 500 : n < 100_000 ? 1_000 : 5_000;
      expect(n % quantum).toBe(0);
      last = n;
    }
  });

  it('round-trips a count through the slider within one quantum', () => {
    for (const n of [1000, 2500, 12_000, 120_000, 480_000, MAX_PARTICLES]) {
      const back = detentToCount(countToDetent(n));
      const quantum = n < 10_000 ? 500 : n < 100_000 ? 1_000 : 5_000;
      expect(Math.abs(back - n)).toBeLessThanOrEqual(quantum);
    }
  });

  it('clamps garbage on both sides', () => {
    expect(detentToCount(-50)).toBe(1000);
    expect(detentToCount(NaN)).toBe(1000);
    expect(detentToCount(PARTICLE_DETENTS * 10)).toBe(MAX_PARTICLES);
    expect(countToDetent(-1)).toBe(0);
    expect(countToDetent(Number.POSITIVE_INFINITY)).toBe(PARTICLE_DETENTS);
  });
});

describe('thousands', () => {
  it('keeps the columns narrow', () => {
    expect(thousands(0)).toBe('0');
    expect(thousands(999)).toBe('999');
    expect(thousands(1500)).toBe('1.5k');
    expect(thousands(120_000)).toBe('120k');
    expect(thousands(1_000_000)).toBe('1000k');
    expect(thousands(NaN)).toBe('—');
  });
});

describe('frameTone', () => {
  it('does not paint a perfect 60 Hz session red', () => {
    expect(frameTone(1000 / 60)).toBe('ok');
    expect(frameTone(18.4)).toBe('ok');
    expect(frameTone(18.5)).toBe('warn');
    expect(frameTone(20.9)).toBe('warn');
    expect(frameTone(21)).toBe('over');
  });
});
