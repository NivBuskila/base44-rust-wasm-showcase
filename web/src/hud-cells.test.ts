import { describe, expect, it } from 'vitest';

import { STAT, STATS_LEN } from './constants';
import { handsPresent, statCells, warpView } from './hud-cells';

function stats(values: Partial<Record<keyof typeof STAT, number>>): Float32Array {
  const a = new Float32Array(STATS_LEN);
  for (const [k, v] of Object.entries(values)) a[STAT[k as keyof typeof STAT]] = v as number;
  return a;
}

describe('statCells', () => {
  it('formats every cell the grid prints', () => {
    const cells = statCells(
      stats({
        PARTICLES_ALIVE: 120_000,
        FLUID_ENERGY: 1.234,
        FLUID_MAX_SPEED: 42.66,
        FLUID_DIVERGENCE: 0.00042,
        MOTION_ENERGY: 0.01239,
        HANDS_PRESENT: 2,
        MASK_PRESENT: 1,
        MASK_COVERAGE: 0.375,
        NAN_REPAIRS: 3,
      }),
    );
    expect(cells).toMatchObject({
      energy: '1.23',
      speed: '42.7',
      divergence: '4.2e−4',
      motion: '0.012',
      hands: '2',
      body: '38%',
      repairs: '3',
    });
    expect(cells.particles).toContain('120');
  });

  it('shows a dash for body coverage while no mask is present', () => {
    expect(statCells(stats({ MASK_PRESENT: 0, MASK_COVERAGE: 0.9 })).body).toBe('—');
  });

  it('survives missing stats and non-finite values', () => {
    expect(statCells(undefined).energy).toBe('0.00');
    expect(statCells(stats({ FLUID_ENERGY: Number.NaN })).energy).toBe('0.00');
  });

  it('clamps the hand count to the two cards that exist', () => {
    expect(statCells(stats({ HANDS_PRESENT: 5 })).hands).toBe('2');
    expect(statCells(stats({ HANDS_PRESENT: -1 })).hands).toBe('0');
    expect(handsPresent(stats({ HANDS_PRESENT: 1.6 }))).toBe(2);
  });
});

describe('warpView', () => {
  it('puts 1x a quarter along the travel and reads as off', () => {
    expect(warpView(stats({ TIME_SCALE: 1 }))).toEqual({
      label: '1.00×',
      pct: '25%',
      warping: false,
    });
  });

  it('is on once time scale leaves the dead band, and clamps at the 4x ceiling', () => {
    expect(warpView(stats({ TIME_SCALE: 0.5 }))).toMatchObject({ pct: '13%', warping: true });
    expect(warpView(stats({ TIME_SCALE: 9 }))).toMatchObject({ pct: '100%', warping: true });
  });
});
