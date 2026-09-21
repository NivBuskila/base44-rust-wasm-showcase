/**
 * The telemetry grid as data: the raw `Float32Array` the engine hands over,
 * turned into the exact strings the cells print, plus the time-warp readout.
 *
 * Pure on purpose — every read here is finite-guarded because `stats` comes
 * straight out of WASM, and that guarding is the part worth testing without a
 * DOM. `hud.ts` only writes what comes back.
 */

import { STAT } from './constants';
import { finite } from './hud-meter';
import { fmtSmall } from './hud-readouts';
import { clamp01, thousands } from './hud-spec';
import type { HudStats } from './types';

/** How the warp bar reads: the label, the fill length, and whether it is on. */
export interface WarpView {
  label: string;
  pct: string;
  warping: boolean;
}

/** Hands present, rounded off the engine's float count. */
export function handsPresent(st: HudStats['stats'] | undefined): number {
  return Math.round(finite(st?.[STAT.HANDS_PRESENT]));
}

/** Every cell's text, keyed by the `CELLS` id it belongs to. */
export function statCells(st: HudStats['stats'] | undefined): Record<string, string> {
  const maskOn = finite(st?.[STAT.MASK_PRESENT]) > 0.5;
  const hands = handsPresent(st);
  return {
    particles: thousands(finite(st?.[STAT.PARTICLES_ALIVE])),
    energy: finite(st?.[STAT.FLUID_ENERGY]).toFixed(2),
    speed: finite(st?.[STAT.FLUID_MAX_SPEED]).toFixed(1),
    divergence: fmtSmall(finite(st?.[STAT.FLUID_DIVERGENCE])),
    motion: finite(st?.[STAT.MOTION_ENERGY]).toFixed(3),
    hands: `${Math.max(0, Math.min(2, hands))}`,
    body: maskOn ? `${(finite(st?.[STAT.MASK_COVERAGE]) * 100).toFixed(0)}%` : '—',
    repairs: finite(st?.[STAT.NAN_REPAIRS]).toFixed(0),
  };
}

/**
 * The warp bar. Full travel is the engine's own 4x clamp, so 1x sits a quarter
 * along: left of the thumb is slow motion, right of it is fast.
 */
export function warpView(st: HudStats['stats'] | undefined): WarpView {
  const timeScale = finite(st?.[STAT.TIME_SCALE]);
  return {
    label: `${timeScale.toFixed(2)}×`,
    pct: `${Math.round(clamp01(timeScale / 4) * 100)}%`,
    warping: Math.abs(timeScale - 1) > 0.05,
  };
}
