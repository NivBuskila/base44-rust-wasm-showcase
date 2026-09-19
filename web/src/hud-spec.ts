/**
 * The HUD's static vocabulary: which meters, cells, sliders, view modes and
 * gestures the panel shows, and the particle slider's log-scale mapping.
 *
 * Pure data and pure functions — nothing here touches the DOM.
 */

import { MAX_PARTICLES } from './constants';
import type { ViewMode } from './types';

/** One frame at 60 Hz. The scale every timing bar is measured against. */
export const FRAME_BUDGET_MS = 1000 / 60;

/**
 * Inference is allowed twice the frame budget: `main.ts` throttles it to 30 Hz,
 * so it competes with every other frame rather than every frame.
 */
const INFERENCE_BUDGET_MS = FRAME_BUDGET_MS * 2;

/** Bottom of the particle-count slider; below this there is nothing to see. */
const PARTICLE_MIN = 1_000;

/** Slider detents for the log-scale particle control. */
export const PARTICLE_DETENTS = 1_000;

export interface MeterSpec {
  readonly id: string;
  readonly label: string;
  readonly budget: number;
  readonly digits: number;
}

export const METERS: readonly MeterSpec[] = [
  { id: 'step', label: 'step', budget: FRAME_BUDGET_MS, digits: 2 },
  { id: 'render', label: 'render', budget: FRAME_BUDGET_MS, digits: 2 },
  { id: 'infer', label: 'infer', budget: INFERENCE_BUDGET_MS, digits: 1 },
];

export interface CellSpec {
  readonly id: string;
  readonly label: string;
}

export const CELLS: readonly CellSpec[] = [
  { id: 'engine', label: 'engine' },
  { id: 'particles', label: 'particles' },
  { id: 'energy', label: 'energy' },
  { id: 'speed', label: 'max speed' },
  { id: 'divergence', label: 'divergence' },
  { id: 'motion', label: 'motion' },
  { id: 'hands', label: 'hands' },
  { id: 'body', label: 'body' },
  { id: 'repairs', label: 'nan repairs' },
];

export interface ParamSpec {
  readonly key: string;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly value: number;
  readonly fmt: (v: number) => string;
}

const d1 = (v: number): string => v.toFixed(1);

const d2 = (v: number): string => v.toFixed(2);

/** `120000` -> `120k`, `1500` -> `1.5k`. Keeps the columns narrow. */
export function thousands(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) < 1000) return v.toFixed(0);
  const k = v / 1000;
  return `${Math.abs(k) < 10 ? k.toFixed(1) : k.toFixed(0)}k`;
}

/**
 * The parameters that change how the fluid *feels*. Defaults match
 * `Params::default` and `ParticleConfig::default` so the panel reads true
 * before anything is touched — the HUD is deliberately not told the engine's
 * state at construction, and sending these on boot would be a pointless write.
 */
export const PARAMS: readonly ParamSpec[] = [
  { key: 'vorticity', label: 'vorticity', min: 0, max: 60, step: 0.5, value: 14, fmt: d1 },
  // Both dissipations clamp to 0..10 in Rust, but past ~3 per second the field
  // is gone inside a frame or two and the whole top of the travel is the same
  // black screen, so the slider stops where the range is still expressive.
  { key: 'dye_dissipation', label: 'dye decay', min: 0, max: 3, step: 0.01, value: 1, fmt: d2 },
  {
    key: 'velocity_dissipation',
    label: 'flow decay',
    min: 0,
    max: 2,
    step: 0.01,
    value: 0.3,
    fmt: d2,
  },
  { key: 'hand_force', label: 'hand force', min: 0, max: 8, step: 0.05, value: 1, fmt: d2 },
  { key: 'flow_force', label: 'flow force', min: 0, max: 8, step: 0.05, value: 0.45, fmt: d2 },
  { key: 'curl_influence', label: 'curl swirl', min: 0, max: 20, step: 0.1, value: 2.2, fmt: d1 },
  {
    key: 'time_scale',
    label: 'time scale',
    min: 0.05,
    max: 4,
    step: 0.05,
    value: 1,
    fmt: (v) => `${v.toFixed(2)}×`,
  },
  { key: 'body_push', label: 'body push', min: 0, max: 4, step: 0.05, value: 1, fmt: d2 },
  {
    key: 'spawn_rate',
    label: 'spawn rate',
    min: 0,
    max: 400_000,
    step: 2_000,
    value: 30_000,
    fmt: (v) => `${thousands(v)}/s`,
  },
  // How long a particle lives before it respawns somewhere random. High up the
  // travel the pool barely recycles, so a gathered cloud stays gathered
  // instead of dissolving under the hand holding it.
  {
    key: 'particle_life',
    label: 'particle life',
    min: 0.2,
    max: 30,
    step: 0.1,
    value: 4.5,
    fmt: (v) => `${v.toFixed(1)}s`,
  },
];

export interface ModeSpec {
  readonly mode: ViewMode;
  readonly key: string;
  readonly hint: string;
}

export const MODES: readonly ModeSpec[] = [
  { mode: 'aether', key: '1', hint: 'dye, particles and bloom' },
  { mode: 'camera', key: '2', hint: 'the camera feed alone' },
  { mode: 'blend', key: '3', hint: 'the camera feed under the full aether look' },
  { mode: 'debug', key: '4', hint: 'obstacles, flow and pressure' },
  { mode: 'particles', key: '5', hint: 'particles on black' },
];

export interface GestureSpec {
  /** Matches `Spell::name` in `aether_core::gesture`, or `warp` for time. */
  readonly spell: string;
  readonly hand: string;
  readonly effect: string;
  readonly icon: string;
}

/**
 * Abstract effect glyphs rather than little hands: the row already names the
 * hand shape, and a 18 px pictogram of a fist is unreadable while "everything
 * rushes inward" is not.
 */
const ICONS: Record<string, string> = {
  attract:
    '<circle cx="12" cy="12" r="2.6" fill="currentColor" stroke="none"/>' +
    '<path d="M9.6 3.4 12 6.2l2.4-2.8M9.6 20.6 12 17.8l2.4 2.8M3.4 9.6 6.2 12l-2.8 2.4M20.6 9.6 17.8 12l2.8 2.4"/>',
  repel:
    '<circle cx="12" cy="12" r="2.8"/>' +
    '<path d="M9.6 6.2 12 3.4l2.4 2.8M9.6 17.8 12 20.6l2.4-2.8M6.2 9.6 3.4 12l2.8 2.4M17.8 9.6 20.6 12l-2.8 2.4"/>',
  vortex:
    '<path d="M12 12.4c0-1.5 1.2-2.7 2.7-2.7 2.1 0 3.9 1.8 3.9 4 0 3-2.5 5.5-5.6 5.5-4 0-7.2-3.3-7.2-7.3 0-4.9 4-8.9 8.9-8.9"/>',
  ignite:
    '<path d="M12 3.4c3.3 3.8 5.1 6.3 5.1 9.1a5.1 5.1 0 0 1-10.2 0c0-2 .9-3.6 2.6-5.6.5 1.4 1.2 2.2 2 2.4-.2-2 0-3.9.5-5.9z"/>',
  freeze:
    '<path d="M12 3v18M4.2 7.5l15.6 9M19.8 7.5l-15.6 9"/>' +
    '<path d="M9.6 5.2 12 6.7l2.4-1.5M9.6 18.8 12 17.3l2.4 1.5"/>',
  shatter:
    '<circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/>' +
    '<path d="M12 6V3M12 21v-3M6 12H3M21 12h-3M7.2 7.2 5.1 5.1M18.9 18.9l-2.1-2.1M7.2 16.8l-2.1 2.1M18.9 5.1l-2.1 2.1"/>',
  warp:
    '<path d="M20 12a8 8 0 1 1-2.7-6"/><path d="M20.2 3.6V7.8h-4.2"/><path d="M12 7.8v4.6l3 1.8"/>',
  release:
    '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="3.2" stroke-dasharray="2 2"/>' +
    '<path d="M12 1.5v1.8M12 20.7v1.8M1.5 12h1.8M20.7 12h1.8"/>',
};

export const GESTURES: readonly GestureSpec[] = [
  { spell: 'attract', hand: 'closed fist', effect: 'gravity well', icon: ICONS.attract },
  { spell: 'repel', hand: 'open palm', effect: 'push everything out', icon: ICONS.repel },
  { spell: 'vortex', hand: 'pinch', effect: 'spin a vortex', icon: ICONS.vortex },
  { spell: 'ignite', hand: 'point', effect: 'paint a hot trail', icon: ICONS.ignite },
  { spell: 'freeze', hand: 'victory', effect: 'chill and damp', icon: ICONS.freeze },
  { spell: 'shatter', hand: 'thumb up', effect: 'burst outward', icon: ICONS.shatter },
  { spell: 'release', hand: 'open a held fist', effect: 'ring shockwave', icon: ICONS.release },
  { spell: 'warp', hand: 'two hands, rotate', effect: 'warp time', icon: ICONS.warp },
];

export const SHORTCUTS: readonly [string, string][] = [
  ['1 – 5', 'aether / camera / blend / debug / particles'],
  ['C', 'camera feed behind the fluid'],
  ['O', 'overdrive: the full 1M particle pool'],
  ['H', 'hide or show this panel'],
  ['R', 'reset the field'],
  ['?', 'this sheet'],
  ['ESC', 'close'],
];

/** Log-scale slider position (0..PARTICLE_DETENTS) -> particle count. */
export function detentToCount(detent: number): number {
  const t = clamp01(detent / PARTICLE_DETENTS);
  const n = PARTICLE_MIN * Math.pow(MAX_PARTICLES / PARTICLE_MIN, t);
  // Round to a readable step so the label does not show 118,431.
  const quantum = n < 10_000 ? 500 : n < 100_000 ? 1_000 : 5_000;
  return Math.max(PARTICLE_MIN, Math.min(MAX_PARTICLES, Math.round(n / quantum) * quantum));
}

/** Inverse of {@link detentToCount}, for seeding the slider. */
export function countToDetent(count: number): number {
  const n = Math.max(PARTICLE_MIN, Math.min(MAX_PARTICLES, count));
  const t = Math.log(n / PARTICLE_MIN) / Math.log(MAX_PARTICLES / PARTICLE_MIN);
  return Math.round(clamp01(t) * PARTICLE_DETENTS);
}

export function clamp01(v: number): number {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}
