/**
 * The instrument panel: telemetry, gesture vocabulary, live controls.
 *
 * Four constraints shape this file.
 *
 * - `main.ts` calls `update` once per rendered frame, but touching the DOM
 *   costs more than the whole simulation step, so the panel repaints at 10 Hz
 *   and every timing accumulates its **peak** between paints. Peak-hold, not
 *   the sample that happened to land on the paint tick: a 40 ms hitch every
 *   300 ms is precisely what a frame-budget readout must not hide.
 * - The DOM is built exactly once, in the constructor. `update` only writes
 *   `textContent`, `data-*` flags and CSS custom properties, and only when the
 *   value changed — no `innerHTML`, no element creation, no layout reads.
 * - `#hud` covers the viewport so the panel, the first-run hint and the help
 *   sheet can be placed independently. That only works because everything in
 *   here inherits `pointer-events: none` and individual controls opt back in;
 *   a HUD that swallowed pointer events across a gesture surface would be
 *   worse than no HUD at all.
 * - Nothing here may throw on engine input. `stats` is a raw `Float32Array`
 *   straight out of WASM, so every read is finite-guarded and every control
 *   clamps before it calls back into the engine.
 *
 * The slider ranges mirror the clamps in `Params::set`
 * (`crates/aether-core/src/config.rs`) and `Engine::set_param`; where a clamp
 * is far wider than the band that actually changes the feel, the slider uses
 * the useful sub-range so the default does not sit at 2% of the travel. It
 * never exceeds the clamp, so no value ever gets silently rewritten.
 */

import { FLUID_H, FLUID_W, MAX_PARTICLES, STAT } from './constants';
import type { EngineTier } from './engine-loader';
import { PRESETS, type ParamPreset } from './hud-presets';
import type { HudCallbacks, HudStats, PerceptionStatus, ViewMode } from './types';

/**
 * DOM repaint period. A full paint measures ~0.10 ms in Chromium, so 10 Hz
 * costs ~1 ms per second of wall clock; per frame it would cost six times that
 * to redraw digits nobody can read at 60 Hz.
 */
const PAINT_MS = 100;

/** One frame at 60 Hz. The scale every timing bar is measured against. */
const FRAME_BUDGET_MS = 1000 / 60;

/**
 * Inference is allowed twice the frame budget: `main.ts` throttles it to 30 Hz,
 * so it competes with every other frame rather than every frame.
 */
const INFERENCE_BUDGET_MS = FRAME_BUDGET_MS * 2;

/** Columns in the frame-time sparkline — at 10 Hz this is ~9 s of history. */
const SPARK_SAMPLES = 92;

/**
 * Longer than this between two `update` calls is the tab having been
 * backgrounded, not a slow frame; charting it would peg the sparkline for nine
 * seconds over something the user never saw.
 */
const FRAME_GAP_MAX_MS = 500;

/**
 * `update` calls before `stats.fps` is believable. It is an EMA seeded at zero
 * in `main.ts`, so it reads ~6 fps on the first frame of a 60 Hz session; by 20
 * frames it is within ~12% of the truth. It is only ever the fallback for when
 * no `update` interval was measurable at all.
 */
const FPS_WARMUP_FRAMES = 20;

/** How long the "show your hands" hint lingers when nothing is detected. */
const HINT_MS = 12_000;

/** Bottom of the particle-count slider; below this there is nothing to see. */
const PARTICLE_MIN = 1_000;

/** Slider detents for the log-scale particle control. */
const PARTICLE_DETENTS = 1_000;

/** Bar tone thresholds, as a fraction of the metric's budget. */
const TONE_WARN = 0.6;

/**
 * Whole-frame thresholds in ms, deliberately not `ratio >= 1`.
 * A healthy 60 Hz session measures *exactly* one budget per frame, so a strict
 * comparison would paint a perfect session permanently red. These are the
 * points where a user actually feels it: ~54 fps and ~48 fps.
 */
const FRAME_WARN_MS = 18.5;
const FRAME_OVER_MS = 21;

type Tone = 'ok' | 'warn' | 'over';

interface MeterSpec {
  readonly id: string;
  readonly label: string;
  readonly budget: number;
  readonly digits: number;
}

const METERS: readonly MeterSpec[] = [
  { id: 'step', label: 'step', budget: FRAME_BUDGET_MS, digits: 2 },
  { id: 'render', label: 'render', budget: FRAME_BUDGET_MS, digits: 2 },
  { id: 'infer', label: 'infer', budget: INFERENCE_BUDGET_MS, digits: 1 },
];

interface CellSpec {
  readonly id: string;
  readonly label: string;
}

const CELLS: readonly CellSpec[] = [
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

interface ParamSpec {
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
function thousands(v: number): string {
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
const PARAMS: readonly ParamSpec[] = [
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

interface ModeSpec {
  readonly mode: ViewMode;
  readonly key: string;
  readonly hint: string;
}

const MODES: readonly ModeSpec[] = [
  { mode: 'aether', key: '1', hint: 'dye, particles and bloom' },
  { mode: 'camera', key: '2', hint: 'the camera feed alone' },
  { mode: 'debug', key: '3', hint: 'obstacles, flow and pressure' },
  { mode: 'particles', key: '4', hint: 'particles on black' },
];

interface GestureSpec {
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

const GESTURES: readonly GestureSpec[] = [
  { spell: 'attract', hand: 'closed fist', effect: 'gravity well', icon: ICONS.attract },
  { spell: 'repel', hand: 'open palm', effect: 'push everything out', icon: ICONS.repel },
  { spell: 'vortex', hand: 'pinch', effect: 'spin a vortex', icon: ICONS.vortex },
  { spell: 'ignite', hand: 'point', effect: 'paint a hot trail', icon: ICONS.ignite },
  { spell: 'freeze', hand: 'victory', effect: 'chill and damp', icon: ICONS.freeze },
  { spell: 'shatter', hand: 'thumb up', effect: 'burst outward', icon: ICONS.shatter },
  { spell: 'release', hand: 'open a held fist', effect: 'ring shockwave', icon: ICONS.release },
  { spell: 'warp', hand: 'two hands, rotate', effect: 'warp time', icon: ICONS.warp },
];

const SHORTCUTS: readonly [string, string][] = [
  ['1 – 4', 'aether / camera / debug / particles'],
  ['C', 'camera feed behind the fluid'],
  ['O', 'overdrive: the full 1M particle pool'],
  ['H', 'hide or show this panel'],
  ['R', 'reset the field'],
  ['?', 'this sheet'],
  ['ESC', 'close'],
];

/** Log-scale slider position (0..PARTICLE_DETENTS) -> particle count. */
function detentToCount(detent: number): number {
  const t = clamp01(detent / PARTICLE_DETENTS);
  const n = PARTICLE_MIN * Math.pow(MAX_PARTICLES / PARTICLE_MIN, t);
  // Round to a readable step so the label does not show 118,431.
  const quantum = n < 10_000 ? 500 : n < 100_000 ? 1_000 : 5_000;
  return Math.max(PARTICLE_MIN, Math.min(MAX_PARTICLES, Math.round(n / quantum) * quantum));
}

/** Inverse of {@link detentToCount}, for seeding the slider. */
function countToDetent(count: number): number {
  const n = Math.max(PARTICLE_MIN, Math.min(MAX_PARTICLES, count));
  const t = Math.log(n / PARTICLE_MIN) / Math.log(MAX_PARTICLES / PARTICLE_MIN);
  return Math.round(clamp01(t) * PARTICLE_DETENTS);
}

function clamp01(v: number): number {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}

/** Any non-finite engine value reads as zero rather than poisoning the DOM. */
function finite(v: number | undefined): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** A latched spell name, or `idle` for anything the engine did not send. */
function spellAt(spells: HudStats['spells'] | undefined, slot: number): string {
  const s = spells?.[slot];
  return typeof s === 'string' && s.length > 0 ? s : 'idle';
}

/**
 * Which hand slots hold a hand, from the only two things the engine reports:
 * a *count* and the per-slot latched spell.
 *
 * The count alone is not enough, because the slots are independent — a single
 * right hand lands in slot 1 with slot 0 empty (`packHands` in `perception.ts`
 * keys the slot off handedness). Treating `count > slot` as presence therefore
 * reads a lone right hand as "hand i idle, hand ii absent", which is both cards
 * wrong at exactly the moment one is casting. A non-idle spell pins a slot down:
 * `spells.rs` forces `Spell::Idle` for any slot whose hand is missing, so a named
 * spell can only come from a hand that is there.
 *
 * One case stays genuinely ambiguous: one hand present, nothing latched. Nothing
 * in `HudStats` says which slot it is, so slot 0 is assumed.
 */
function presence(hands: number, spells: HudStats['spells'] | undefined): [boolean, boolean] {
  const cast0 = spellAt(spells, 0) !== 'idle';
  const cast1 = spellAt(spells, 1) !== 'idle';
  if (cast0 && cast1) return [true, true];
  if (hands <= 0) return [false, false];
  if (hands >= 2) return [true, true];
  if (cast1 && !cast0) return [false, true];
  return [true, false];
}

function toneOf(ratio: number): Tone {
  if (ratio >= 1) return 'over';
  if (ratio >= TONE_WARN) return 'warn';
  return 'ok';
}

/** Tone for a whole-frame time in ms. See {@link FRAME_WARN_MS}. */
function frameTone(ms: number): Tone {
  if (ms >= FRAME_OVER_MS) return 'over';
  if (ms >= FRAME_WARN_MS) return 'warn';
  return 'ok';
}

/** Escapes text destined for the one-shot markup build. */
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * A label / track / number triple with peak-hold. `sample` is called every
 * frame; `paint` drains the peak and is called at 10 Hz.
 */
class Meter {
  private peak = 0;
  private lastWidth = -1;
  private lastTone: Tone | '' = '';

  constructor(
    private readonly fill: HTMLElement,
    private readonly value: HTMLElement,
    private readonly budget: number,
    private readonly digits: number,
  ) {}

  sample(ms: number): void {
    const v = finite(ms);
    if (v > this.peak) this.peak = v;
  }

  /** Drops the held peak without touching the DOM, for when nobody is looking. */
  drain(): void {
    this.peak = 0;
  }

  paint(): void {
    const ms = this.peak;
    this.peak = 0;
    const ratio = ms / this.budget;
    const width = Math.round(Math.min(1, ratio) * 100);
    if (width !== this.lastWidth) {
      this.lastWidth = width;
      this.fill.style.setProperty('--v', `${width}%`);
    }
    const tone = toneOf(ratio);
    if (tone !== this.lastTone) {
      this.lastTone = tone;
      this.fill.dataset.tone = tone;
    }
    const text = ms.toFixed(this.digits);
    if (this.value.textContent !== text) this.value.textContent = text;
  }
}

export class Hud {
  private readonly root: HTMLElement;
  private readonly cb: HudCallbacks;

  private readonly panel: HTMLElement;
  private readonly body: HTMLElement;
  private readonly collapseBtn: HTMLButtonElement;
  private readonly fpsEl: HTMLElement;
  private readonly ambientEl: HTMLElement;
  private readonly engineBadge: HTMLElement;
  private readonly meters: Meter[] = [];
  private readonly spark: HTMLCanvasElement;
  private readonly sparkCtx: CanvasRenderingContext2D | null;
  private readonly cells = new Map<string, HTMLElement>();
  private readonly spellEls: HTMLElement[] = [];
  private readonly legendEls = new Map<string, HTMLElement>();
  private readonly warpValue: HTMLElement;
  private readonly warpFill: HTMLElement;
  private readonly percepEl: HTMLElement;
  private readonly percepHead: HTMLElement;
  private readonly percepWhy: HTMLElement;
  private readonly hintEl: HTMLElement;
  private readonly helpEl: HTMLElement;
  private readonly modeBtns: HTMLButtonElement[] = [];
  private readonly camBtn: HTMLButtonElement;
  private readonly camValue: HTMLElement;
  private readonly particleInput: HTMLInputElement;
  private readonly particleValue: HTMLElement;
  private readonly presetBtns = new Map<string, HTMLButtonElement>();

  /** Frame-time history, newest last, as a fill-then-shift window. */
  private readonly history = new Float32Array(SPARK_SAMPLES);
  private historyLen = 0;
  private framePeak = 0;
  private lastUpdateMs = 0;
  /** `update` calls so far, capped at {@link FPS_WARMUP_FRAMES}. */
  private updates = 0;

  private cameraOn = true;
  private overdriveOn = false;
  private readonly overdriveBtn: HTMLButtonElement;
  /** Slider position to restore when overdrive is switched off. */
  private particlesBeforeOverdrive = 0;
  private collapsed = true;
  private hiddenAll = false;
  private helpOpen = false;
  private lastPaintMs = 0;
  private hintStartMs = 0;
  private hintDone = false;
  private lastPercep = '';
  private lastFpsTone = '';
  private lastAmbient: boolean | null = null;
  private readonly resizeObs: ResizeObserver | null = null;

  constructor(root: HTMLElement, callbacks: HudCallbacks) {
    this.root = root;
    this.cb = callbacks;
    this.root.classList.add('hud-root', 'is-collapsed');
    this.root.innerHTML = markup();

    this.panel = this.q('.hud-panel');
    this.body = this.q('.hud-body');
    this.collapseBtn = this.q('[data-act="collapse"]');
    this.fpsEl = this.q('[data-fps]');
    this.ambientEl = this.q('[data-ambient]');
    this.engineBadge = this.q('[data-engine]');
    this.spark = this.q('canvas.hud-spark');
    this.sparkCtx = this.spark.getContext('2d');
    this.warpValue = this.q('[data-warp-value]');
    this.warpFill = this.q('[data-warp-fill]');
    this.percepEl = this.q('.hud-percep');
    this.percepHead = this.q('[data-percep-head]');
    this.percepWhy = this.q('[data-percep-why]');
    this.hintEl = this.q('.hud-hint');
    this.helpEl = this.q('.hud-help');
    this.camBtn = this.q('[data-act="camera"]');
    this.camValue = this.q('[data-camera-value]');
    this.particleInput = this.q('[data-particles]');
    this.particleValue = this.q('[data-particles-value]');
    this.overdriveBtn = this.q('[data-act="overdrive"]');

    for (const m of METERS) {
      this.meters.push(
        new Meter(
          this.q(`[data-meter="${m.id}"] .trk-fill`),
          this.q(`[data-meter="${m.id}"] .m-val`),
          m.budget,
          m.digits,
        ),
      );
    }
    for (const c of CELLS) this.cells.set(c.id, this.q(`[data-cell="${c.id}"] b`));
    for (const g of GESTURES) this.legendEls.set(g.spell, this.q(`[data-spell="${g.spell}"]`));
    this.spellEls.push(this.q('[data-hand="0"] b'), this.q('[data-hand="1"] b'));

    // The status line lives outside the scrolling body — "why is tracking off"
    // must not be something you have to scroll to — so collapsing hides both.
    this.body.hidden = true;
    this.percepEl.hidden = true;
    this.wireControls();
    this.wireKeys();

    // The sparkline is the only canvas in the HUD; keeping its backing store in
    // step with its CSS box is what stops it from looking like a stretched JPEG
    // on a phone or a high-DPI display.
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObs = new ResizeObserver(() => this.resizeSpark());
      this.resizeObs.observe(this.spark);
    }
    this.resizeSpark();
  }

  /**
   * Accumulates every frame, repaints at 10 Hz.
   *
   * Safe to call with a short or garbage `stats` array: everything is read
   * through {@link finite} and clamped.
   */
  update(s: HudStats): void {
    const now = performance.now();
    const gap = this.lastUpdateMs > 0 ? now - this.lastUpdateMs : 0;
    this.lastUpdateMs = now;

    // `update` runs exactly once per rendered frame, so the interval between
    // calls *is* the frame period — and unlike `stats.fps` it is not floored by
    // the loop's own 50 ms dt clamp, which reports a hard stall as a tidy
    // 20 fps. A gap over half a second is the tab having been away, not a slow
    // frame, so it is dropped rather than spiking the history.
    if (gap > 0 && gap <= FRAME_GAP_MAX_MS && gap > this.framePeak) this.framePeak = gap;
    if (this.updates < FPS_WARMUP_FRAMES) this.updates++;
    this.meters[0]?.sample(s.stepMs);
    this.meters[1]?.sample(s.renderMs);
    this.meters[2]?.sample(s.inferenceMs);

    if (this.hintStartMs === 0) this.hintStartMs = now;
    if (now - this.lastPaintMs < PAINT_MS) return;
    this.lastPaintMs = now;

    // The worst frame in the window, not the average: one bad frame in ten is
    // what a user actually notices. When nothing was measurable — every gap in
    // this window was long enough to look like a backgrounded tab, which is
    // what a cold start looks like — fall back to the loop's own average. The
    // number, the tone and the sparkline all read this single value, so they
    // can never disagree.
    const loopMs = finite(s.fps) > 0.001 ? Math.min(1000, 1000 / finite(s.fps)) : 0;
    // `stats.fps` is only an honest fallback once the loop's own smoothing has
    // warmed up: it is an EMA seeded at zero, so the first frame of a session
    // reports ~6 fps at a true 60 Hz. Charting that invents a 166 ms frame that
    // never happened, and because the sparkline scales to the worst sample in
    // its window, that phantom flattens the next nine seconds of real history
    // into two pixels. With nothing measured and nothing trustworthy to fall
    // back on, the chip keeps its placeholder instead of guessing.
    const frameMs =
      this.framePeak > 0.001 ? this.framePeak : this.updates >= FPS_WARMUP_FRAMES ? loopMs : 0;
    this.framePeak = 0;
    if (frameMs > 0.001) {
      this.push(frameMs);
      const worstFps = 1000 / frameMs;
      this.setText(this.fpsEl, worstFps >= 10 ? worstFps.toFixed(0) : worstFps.toFixed(1));
      const fpsTone = frameTone(frameMs);
      if (fpsTone !== this.lastFpsTone) {
        this.lastFpsTone = fpsTone;
        this.fpsEl.dataset.tone = fpsTone;
      }
    }

    const st = s.stats;
    const ambient = finite(st?.[STAT.AMBIENT]) > 0.5;
    if (ambient !== this.lastAmbient) {
      this.lastAmbient = ambient;
      this.ambientEl.hidden = !ambient;
    }
    this.tickHint(s, now);

    if (this.hiddenAll || this.collapsed) {
      // Still drain the peak-holds: expanding the panel after a minute must not
      // show the spike from whenever it happened to be collapsed. The sparkline
      // history keeps filling, so the moment it opens it has real context.
      for (const m of this.meters) m.drain();
      return;
    }

    for (const m of this.meters) m.paint();
    this.drawSpark();

    const hands = Math.round(finite(st?.[STAT.HANDS_PRESENT]));
    const maskOn = finite(st?.[STAT.MASK_PRESENT]) > 0.5;
    this.cell('particles', thousands(finite(st?.[STAT.PARTICLES_ALIVE])));
    this.cell('energy', finite(st?.[STAT.FLUID_ENERGY]).toFixed(2));
    this.cell('speed', finite(st?.[STAT.FLUID_MAX_SPEED]).toFixed(1));
    this.cell('divergence', fmtSmall(finite(st?.[STAT.FLUID_DIVERGENCE])));
    this.cell('motion', finite(st?.[STAT.MOTION_ENERGY]).toFixed(3));
    this.cell('hands', `${Math.max(0, Math.min(2, hands))}`);
    this.cell('body', maskOn ? `${(finite(st?.[STAT.MASK_COVERAGE]) * 100).toFixed(0)}%` : '—');
    this.cell('repairs', finite(st?.[STAT.NAN_REPAIRS]).toFixed(0));

    const timeScale = finite(st?.[STAT.TIME_SCALE]);
    const warping = Math.abs(timeScale - 1) > 0.05;
    this.setText(this.warpValue, `${timeScale.toFixed(2)}×`);
    // Full travel is the engine's own 4x clamp, so 1x sits a quarter along:
    // left of the thumb is slow motion, right of it is fast.
    const warpPct = `${Math.round(clamp01(timeScale / 4) * 100)}%`;
    if (this.warpFill.style.getPropertyValue('--v') !== warpPct) {
      this.warpFill.style.setProperty('--v', warpPct);
    }
    this.setData(this.warpFill, 'tone', warping ? 'on' : 'off');

    const held = presence(hands, s.spells);
    for (let i = 0; i < this.spellEls.length; i++) {
      const el = this.spellEls[i];
      if (!el) continue;
      const spell = spellAt(s.spells, i);
      const present = held[i] === true;
      this.setText(el, !present ? 'no hand' : spell);
      const card = el.parentElement;
      if (card) {
        this.setData(card, 'state', !present ? 'off' : spell === 'idle' ? 'idle' : 'cast');
      }
    }

    for (const [spell, el] of this.legendEls) {
      const on = spell === 'warp' ? warping : s.spells?.[0] === spell || s.spells?.[1] === spell;
      this.setData(el, 'on', on ? '1' : '');
    }

    this.paintPerception(s.perception);
  }

  /**
   * Shows which engine build is running. Static for the session, so it is set
   * once rather than re-derived on every `update`.
   */
  setEngineTier(tier: EngineTier): void {
    const label = tier.name === 'threads' ? `${tier.threads} thr · simd` : '1 thr · simd';
    this.cell('engine', label);
    this.engineBadge.hidden = tier.name !== 'threads';
    this.setText(this.engineBadge, `${tier.threads} threads`);
    this.engineBadge.title = `Rust engine on ${tier.threads} worker threads (${tier.reason})`;
  }

  /** Detaches global listeners. Not used by `main.ts`; here for teardown. */
  dispose(): void {
    window.removeEventListener('keydown', this.onKey);
    this.resizeObs?.disconnect();
  }

  // ------------------------------------------------------------------ wiring

  private wireControls(): void {
    this.collapseBtn.addEventListener('click', () => this.setCollapsed(!this.collapsed));
    this.q('[data-act="help"]').addEventListener('click', () => this.setHelp(!this.helpOpen));
    this.q('[data-act="help-close"]').addEventListener('click', () => this.setHelp(false));
    // Clicking the backdrop dismisses; clicking the sheet must not.
    this.helpEl.addEventListener('click', (e) => {
      if (e.target === this.helpEl) this.setHelp(false);
    });

    for (const spec of MODES) {
      const btn = this.q<HTMLButtonElement>(`[data-mode="${spec.mode}"]`);
      btn.addEventListener('click', () => this.setMode(spec.mode));
      this.modeBtns.push(btn);
    }
    this.camBtn.addEventListener('click', () => this.setCamera(!this.cameraOn));
    this.overdriveBtn.addEventListener('click', () => this.setOverdrive(!this.overdriveOn));
    this.q('[data-act="reset"]').addEventListener('click', () => this.cb.onReset());

    this.particleInput.value = String(countToDetent(120_000));
    this.setText(this.particleValue, thousands(detentToCount(countToDetent(120_000))));
    // Live label on drag, commit on release: resizing the pool reallocates and
    // re-seeds up to 220k particles, so firing it per input event would stutter
    // the very frame rate this panel is reporting.
    this.particleInput.addEventListener('input', () => {
      this.setText(this.particleValue, thousands(detentToCount(this.particleInput.valueAsNumber)));
    });
    this.particleInput.addEventListener('change', () => {
      // Dragging the slider by hand leaves overdrive; it is a preset, not a lock.
      if (this.overdriveOn) this.setOverdrive(false, false);
      this.cb.onParticleCount(detentToCount(this.particleInput.valueAsNumber));
    });

    for (const preset of PRESETS) {
      const btn = this.q<HTMLButtonElement>(`[data-preset="${preset.id}"]`);
      btn.addEventListener('click', () => this.applyPreset(preset));
      this.presetBtns.set(preset.id, btn);
    }

    for (const p of PARAMS) {
      const input = this.q<HTMLInputElement>(`[data-param="${p.key}"]`);
      const value = this.q(`[data-param-value="${p.key}"]`);
      input.addEventListener('input', () => {
        const raw = input.valueAsNumber;
        const v = Number.isFinite(raw) ? Math.min(p.max, Math.max(p.min, raw)) : p.value;
        this.setText(value, p.fmt(v));
        this.cb.onParam(p.key, v);
        // Hand-tuning past a preset means the panel is no longer showing it.
        this.markPreset(null);
      });
    }
  }

  /**
   * Writes every parameter of a preset, including the ones it does not name —
   * those fall back to the slider's own default, so the result is the same
   * whatever was set before, rather than a mix of two presets.
   */
  private applyPreset(preset: ParamPreset): void {
    for (const p of PARAMS) {
      const v = preset.values[p.key] ?? p.value;
      const input = this.q<HTMLInputElement>(`[data-param="${p.key}"]`);
      input.valueAsNumber = v;
      this.setText(this.q(`[data-param-value="${p.key}"]`), p.fmt(v));
      this.cb.onParam(p.key, v);
    }
    this.markPreset(preset.id);
  }

  /** Lights the active preset button, or none of them after a manual edit. */
  private markPreset(id: string | null): void {
    for (const [key, btn] of this.presetBtns) {
      btn.setAttribute('aria-pressed', String(key === id));
    }
  }

  private wireKeys(): void {
    window.addEventListener('keydown', this.onKey);
  }

  private readonly onKey = (e: KeyboardEvent): void => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // Typing in a control — including dragging a slider with the arrow keys —
    // must never be hijacked by a single-letter shortcut.
    const t = e.target;
    if (t instanceof HTMLElement) {
      const tag = t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable) return;
    }

    const mode = MODES.find((m) => m.key === e.key);
    if (mode) {
      // No un-hiding here: the view change is visible in the canvas itself, and
      // someone who pressed H wants the screen clean.
      this.setMode(mode.mode);
    } else if (e.key === 'c' || e.key === 'C') {
      this.setCamera(!this.cameraOn);
    } else if (e.key === 'o' || e.key === 'O') {
      this.setOverdrive(!this.overdriveOn);
    } else if (e.key === 'h' || e.key === 'H') {
      this.setHiddenAll(!this.hiddenAll);
    } else if (e.key === 'r' || e.key === 'R') {
      this.cb.onReset();
    } else if (e.key === '?' || e.key === '/') {
      this.setHelp(!this.helpOpen);
    } else if (e.key === 'Escape') {
      if (!this.helpOpen) return;
      this.setHelp(false);
    } else {
      return;
    }
    e.preventDefault();
  };

  // ------------------------------------------------------------------- state

  private setMode(mode: ViewMode): void {
    for (const btn of this.modeBtns) {
      btn.setAttribute('aria-pressed', btn.dataset.mode === mode ? 'true' : 'false');
    }
    this.cb.onViewMode(mode);
  }

  private setCamera(on: boolean): void {
    this.cameraOn = on;
    this.camBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    this.setText(this.camValue, on ? 'on' : 'off');
    this.cb.onToggleCamera(on);
  }

  /**
   * Toggles the 1M-particle preset. `restoreSlider` is false when the user is
   * the one moving the slider, so their new position is not overwritten.
   */
  private setOverdrive(on: boolean, restoreSlider = true): void {
    if (on === this.overdriveOn) return;
    this.overdriveOn = on;
    this.overdriveBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    if (on) {
      this.particlesBeforeOverdrive = this.particleInput.valueAsNumber;
      this.particleInput.value = String(PARTICLE_DETENTS);
      this.setText(this.particleValue, thousands(MAX_PARTICLES));
    } else if (restoreSlider) {
      this.particleInput.value = String(this.particlesBeforeOverdrive);
      this.setText(this.particleValue, thousands(detentToCount(this.particlesBeforeOverdrive)));
    }
    this.cb.onOverdrive(on);
  }

  private setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    this.root.classList.toggle('is-collapsed', collapsed);
    this.body.hidden = collapsed;
    this.percepEl.hidden = collapsed;
    this.collapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    this.collapseBtn.setAttribute('aria-label', collapsed ? 'Expand the panel' : 'Collapse to fps');
    if (!collapsed) this.resizeSpark();
  }

  /**
   * `H` hides the panel and the hint but not the help sheet: `?` has to work
   * from the hidden state, otherwise the way back is unguessable.
   */
  private setHiddenAll(hidden: boolean): void {
    this.hiddenAll = hidden;
    this.root.classList.toggle('is-hidden', hidden);
    this.panel.setAttribute('aria-hidden', hidden ? 'true' : 'false');
  }

  private setHelp(open: boolean): void {
    this.helpOpen = open;
    this.helpEl.classList.toggle('is-open', open);
    this.helpEl.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open) this.q<HTMLButtonElement>('[data-act="help-close"]').focus();
  }

  private tickHint(s: HudStats, now: number): void {
    if (this.hintDone) return;
    const cast = s.spells?.[0] !== 'idle' || s.spells?.[1] !== 'idle';
    const seen = finite(s.stats?.[STAT.HANDS_PRESENT]) > 0;
    if (!cast && !seen && now - this.hintStartMs < HINT_MS) return;
    this.hintDone = true;
    this.hintEl.classList.add('is-gone');
  }

  private paintPerception(p: PerceptionStatus): void {
    let head: string;
    let why: string;
    let tone: string;
    if (p.kind === 'ready') {
      head = 'hand + body tracking live';
      why =
        p.delegate === 'GPU'
          ? 'GPU delegate'
          : 'CPU delegate — inference will cost more per frame';
      tone = 'ok';
    } else if (p.kind === 'loading') {
      head = 'loading the vision models';
      why = 'optical flow is already driving the fluid';
      tone = 'wait';
    } else {
      head = 'optical flow only';
      why = `${p.reason} Move, and the fluid still answers.`;
      tone = 'warn';
    }
    // The status changes almost never; a string compare is cheaper than three
    // DOM writes at 10 Hz.
    const sig = `${tone}|${head}|${why}`;
    if (sig === this.lastPercep) return;
    this.lastPercep = sig;
    this.percepEl.dataset.tone = tone;
    this.percepHead.textContent = head;
    this.percepWhy.textContent = why;
  }

  // ---------------------------------------------------------------- painting

  private push(ms: number): void {
    if (this.historyLen < SPARK_SAMPLES) {
      this.history[this.historyLen++] = ms;
    } else {
      this.history.copyWithin(0, 1);
      this.history[SPARK_SAMPLES - 1] = ms;
    }
  }

  private resizeSpark(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.spark.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.spark.clientHeight * dpr));
    if (this.spark.width !== w) this.spark.width = w;
    if (this.spark.height !== h) this.spark.height = h;
    this.drawSpark();
  }

  private drawSpark(): void {
    const ctx = this.sparkCtx;
    if (!ctx) return;
    const w = this.spark.width;
    const h = this.spark.height;
    if (w < 2 || h < 2) return;

    ctx.clearRect(0, 0, w, h);
    // Scale to two frame budgets, or to the worst spike, so a 30 fps session is
    // visibly over budget rather than silently clipped at the top.
    let worst = FRAME_BUDGET_MS * 2;
    for (let i = 0; i < this.historyLen; i++) {
      const v = this.history[i] ?? 0;
      if (v > worst) worst = v;
    }

    const budgetY = h - (FRAME_BUDGET_MS / worst) * h;
    ctx.fillStyle = 'rgba(126, 166, 214, 0.22)';
    ctx.fillRect(0, Math.round(budgetY), w, 1);

    const colW = w / SPARK_SAMPLES;
    const barW = Math.max(1, colW * 0.68);
    // Right-aligned: the newest sample always sits at the right edge, so a
    // partly-filled window reads as history scrolling in rather than a ramp.
    const offset = SPARK_SAMPLES - this.historyLen;
    for (let i = 0; i < this.historyLen; i++) {
      const v = this.history[i] ?? 0;
      const tone = frameTone(v);
      ctx.fillStyle =
        tone === 'over' ? '#ff6b8b' : tone === 'warn' ? '#ffc65c' : 'rgba(53, 214, 255, 0.85)';
      const bh = Math.max(1, (v / worst) * h);
      ctx.fillRect((offset + i) * colW, h - bh, barW, bh);
    }
  }

  private cell(id: string, text: string): void {
    const el = this.cells.get(id);
    if (el) this.setText(el, text);
  }

  private setText(el: HTMLElement, text: string): void {
    if (el.textContent !== text) el.textContent = text;
  }

  /**
   * Both of these read before they write. An unconditional write is a style
   * invalidation on that element, and at 10 Hz across ~20 nodes that is real
   * work for nothing: almost every field is unchanged between paints.
   */
  private setData(el: HTMLElement, key: string, value: string): void {
    if (el.dataset[key] !== value) el.dataset[key] = value;
  }

  /** The markup is ours and one line old; a miss here is a bug in this file. */
  private q<T extends HTMLElement>(sel: string): T {
    const el = this.root.querySelector(sel);
    if (!el) throw new Error(`hud: missing element ${sel}`);
    return el as T;
  }
}

/** `0.00042` -> `4.2e-4`; keeps the divergence column from reading `0.0000`. */
function fmtSmall(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 0.001) return v.toExponential(1).replace('e-', 'e−');
  return v.toFixed(4);
}

function meterRow(m: MeterSpec): string {
  return `
    <div class="m-row" data-meter="${m.id}">
      <span class="m-key">${esc(m.label)}</span>
      <span class="trk"><i class="trk-fill" data-tone="ok"></i></span>
      <span class="m-val">0.00</span>
      <span class="m-unit">ms</span>
    </div>`;
}

function cellBox(c: CellSpec): string {
  return `
    <div class="c-box" data-cell="${c.id}">
      <span>${esc(c.label)}</span>
      <b>—</b>
    </div>`;
}

function handCard(slot: number, name: string): string {
  return `
    <div class="h-card" data-hand="${slot}" data-state="off">
      <span>${esc(name)}</span>
      <b>no hand</b>
    </div>`;
}

/**
 * The legend appears twice — in the panel and in the help sheet. Only the panel
 * copy carries `data-spell`, so the live-highlight selectors stay unambiguous.
 */
function legendRow(g: GestureSpec, live: boolean): string {
  const id = live ? `data-spell="${g.spell}" data-on=""` : '';
  return `
    <li class="g-row" ${id}>
      <span class="g-ico">${svg(g.icon)}</span>
      <span class="g-txt"><b>${esc(g.effect)}</b><i>${esc(g.hand)}</i></span>
    </li>`;
}

function paramRow(p: ParamSpec): string {
  return `
    <label class="p-row">
      <span class="p-key">${esc(p.label)}</span>
      <input
        type="range"
        data-param="${p.key}"
        min="${p.min}"
        max="${p.max}"
        step="${p.step}"
        value="${p.value}"
        aria-label="${esc(p.label)}"
      />
      <span class="p-val" data-param-value="${p.key}">${esc(p.fmt(p.value))}</span>
    </label>`;
}

function svg(paths: string): string {
  return (
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    `stroke-linejoin="round">${paths}</svg>`
  );
}

/** The whole panel, built once. */
function markup(): string {
  const presetBtns = PRESETS.map(
    (p) => `
      <button
        type="button"
        class="seg-btn"
        data-preset="${p.id}"
        aria-pressed="${p.id === 'default'}"
        title="${esc(p.hint)}"
      >${esc(p.label)}</button>`,
  ).join('');
  const modeBtns = MODES.map(
    (m) => `
      <button
        type="button"
        class="seg-btn"
        data-mode="${m.mode}"
        aria-pressed="${m.mode === 'aether' ? 'true' : 'false'}"
        title="${esc(m.hint)} (${m.key})"
      >${esc(m.mode)}<span class="seg-key">${m.key}</span></button>`,
  ).join('');

  const keyRows = SHORTCUTS.map(
    ([k, what]) => `<div class="k-row"><kbd>${esc(k)}</kbd><span>${esc(what)}</span></div>`,
  ).join('');

  const chips = GESTURES.map(
    (g) => `<span class="chip">${svg(g.icon)}${esc(g.effect)}</span>`,
  ).join('');

  return `
  <aside class="hud-panel" aria-label="Aether telemetry and controls">
    <header class="hud-head">
      <div class="hud-mark">
        <b>AETHER</b>
        <span>fluid reality engine</span>
      </div>
      <div class="hud-fps">
        <b data-fps data-tone="ok">—</b><span>fps</span>
      </div>
      <div class="hud-acts">
        <button type="button" class="ico-btn" data-act="help" aria-label="Help and keyboard shortcuts" title="Help (?)">
          ${svg('<circle cx="12" cy="12" r="9"/><path d="M9.6 9.2a2.5 2.5 0 1 1 3.6 2.3c-.8.5-1.2 1-1.2 2"/><path d="M12 17.2h.01"/>')}
        </button>
        <button type="button" class="ico-btn" data-act="collapse" aria-expanded="false" aria-label="Expand the panel" title="Collapse or expand">
          ${svg('<path class="chev-d" d="M6 9.5l6 5.5 6-5.5"/><path class="chev-u" d="M6 14.5l6-5.5 6 5.5"/>')}
        </button>
      </div>
      <span class="hud-badge" data-ambient hidden>ambient</span>
      <span class="hud-badge is-engine" data-engine hidden></span>
    </header>

    <div class="hud-body">
      <section class="hud-sec">
        <h2>frame budget<span>16.7 ms @ 60 hz</span></h2>
        ${METERS.map(meterRow).join('')}
        <canvas class="hud-spark" aria-hidden="true"></canvas>
      </section>

      <section class="hud-sec">
        <h2>field</h2>
        <div class="c-grid">${CELLS.map(cellBox).join('')}</div>
      </section>

      <section class="hud-sec">
        <h2>view</h2>
        <div class="seg" role="group" aria-label="View mode">${modeBtns}</div>
        <button type="button" class="row-btn" data-act="camera" aria-pressed="true">
          <span>camera feed</span><b data-camera-value>on</b>
        </button>
        <label class="p-row">
          <span class="p-key">particles</span>
          <input type="range" data-particles min="0" max="${PARTICLE_DETENTS}" step="1" value="0" aria-label="Particle count" />
          <span class="p-val" data-particles-value>120k</span>
        </label>
        <button type="button" class="row-btn is-overdrive" data-act="overdrive" aria-pressed="false">
          <span>overdrive · 1M particles</span><b>O</b>
        </button>
        <button type="button" class="row-btn is-action" data-act="reset">
          <span>reset the field</span><b>R</b>
        </button>
      </section>

      <section class="hud-sec">
        <h2>spells<span>latched per hand</span></h2>
        <div class="h-grid">${handCard(0, 'hand i')}${handCard(1, 'hand ii')}</div>
        <div class="m-row m-warp">
          <span class="m-key">time</span>
          <span class="trk"><i class="trk-fill" data-warp-fill data-tone="off"></i></span>
          <span class="m-val" data-warp-value>1.00×</span>
          <span class="m-unit"></span>
        </div>
        <ul class="g-list">${GESTURES.map((g) => legendRow(g, true)).join('')}</ul>
      </section>

      <details class="hud-sec hud-adv">
        <summary>engine parameters</summary>
        <div class="seg is-presets" role="group" aria-label="Parameter presets">${presetBtns}</div>
        ${PARAMS.map(paramRow).join('')}
      </details>
    </div>

    <p class="hud-percep" data-tone="wait">
      <i class="dot" aria-hidden="true"></i>
      <b data-percep-head>starting up</b>
      <span data-percep-why>optical flow is already driving the fluid</span>
    </p>
  </aside>

  <div class="hud-hint" aria-hidden="true">
    <b>show your hands</b>
    <div class="chips">${chips}</div>
    <span class="hint-keys">? gesture guide &middot; H hide</span>
  </div>

  <div class="hud-help" role="dialog" aria-modal="false" aria-label="Aether reference" aria-hidden="true">
    <div class="help-sheet">
      <header>
        <b>AETHER</b><span>reference</span>
        <button type="button" class="ico-btn" data-act="help-close" aria-label="Close help" title="Close (Esc)">
          ${svg('<path d="M6 6l12 12M18 6 6 18"/>')}
        </button>
      </header>
      <div class="help-cols">
        <div>
          <h3>gestures</h3>
          <ul class="g-list is-static">${GESTURES.map((g) => legendRow(g, false)).join('')}</ul>
        </div>
        <div>
          <h3>keys</h3>
          <div class="k-list">${keyRows}</div>
          <h3>how it works</h3>
          <p class="help-note">
            Hand and body landmarks steer a ${FLUID_W}×${FLUID_H} fluid solver
            compiled to WebAssembly. With no camera or no models, optical
            flow — or the engine's own ambient drive — keeps the field alive.
          </p>
        </div>
      </div>
    </div>
  </div>`;
}
