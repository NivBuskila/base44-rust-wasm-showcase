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

import { MAX_PARTICLES, STAT } from './constants';
import type { EngineTier } from './engine-loader';
import { ComboBook, comboState, duetState } from './hud-combos';
import { FrameTimer } from './hud-frame-timer';
import { hudAction } from './hud-keys';
import { markup } from './hud-markup';
import { Meter, finite, frameTone } from './hud-meter';
import { PRESETS, type ParamPreset } from './hud-presets';
import { fmtSmall, presence, spellAt } from './hud-readouts';
import { Sparkline } from './hud-spark';
import { statusLine } from './hud-status';
import { GestureTutorial } from './tutorial';
import {
  CELLS,
  GESTURES,
  METERS,
  MODES,
  PARAMS,
  PARTICLE_DETENTS,
  clamp01,
  countToDetent,
  detentToCount,
  thousands,
} from './hud-spec';
import type { HudCallbacks, HudStats, PerceptionStatus, ViewMode } from './types';

/**
 * DOM repaint period. A full paint measures ~0.10 ms in Chromium, so 10 Hz
 * costs ~1 ms per second of wall clock; per frame it would cost six times that
 * to redraw digits nobody can read at 60 Hz.
 */
const PAINT_MS = 100;

/** How long the "show your hands" hint lingers when nothing is detected. */
const HINT_MS = 12_000;

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
  private readonly spark: Sparkline;
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
  private readonly combos: ComboBook;
  private readonly tutorial: GestureTutorial;

  private readonly frameTimer = new FrameTimer();

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
    this.spark = new Sparkline(this.q('canvas.hud-spark'));
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
    // Outside the panel on purpose: sequence progress has to stay visible when
    // the panel is collapsed, which is how most of a session is spent.
    this.combos = new ComboBook(this.root);
    // Also outside the panel: the card sits over the stage where the hand is.
    this.tutorial = new GestureTutorial(this.root, {
      onMode: (on) => this.cb.onPractice(on),
    });
    this.wireControls();
    this.wireKeys();

  }

  /**
   * Accumulates every frame, repaints at 10 Hz.
   *
   * Safe to call with a short or garbage `stats` array: everything is read
   * through {@link finite} and clamped.
   */
  update(s: HudStats): void {
    const now = performance.now();
    this.frameTimer.sample(now);
    this.meters[0]?.sample(s.stepMs);
    this.meters[1]?.sample(s.renderMs);
    this.meters[2]?.sample(s.inferenceMs);

    if (this.hintStartMs === 0) this.hintStartMs = now;

    // Every frame, ahead of the 10 Hz gate: a charge bar that steps at 10 Hz
    // reads as lag in the recognition itself, and each write is diffed inside
    // the component, so an unchanged sequence costs nothing.
    if (s.comboBook) this.combos.setBook(s.comboBook);
    if (s.duetBook) this.combos.setDuetBook(s.duetBook);
    this.combos.update(comboState(s.comboProgress), duetState(s.duetProgress));
    // Every frame too: its hold bar is the feedback that the pose is being read.
    this.tutorial.update(s.spells ?? [], finite(s.stats?.[STAT.HANDS_PRESENT]), now, s.hands ?? null);

    if (now - this.lastPaintMs < PAINT_MS) return;
    this.lastPaintMs = now;

    // The worst frame in the window, not the average: one bad frame in ten is
    // what a user actually notices. The number, the tone and the sparkline all
    // read this single value, so they can never disagree. See
    // `hud-frame-timer.ts` for why an unmeasurable window reports nothing.
    const frameMs = this.frameTimer.read(finite(s.fps));
    if (frameMs > 0.001) {
      this.spark.push(frameMs);
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
    this.spark.draw();

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
    this.spark.dispose();
  }

  // ------------------------------------------------------------------ wiring

  private wireControls(): void {
    this.collapseBtn.addEventListener('click', () => this.setCollapsed(!this.collapsed));
    this.q('[data-act="help"]').addEventListener('click', () => this.setHelp(!this.helpOpen));
    this.q('[data-act="tutorial"]').addEventListener('click', () => this.toggleTutorial());
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
    const act = hudAction(e, this.helpOpen);
    if (!act) return;
    switch (act.kind) {
      case 'mode':
        // No un-hiding here: the view change is visible in the canvas itself,
        // and someone who pressed H wants the screen clean.
        this.setMode(act.mode);
        break;
      case 'camera':
        this.setCamera(!this.cameraOn);
        break;
      case 'overdrive':
        this.setOverdrive(!this.overdriveOn);
        break;
      case 'hideAll':
        this.setHiddenAll(!this.hiddenAll);
        break;
      case 'reset':
        this.cb.onReset();
        break;
      case 'tutorial':
        this.toggleTutorial();
        break;
      case 'help':
        this.setHelp(!this.helpOpen);
        break;
      case 'closeHelp':
        this.setHelp(false);
        break;
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
    if (!collapsed) this.spark.resize();
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

  private toggleTutorial(): void {
    this.tutorial.toggle();
    // Both sit at the bottom of the stage; the card supersedes the hint.
    if (this.tutorial.active && !this.hintDone) {
      this.hintDone = true;
      this.hintEl.classList.add('is-gone');
    }
    if (this.tutorial.active && this.hiddenAll) this.setHiddenAll(false);
  }

  private tickHint(s: HudStats, now: number): void {
    if (this.hintDone) return;
    if (this.tutorial.active) {
      this.hintDone = true;
      this.hintEl.classList.add('is-gone');
      return;
    }
    const cast = s.spells?.[0] !== 'idle' || s.spells?.[1] !== 'idle';
    const seen = finite(s.stats?.[STAT.HANDS_PRESENT]) > 0;
    if (!cast && !seen && now - this.hintStartMs < HINT_MS) return;
    this.hintDone = true;
    this.hintEl.classList.add('is-gone');
  }

  private paintPerception(p: PerceptionStatus): void {
    const { head, why, tone } = statusLine(p);
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

