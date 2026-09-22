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

import { STAT } from "./constants";
import type { EngineTier } from "./engine-loader";
import { ComboBook, comboState, duetState } from "./hud-combos";
import { hudAction } from "./hud-keys";
import { markup } from "./hud-markup";
import { finite } from "./hud-meter";
import { ParamControls } from "./hud-params";
import { PoolControl } from "./hud-pool";
import { Telemetry } from "./hud-telemetry";
import { GestureTutorial } from "./tutorial";
import { MODES } from "./hud-spec";
import type { HudCallbacks, HudStats, ViewMode } from "./types";

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
  /** Every readout in the panel; see `hud-telemetry.ts`. */
  private readonly telemetry: Telemetry;
  private readonly hintEl: HTMLElement;
  private readonly helpEl: HTMLElement;
  private readonly modeBtns: HTMLButtonElement[] = [];
  private readonly camBtn: HTMLButtonElement;
  private readonly camValue: HTMLElement;
  private readonly pool: PoolControl;
  private readonly combos: ComboBook;
  private readonly tutorial: GestureTutorial;

  private cameraOn = true;
  private collapsed = true;
  private hiddenAll = false;
  private helpOpen = false;
  private lastPaintMs = 0;
  private hintStartMs = 0;
  private hintDone = false;

  constructor(root: HTMLElement, callbacks: HudCallbacks) {
    this.root = root;
    this.cb = callbacks;
    this.root.classList.add("hud-root", "is-collapsed");
    this.root.innerHTML = markup();

    this.panel = this.q(".hud-panel");
    this.body = this.q(".hud-body");
    this.collapseBtn = this.q('[data-act="collapse"]');
    this.telemetry = new Telemetry(this.root);
    this.hintEl = this.q(".hud-hint");
    this.helpEl = this.q(".hud-help");
    this.camBtn = this.q('[data-act="camera"]');
    this.camValue = this.q("[data-camera-value]");
    this.pool = new PoolControl(this.root, {
      onParticleCount: (count) => this.cb.onParticleCount(count),
      onOverdrive: (on) => this.cb.onOverdrive(on),
    });

    // The status line lives outside the scrolling body — "why is tracking off"
    // must not be something you have to scroll to — so collapsing hides both.
    this.body.hidden = true;
    this.telemetry.percepEl.hidden = true;
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
    this.telemetry.sample(s, now);

    if (this.hintStartMs === 0) this.hintStartMs = now;

    // Every frame, ahead of the 10 Hz gate: a charge bar that steps at 10 Hz
    // reads as lag in the recognition itself, and each write is diffed inside
    // the component, so an unchanged sequence costs nothing.
    if (s.comboBook) this.combos.setBook(s.comboBook);
    if (s.duetBook) this.combos.setDuetBook(s.duetBook);
    this.combos.update(comboState(s.comboProgress), duetState(s.duetProgress));
    // Every frame too: its hold bar is the feedback that the pose is being read.
    this.tutorial.update(
      s.spells ?? [],
      finite(s.stats?.[STAT.HANDS_PRESENT]),
      now,
      s.hands ?? null,
    );

    if (now - this.lastPaintMs < PAINT_MS) return;
    this.lastPaintMs = now;

    this.telemetry.paintHeader(s);
    this.tickHint(s, now);

    if (this.hiddenAll || this.collapsed) {
      // Still drain the peak-holds: expanding the panel after a minute must not
      // show the spike from whenever it happened to be collapsed. The sparkline
      // history keeps filling, so the moment it opens it has real context.
      this.telemetry.drain();
      return;
    }

    this.telemetry.paintBody(s);
  }

  /** Shows which engine build is running; static for the session. */
  setEngineTier(tier: EngineTier): void {
    this.telemetry.setEngineTier(tier);
  }

  /** Detaches global listeners. Not used by `main.ts`; here for teardown. */
  dispose(): void {
    window.removeEventListener("keydown", this.onKey);
    this.telemetry.dispose();
  }

  // ------------------------------------------------------------------ wiring

  private wireControls(): void {
    this.collapseBtn.addEventListener("click", () =>
      this.setCollapsed(!this.collapsed),
    );
    this.q('[data-act="help"]').addEventListener("click", () =>
      this.setHelp(!this.helpOpen),
    );
    this.q('[data-act="tutorial"]').addEventListener("click", () =>
      this.toggleTutorial(),
    );
    this.q('[data-act="help-close"]').addEventListener("click", () =>
      this.setHelp(false),
    );
    // Clicking the backdrop dismisses; clicking the sheet must not.
    this.helpEl.addEventListener("click", (e) => {
      if (e.target === this.helpEl) this.setHelp(false);
    });

    for (const spec of MODES) {
      const btn = this.q<HTMLButtonElement>(`[data-mode="${spec.mode}"]`);
      btn.addEventListener("click", () => this.setMode(spec.mode));
      this.modeBtns.push(btn);
    }
    this.camBtn.addEventListener("click", () => this.setCamera(!this.cameraOn));
    this.q('[data-act="reset"]').addEventListener("click", () =>
      this.cb.onReset(),
    );

    new ParamControls(this.root, (key, v) => this.cb.onParam(key, v));
  }

  private wireKeys(): void {
    window.addEventListener("keydown", this.onKey);
  }

  private readonly onKey = (e: KeyboardEvent): void => {
    const act = hudAction(e, this.helpOpen);
    if (!act) return;
    switch (act.kind) {
      case "mode":
        // No un-hiding here: the view change is visible in the canvas itself,
        // and someone who pressed H wants the screen clean.
        this.setMode(act.mode);
        break;
      case "camera":
        this.setCamera(!this.cameraOn);
        break;
      case "overdrive":
        this.pool.toggleOverdrive();
        break;
      case "hideAll":
        this.setHiddenAll(!this.hiddenAll);
        break;
      case "reset":
        this.cb.onReset();
        break;
      case "tutorial":
        this.toggleTutorial();
        break;
      case "help":
        this.setHelp(!this.helpOpen);
        break;
      case "closeHelp":
        this.setHelp(false);
        break;
    }
    e.preventDefault();
  };

  // ------------------------------------------------------------------- state

  private setMode(mode: ViewMode): void {
    for (const btn of this.modeBtns) {
      btn.setAttribute(
        "aria-pressed",
        btn.dataset.mode === mode ? "true" : "false",
      );
    }
    this.cb.onViewMode(mode);
  }

  private setCamera(on: boolean): void {
    this.cameraOn = on;
    this.camBtn.setAttribute("aria-pressed", on ? "true" : "false");
    const label = on ? "on" : "off";
    if (this.camValue.textContent !== label) this.camValue.textContent = label;
    this.cb.onToggleCamera(on);
  }

  private setCollapsed(collapsed: boolean): void {
    this.collapsed = collapsed;
    this.root.classList.toggle("is-collapsed", collapsed);
    this.body.hidden = collapsed;
    this.telemetry.percepEl.hidden = collapsed;
    this.collapseBtn.setAttribute(
      "aria-expanded",
      collapsed ? "false" : "true",
    );
    this.collapseBtn.setAttribute(
      "aria-label",
      collapsed ? "Expand the panel" : "Collapse to fps",
    );
    if (!collapsed) this.telemetry.resizeSpark();
  }

  /**
   * `H` hides the panel and the hint but not the help sheet: `?` has to work
   * from the hidden state, otherwise the way back is unguessable.
   */
  private setHiddenAll(hidden: boolean): void {
    this.hiddenAll = hidden;
    this.root.classList.toggle("is-hidden", hidden);
    this.panel.setAttribute("aria-hidden", hidden ? "true" : "false");
  }

  private setHelp(open: boolean): void {
    this.helpOpen = open;
    this.helpEl.classList.toggle("is-open", open);
    this.helpEl.setAttribute("aria-hidden", open ? "false" : "true");
    if (open) this.q<HTMLButtonElement>('[data-act="help-close"]').focus();
  }

  private toggleTutorial(): void {
    this.tutorial.toggle();
    // Both sit at the bottom of the stage; the card supersedes the hint.
    if (this.tutorial.active && !this.hintDone) {
      this.hintDone = true;
      this.hintEl.classList.add("is-gone");
    }
    if (this.tutorial.active && this.hiddenAll) this.setHiddenAll(false);
  }

  private tickHint(s: HudStats, now: number): void {
    if (this.hintDone) return;
    if (this.tutorial.active) {
      this.hintDone = true;
      this.hintEl.classList.add("is-gone");
      return;
    }
    const cast = s.spells?.[0] !== "idle" || s.spells?.[1] !== "idle";
    const seen = finite(s.stats?.[STAT.HANDS_PRESENT]) > 0;
    if (!cast && !seen && now - this.hintStartMs < HINT_MS) return;
    this.hintDone = true;
    this.hintEl.classList.add("is-gone");
  }

  /** The markup is ours and one line old; a miss here is a bug in this file. */
  private q<T extends HTMLElement>(sel: string): T {
    const el = this.root.querySelector(sel);
    if (!el) throw new Error(`hud: missing element ${sel}`);
    return el as T;
  }
}
