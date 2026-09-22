/**
 * The panel's readout half, split out of `hud.ts`: the fps digits, the frame
 * sparkline, the three budget meters, the telemetry cells, the warp bar, the
 * spell/gesture cards and the perception status line.
 *
 * Why it is one island rather than a set of loose fields on the HUD: every one
 * of these is written from the same 10 Hz paint and from nowhere else, and each
 * write is diffed before it lands (`setText`/`setData`) because an
 * unconditional write is a style invalidation, and at 10 Hz across ~20 nodes
 * that is real work for nothing.
 *
 * Two rules the HUD depends on:
 * - `sample` runs every frame, `paintHeader`/`paintBody` at 10 Hz — the meters
 *   and the frame timer are peak-hold, so the worst frame in the window is what
 *   gets shown, never the sample that happened to land on the tick;
 * - while the panel is collapsed or hidden the body is not painted but the
 *   peaks are still `drain`ed, so expanding it never shows a spike from
 *   whenever it happened to be closed.
 */

import { STAT } from "./constants";
import type { EngineTier } from "./engine-loader";
import { handsPresent, statCells, warpView } from "./hud-cells";
import { FrameTimer } from "./hud-frame-timer";
import { Meter, finite, frameTone } from "./hud-meter";
import { Sparkline } from "./hud-spark";
import { CELLS, METERS } from "./hud-spec";
import { SpellReadout } from "./hud-spells";
import { statusLine } from "./hud-status";
import type { HudStats, PerceptionStatus } from "./types";

export class Telemetry {
  private readonly fpsEl: HTMLElement;
  private readonly ambientEl: HTMLElement;
  private readonly engineBadge: HTMLElement;
  private readonly meters: Meter[] = [];
  private readonly spark: Sparkline;
  private readonly cells = new Map<string, HTMLElement>();
  private readonly warpValue: HTMLElement;
  private readonly warpFill: HTMLElement;
  private readonly spells: SpellReadout;
  readonly percepEl: HTMLElement;
  private readonly percepHead: HTMLElement;
  private readonly percepWhy: HTMLElement;

  private readonly frameTimer = new FrameTimer();
  private lastFpsTone = "";
  private lastAmbient: boolean | null = null;
  private lastPercep = "";

  constructor(private readonly root: HTMLElement) {
    this.fpsEl = this.q("[data-fps]");
    this.ambientEl = this.q("[data-ambient]");
    this.engineBadge = this.q("[data-engine]");
    this.spark = new Sparkline(this.q("canvas.hud-spark"));
    this.warpValue = this.q("[data-warp-value]");
    this.warpFill = this.q("[data-warp-fill]");
    this.percepEl = this.q(".hud-percep");
    this.percepHead = this.q("[data-percep-head]");
    this.percepWhy = this.q("[data-percep-why]");

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
    for (const c of CELLS)
      this.cells.set(c.id, this.q(`[data-cell="${c.id}"] b`));
    this.spells = new SpellReadout(root);
  }

  /** Every frame: accumulate the peaks the paint reads. */
  sample(s: HudStats, now: number): void {
    this.frameTimer.sample(now);
    this.meters[0]?.sample(s.stepMs);
    this.meters[1]?.sample(s.renderMs);
    this.meters[2]?.sample(s.inferenceMs);
  }

  /**
   * The two readouts that stay visible while the panel is collapsed: the worst
   * frame in the window as fps (plus its tone and the sparkline sample) and the
   * ambient badge. See `hud-frame-timer.ts` for why an unmeasurable window
   * reports nothing at all.
   */
  paintHeader(s: HudStats): void {
    const frameMs = this.frameTimer.read(finite(s.fps));
    if (frameMs > 0.001) {
      this.spark.push(frameMs);
      const worstFps = 1000 / frameMs;
      this.setText(
        this.fpsEl,
        worstFps >= 10 ? worstFps.toFixed(0) : worstFps.toFixed(1),
      );
      const fpsTone = frameTone(frameMs);
      if (fpsTone !== this.lastFpsTone) {
        this.lastFpsTone = fpsTone;
        this.fpsEl.dataset.tone = fpsTone;
      }
    }

    const ambient = finite(s.stats?.[STAT.AMBIENT]) > 0.5;
    if (ambient !== this.lastAmbient) {
      this.lastAmbient = ambient;
      this.ambientEl.hidden = !ambient;
    }
  }

  /** Drops the accumulated peaks without painting; used while collapsed. */
  drain(): void {
    for (const m of this.meters) m.drain();
  }

  /** Everything inside the panel body, painted only while it is open. */
  paintBody(s: HudStats): void {
    for (const m of this.meters) m.paint();
    this.spark.draw();

    const st = s.stats;
    for (const [id, text] of Object.entries(statCells(st))) this.cell(id, text);

    const warp = warpView(st);
    this.setText(this.warpValue, warp.label);
    if (this.warpFill.style.getPropertyValue("--v") !== warp.pct) {
      this.warpFill.style.setProperty("--v", warp.pct);
    }
    this.setData(this.warpFill, "tone", warp.warping ? "on" : "off");

    this.spells.paint(handsPresent(st), s.spells, warp.warping);
    this.paintPerception(s.perception);
  }

  /**
   * Shows which engine build is running. Static for the session, so it is set
   * once rather than re-derived on every paint.
   */
  setEngineTier(tier: EngineTier): void {
    const label =
      tier.name === "threads" ? `${tier.threads} thr · simd` : "1 thr · simd";
    this.cell("engine", label);
    this.engineBadge.hidden = tier.name !== "threads";
    this.setText(this.engineBadge, `${tier.threads} threads`);
    this.engineBadge.title = `Rust engine on ${tier.threads} worker threads (${tier.reason})`;
  }

  /** The sparkline canvas has no size while the body is hidden. */
  resizeSpark(): void {
    this.spark.resize();
  }

  dispose(): void {
    this.spark.dispose();
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

  private cell(id: string, text: string): void {
    const el = this.cells.get(id);
    if (el) this.setText(el, text);
  }

  private setText(el: HTMLElement, text: string): void {
    if (el.textContent !== text) el.textContent = text;
  }

  private setData(el: HTMLElement, key: string, value: string): void {
    if (el.dataset[key] !== value) el.dataset[key] = value;
  }

  private q<T extends HTMLElement>(sel: string): T {
    const el = this.root.querySelector(sel);
    if (!el) throw new Error(`hud: missing element ${sel}`);
    return el as T;
  }
}
