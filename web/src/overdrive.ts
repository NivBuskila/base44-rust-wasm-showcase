/**
 * Overdrive — the "look what Rust can do" mode.
 *
 * Flips the engine to its full one-million-particle pool and raises the spawn
 * rate so the pool actually fills, then shows a live throughput strip at the
 * bottom of the stage: particles alive, particle-steps per second, and the
 * engine step time. Everything here is presentation; the numbers come from the
 * engine's own stats and the loop's own timers.
 */

import { MAX_PARTICLES, STAT } from './constants';

/** Particles the pool is raised to when overdrive is on. */
export const OVERDRIVE_PARTICLES = MAX_PARTICLES;
/**
 * Spawn rate that keeps a 1M pool full at the default 4.5 s lifetime
 * (1M / 4.5 s ≈ 222k/s). Clamps to 400k in Rust, so this is well inside.
 */
export const OVERDRIVE_SPAWN_RATE = 230_000;
/** `Params::default().spawn_rate`; restored when overdrive is switched off. */
export const DEFAULT_SPAWN_RATE = 30_000;

/** How often the banner repaints, so the digits are readable rather than a blur. */
const PAINT_MS = 120;

/** `1234567` -> `1.23M`, `84200` -> `84.2k`. */
function big(v: number): string {
  if (!Number.isFinite(v) || v < 0) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}G`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}

export class OverdriveBanner {
  private readonly el: HTMLElement;
  private readonly alive: HTMLElement;
  private readonly rate: HTMLElement;
  private readonly step: HTMLElement;
  private lastPaint = 0;
  private on = false;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('aside');
    this.el.className = 'od-banner';
    this.el.hidden = true;
    this.el.setAttribute('aria-live', 'off');
    this.el.innerHTML = `
      <span class="od-tag">overdrive</span>
      <span class="od-cell"><b data-od-alive>—</b><i>particles alive</i></span>
      <span class="od-cell"><b data-od-rate>—</b><i>particle-steps / s</i></span>
      <span class="od-cell"><b data-od-step>—</b><i>engine step</i></span>
      <span class="od-src">rust · wasm · single thread</span>`;
    parent.appendChild(this.el);
    this.alive = this.el.querySelector('[data-od-alive]')!;
    this.rate = this.el.querySelector('[data-od-rate]')!;
    this.step = this.el.querySelector('[data-od-step]')!;
  }

  get active(): boolean {
    return this.on;
  }

  setActive(on: boolean): void {
    this.on = on;
    this.el.hidden = !on;
    // Restart the entrance animation each time it is switched on.
    if (on) {
      this.el.classList.remove('is-in');
      void this.el.offsetWidth;
      this.el.classList.add('is-in');
    }
  }

  /** Called once per frame; cheap when off, throttled when on. */
  update(stats: Float32Array, fps: number, stepMs: number): void {
    if (!this.on) return;
    const now = performance.now();
    if (now - this.lastPaint < PAINT_MS) return;
    this.lastPaint = now;

    const alive = stats[STAT.PARTICLES_ALIVE] ?? 0;
    this.alive.textContent = big(alive);
    this.rate.textContent = big(alive * Math.max(0, fps));
    this.step.textContent = `${stepMs.toFixed(1)} ms`;
  }
}
