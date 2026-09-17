/**
 * On-screen diagnostics and controls.
 *
 * MINIMAL BASELINE — shows the frame budget and the key engine counters so the
 * app is inspectable from the first commit. The full HUD (parameter sliders,
 * view-mode switcher, gesture legend, perception status) replaces the markup
 * and `update` body while keeping this surface intact.
 */

import { STAT } from './constants';
import type { HudCallbacks, HudStats } from './types';

export class Hud {
  private readonly root: HTMLElement;
  private readonly readout: HTMLElement;
  private lastPaintMs = 0;

  constructor(
    root: HTMLElement,
    private readonly callbacks: HudCallbacks,
  ) {
    this.root = root;
    this.root.innerHTML = `
      <div class="hud-panel">
        <div class="hud-title">AETHER</div>
        <pre class="hud-readout" id="hud-readout"></pre>
      </div>`;
    this.readout = this.root.querySelector('#hud-readout')!;
    // Referenced so the callback surface stays live for the full HUD.
    void this.callbacks;
  }

  update(s: HudStats): void {
    // The DOM is ~40x slower to touch than the whole simulation step, so the
    // readout repaints at 10 Hz rather than on every frame.
    const now = performance.now();
    if (now - this.lastPaintMs < 100) return;
    this.lastPaintMs = now;

    const st = s.stats;
    const perception =
      s.perception.kind === 'ready'
        ? `ready (${s.perception.delegate})`
        : s.perception.kind === 'loading'
          ? 'loading'
          : `off — ${s.perception.reason}`;

    this.readout.textContent = [
      `fps        ${s.fps.toFixed(0).padStart(5)}`,
      `step       ${s.stepMs.toFixed(2).padStart(5)} ms`,
      `render     ${s.renderMs.toFixed(2).padStart(5)} ms`,
      `inference  ${s.inferenceMs.toFixed(1).padStart(5)} ms`,
      ``,
      `particles  ${(st[STAT.PARTICLES_ALIVE] ?? 0).toFixed(0)}`,
      `motion     ${(st[STAT.MOTION_ENERGY] ?? 0).toFixed(3)}`,
      `energy     ${(st[STAT.FLUID_ENERGY] ?? 0).toFixed(3)}`,
      `divergence ${(st[STAT.FLUID_DIVERGENCE] ?? 0).toFixed(4)}`,
      `hands      ${(st[STAT.HANDS_PRESENT] ?? 0).toFixed(0)}`,
      `body       ${(st[STAT.MASK_PRESENT] ?? 0) > 0.5 ? `${((st[STAT.MASK_COVERAGE] ?? 0) * 100).toFixed(0)}%` : 'no'}`,
      `spells     ${s.spells[0]} / ${s.spells[1]}`,
      `mode       ${(st[STAT.AMBIENT] ?? 0) > 0.5 ? 'ambient' : 'live'}`,
      `perception ${perception}`,
    ].join('\n');
  }
}
