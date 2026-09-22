/**
 * The pool island: the particle-count slider, its label and the overdrive
 * preset.
 *
 * Those three are one control, not three: the slider's committed value *is* the
 * pool size, overdrive is a preset that parks the slider at the 1M detent and
 * restores it on the way out, and a manual drag has to leave overdrive because
 * it is a preset, not a lock. Keeping that little state machine — and the
 * remembered detent behind it — beside the elements it writes means `hud.ts`
 * only says `toggleOverdrive()`, which is all the keyboard handler knows.
 *
 * `hud-params.ts` is the same shape for the parameter sliders; this is the pool.
 */

import { MAX_PARTICLES } from "./constants";
import {
  PARTICLE_DETENTS,
  countToDetent,
  detentToCount,
  thousands,
} from "./hud-spec";

/** Pool size the slider starts at, matching the engine's own default. */
const DEFAULT_COUNT = 120_000;

export interface PoolCallbacks {
  onParticleCount: (count: number) => void;
  onOverdrive: (on: boolean) => void;
}

export class PoolControl {
  private readonly input: HTMLInputElement;
  private readonly value: HTMLElement;
  private readonly overdriveBtn: HTMLButtonElement;
  private readonly cb: PoolCallbacks;

  private overdriveOn = false;
  /** Slider position to restore when overdrive is switched off. */
  private beforeOverdrive = 0;

  constructor(root: HTMLElement, cb: PoolCallbacks) {
    this.cb = cb;
    const q = <T extends HTMLElement>(sel: string): T => {
      const el = root.querySelector(sel);
      if (!el) throw new Error(`hud: missing element ${sel}`);
      return el as T;
    };
    this.input = q("[data-particles]");
    this.value = q("[data-particles-value]");
    this.overdriveBtn = q('[data-act="overdrive"]');

    this.input.value = String(countToDetent(DEFAULT_COUNT));
    this.label(detentToCount(countToDetent(DEFAULT_COUNT)));

    // Live label on drag, commit on release: resizing the pool reallocates and
    // re-seeds up to 220k particles, so firing it per input event would stutter
    // the very frame rate the panel is reporting.
    this.input.addEventListener("input", () => {
      this.label(detentToCount(this.input.valueAsNumber));
    });
    this.input.addEventListener("change", () => {
      if (this.overdriveOn) this.setOverdrive(false, false);
      this.cb.onParticleCount(detentToCount(this.input.valueAsNumber));
    });
    this.overdriveBtn.addEventListener("click", () => this.toggleOverdrive());
  }

  get overdrive(): boolean {
    return this.overdriveOn;
  }

  toggleOverdrive(): void {
    this.setOverdrive(!this.overdriveOn);
  }

  /**
   * `restoreSlider` is false when the user is the one moving the slider, so
   * their new position is not overwritten on the way out of overdrive.
   */
  private setOverdrive(on: boolean, restoreSlider = true): void {
    if (on === this.overdriveOn) return;
    this.overdriveOn = on;
    this.overdriveBtn.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) {
      this.beforeOverdrive = this.input.valueAsNumber;
      this.input.value = String(PARTICLE_DETENTS);
      this.label(MAX_PARTICLES);
    } else if (restoreSlider) {
      this.input.value = String(this.beforeOverdrive);
      this.label(detentToCount(this.beforeOverdrive));
    }
    this.cb.onOverdrive(on);
  }

  private label(count: number): void {
    const text = thousands(count);
    if (this.value.textContent !== text) this.value.textContent = text;
  }
}
