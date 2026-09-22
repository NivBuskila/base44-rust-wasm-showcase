/**
 * The parameter sliders and the preset row — the one self-contained island of
 * the panel: the inputs, their value labels, and the preset buttons that write
 * all of them at once. Nothing else in the HUD reads or writes these controls,
 * so they own their own elements instead of more fields on `Hud`.
 *
 * Every value is clamped to the slider's own range before it reaches the
 * engine, which is what keeps a garbage `valueAsNumber` (an empty input reads
 * `NaN`) from being passed through.
 */

import { PRESETS, type ParamPreset } from "./hud-presets";
import { PARAMS } from "./hud-spec";

type OnParam = (key: string, value: number) => void;

export class ParamControls {
  private readonly presetBtns = new Map<string, HTMLButtonElement>();

  constructor(
    private readonly root: HTMLElement,
    private readonly onParam: OnParam,
  ) {
    for (const preset of PRESETS) {
      const btn = this.q<HTMLButtonElement>(`[data-preset="${preset.id}"]`);
      btn.addEventListener("click", () => this.apply(preset));
      this.presetBtns.set(preset.id, btn);
    }

    for (const p of PARAMS) {
      const input = this.q<HTMLInputElement>(`[data-param="${p.key}"]`);
      const value = this.q(`[data-param-value="${p.key}"]`);
      input.addEventListener("input", () => {
        const raw = input.valueAsNumber;
        const v = Number.isFinite(raw)
          ? Math.min(p.max, Math.max(p.min, raw))
          : p.value;
        this.setText(value, p.fmt(v));
        this.onParam(p.key, v);
        // Hand-tuning past a preset means the panel is no longer showing it.
        this.mark(null);
      });
    }
  }

  /**
   * Writes every parameter of a preset, including the ones it does not name —
   * those fall back to the slider's own default, so the result is the same
   * whatever was set before, rather than a mix of two presets.
   */
  apply(preset: ParamPreset): void {
    for (const p of PARAMS) {
      const v = preset.values[p.key] ?? p.value;
      const input = this.q<HTMLInputElement>(`[data-param="${p.key}"]`);
      input.valueAsNumber = v;
      this.setText(this.q(`[data-param-value="${p.key}"]`), p.fmt(v));
      this.onParam(p.key, v);
    }
    this.mark(preset.id);
  }

  /** Lights the active preset button, or none of them after a manual edit. */
  mark(id: string | null): void {
    for (const [key, btn] of this.presetBtns) {
      btn.setAttribute("aria-pressed", String(key === id));
    }
  }

  private setText(el: HTMLElement, text: string): void {
    if (el.textContent !== text) el.textContent = text;
  }

  private q<T extends HTMLElement>(sel: string): T {
    const el = this.root.querySelector(sel);
    if (!el) throw new Error(`hud: missing element ${sel}`);
    return el as T;
  }
}
