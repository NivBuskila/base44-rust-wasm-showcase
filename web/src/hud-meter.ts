/**
 * Timing bars: budget tones and the peak-hold meter row.
 *
 * `sample` is called every frame; `paint` drains the peak and is called at
 * 10 Hz, so a 40 ms hitch between paints is shown rather than hidden.
 */

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

export type Tone = 'ok' | 'warn' | 'over';

/** Any non-finite engine value reads as zero rather than poisoning the DOM. */
export function finite(v: number | undefined): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function toneOf(ratio: number): Tone {
  if (ratio >= 1) return 'over';
  if (ratio >= TONE_WARN) return 'warn';
  return 'ok';
}

/** Tone for a whole-frame time in ms. See {@link FRAME_WARN_MS}. */
export function frameTone(ms: number): Tone {
  if (ms >= FRAME_OVER_MS) return 'over';
  if (ms >= FRAME_WARN_MS) return 'warn';
  return 'ok';
}

/**
 * A label / track / number triple with peak-hold. `sample` is called every
 * frame; `paint` drains the peak and is called at 10 Hz.
 */
export class Meter {
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
