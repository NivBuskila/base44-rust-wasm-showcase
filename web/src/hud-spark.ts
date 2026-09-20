/**
 * The frame-time sparkline: a fill-then-shift window of the worst frame per
 * paint, drawn right-aligned so a partly-filled window reads as history
 * scrolling in rather than a ramp.
 *
 * The backing store is kept in step with the CSS box, which is what stops it
 * from looking like a stretched JPEG on a phone or a high-DPI display.
 */

import { frameTone } from './hud-meter';
import { FRAME_BUDGET_MS } from './hud-spec';

/** Columns in the frame-time sparkline — at 10 Hz this is ~9 s of history. */
const SPARK_SAMPLES = 92;

export class Sparkline {
  private readonly ctx: CanvasRenderingContext2D | null;
  /** Frame-time history, newest last, as a fill-then-shift window. */
  private readonly history = new Float32Array(SPARK_SAMPLES);
  private historyLen = 0;
  private readonly resizeObs: ResizeObserver | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d');
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObs = new ResizeObserver(() => this.resize());
      this.resizeObs.observe(canvas);
    }
    this.resize();
  }

  dispose(): void {
    this.resizeObs?.disconnect();
  }

  push(ms: number): void {
    if (this.historyLen < SPARK_SAMPLES) {
      this.history[this.historyLen++] = ms;
    } else {
      this.history.copyWithin(0, 1);
      this.history[SPARK_SAMPLES - 1] = ms;
    }
  }

  resize(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this.draw();
  }

  draw(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
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
}
