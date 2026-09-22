/**
 * The segmentation mask downscale. Owns nothing but one reusable buffer.
 */

import { MASK_IN_MAX_DIM } from '../../constants';
import type { MaskFrame } from '../../types';

/**
 * Box-downscales a segmentation mask to fit `MASK_IN_MAX_DIM`.
 *
 * `app.ts` silently drops anything larger than the engine's staging buffer, so
 * an oversized mask would mean no body collision at all. Averaging rather than
 * point-sampling matters here: the mask is a coverage field the engine
 * thresholds, and dropping 15 of every 16 pixels makes a thin limb flicker in
 * and out of existence between frames.
 *
 * The output is *not* mirrored — `push_mask` does that itself.
 */
export class MaskScaler {
  private buf = new Float32Array(0);
  private readonly frame: MaskFrame = { data: this.buf, width: 0, height: 0 };

  /** Returns a frame that is valid until the next call, or null if unusable. */
  fit(data: Float32Array, width: number, height: number): MaskFrame | null {
    const w = Math.floor(width);
    const h = Math.floor(height);
    if (!(w > 0) || !(h > 0) || data.length < w * h) return null;

    if (w <= MASK_IN_MAX_DIM && h <= MASK_IN_MAX_DIM) {
      // Pass the model's own buffer through; the caller copies it into WASM
      // before the next inference can touch it, and `BodyMask::stage` already
      // clamps non-finite confidences.
      this.frame.data = data;
      this.frame.width = w;
      this.frame.height = h;
      return this.frame;
    }

    const scale = MASK_IN_MAX_DIM / Math.max(w, h);
    const outW = Math.max(1, Math.min(MASK_IN_MAX_DIM, Math.floor(w * scale)));
    const outH = Math.max(1, Math.min(MASK_IN_MAX_DIM, Math.floor(h * scale)));
    if (this.buf.length < outW * outH) this.buf = new Float32Array(outW * outH);
    const out = this.buf;

    for (let oy = 0; oy < outH; oy++) {
      const y0 = Math.floor((oy * h) / outH);
      const y1 = Math.max(y0 + 1, Math.floor(((oy + 1) * h) / outH));
      for (let ox = 0; ox < outW; ox++) {
        const x0 = Math.floor((ox * w) / outW);
        const x1 = Math.max(x0 + 1, Math.floor(((ox + 1) * w) / outW));
        let sum = 0;
        for (let y = y0; y < y1; y++) {
          const row = y * w;
          for (let x = x0; x < x1; x++) {
            // Written as nested comparisons rather than a clamp so that a NaN
            // confidence collapses to 0 instead of poisoning the average.
            const v = data[row + x];
            sum += v > 0 ? (v < 1 ? v : 1) : 0;
          }
        }
        out[oy * outW + ox] = sum / ((y1 - y0) * (x1 - x0));
      }
    }

    this.frame.data = out;
    this.frame.width = outW;
    this.frame.height = outH;
    return this.frame;
  }
}

