/**
 * Stirring the fluid with a finger or the mouse when there is no camera.
 *
 * Without a camera the engine only runs its ambient drive, and a phone visitor
 * who declined the permission is left watching a screensaver. This paints a
 * soft bright blob under the pointer into the optical-flow luma plane and
 * pushes it the way a camera frame would be, so the Rust optical flow reads the
 * blob's motion and the fluid follows the finger. No engine change: to the
 * engine it is a camera that only ever sees one glowing dot.
 *
 * The engine applies the last computed flow on every step, not only on the
 * step after a push, so after the pointer stops a few identical planes are
 * pushed to bring the flow back to zero. `push_luma` does not count as
 * perception, so the ambient drive keeps running underneath.
 */

import { FLOW_H, FLOW_W } from './constants';

/** Push cadence: the camera path's ~30 Hz. */
const PUSH_MS = 1000 / 30;
/** Identical planes pushed after the last move: the first zeroes the flow. */
const SETTLE_PUSHES = 3;
/** Blob radius (Gaussian sigma) as a fraction of the plane height. */
const SIGMA = 0.07;
/** Pointer targets that are controls, not the stage. */
const NOT_THE_STAGE = '#boot, .hud-panel, .hud-help, .tut, .hud-combos, button, input, a';

/**
 * Paints a Gaussian blob centred at normalised `(u, v)` on a black plane.
 * `u` and `v` are screen fractions: the plane is in the same mirrored-view
 * space as a camera plane, which already matches the screen.
 */
export function paintStir(plane: Uint8Array, w: number, h: number, u: number, v: number): void {
  const cx = u * (w - 1);
  const cy = v * (h - 1);
  const sigma = SIGMA * h;
  const inv = 1 / (2 * sigma * sigma);
  // Beyond 3 sigma the blob is below one luma step; skip the exp there.
  const reach = Math.ceil(3 * sigma);
  plane.fill(0);
  const x0 = Math.max(0, Math.floor(cx - reach));
  const x1 = Math.min(w - 1, Math.ceil(cx + reach));
  const y0 = Math.max(0, Math.floor(cy - reach));
  const y1 = Math.min(h - 1, Math.ceil(cy + reach));
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      plane[y * w + x] = Math.round(255 * Math.exp(-(dx * dx + dy * dy) * inv));
    }
  }
}

export class PointerStir {
  private enabled = false;
  private u = 0.5;
  private v = 0.5;
  /** A move since the last push. */
  private moved = false;
  /** Still planes still owed after the pointer stopped. */
  private settle = 0;
  /** Nothing is pushed until the pointer has been seen once. */
  private seen = false;
  private lastPushMs = 0;

  /** Starts listening. The stage covers the viewport, so window coordinates are stage coordinates. */
  enable(): void {
    if (this.enabled) return;
    this.enabled = true;
    window.addEventListener('pointermove', this.onPointer, { passive: true });
    window.addEventListener('pointerdown', this.onPointer, { passive: true });
  }

  dispose(): void {
    this.enabled = false;
    window.removeEventListener('pointermove', this.onPointer);
    window.removeEventListener('pointerdown', this.onPointer);
  }

  private readonly onPointer = (event: PointerEvent): void => {
    if (event.target instanceof Element && event.target.closest(NOT_THE_STAGE)) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (!w || !h) return;
    this.u = Math.min(1, Math.max(0, event.clientX / w));
    this.v = Math.min(1, Math.max(0, event.clientY / h));
    this.moved = true;
    this.seen = true;
  };

  /**
   * Writes the next plane into `luma` when one is due and returns the time
   * since the previous one in seconds, for `push_luma`; 0 when nothing is due.
   */
  pump(nowMs: number, luma: Uint8Array): number {
    if (!this.enabled || !this.seen) return 0;
    if (!this.moved && this.settle === 0) return 0;
    if (this.lastPushMs !== 0 && nowMs - this.lastPushMs < PUSH_MS) return 0;

    if (this.moved) {
      this.moved = false;
      this.settle = SETTLE_PUSHES;
    } else {
      this.settle--;
    }
    paintStir(luma, FLOW_W, FLOW_H, this.u, this.v);
    // A long pause since the last push is not motion: report one camera frame.
    const dt = this.lastPushMs === 0 || nowMs - this.lastPushMs > 250 ? 1 / 30 : (nowMs - this.lastPushMs) / 1000;
    this.lastPushMs = nowMs;
    return dt;
  }
}
