/**
 * The finger gun's laser sight.
 *
 * The engine reports an aim ray per hand (`[active, x, y, dx, dy]` in the same
 * normalised, already-mirrored view space as the hand overlay), and this turns
 * it into a dashed line of overlay vertices running from the fingertip to the
 * edge of the field. Dashes rather than one long segment: a solid beam reads as
 * a rendering artefact across the whole frame, while a fading dash train reads
 * as a sight and shows which way the shot will travel.
 *
 * A ray with `dir == [0, 0]` is the gun aimed at the lens — there is no screen
 * bearing to draw, so nothing is emitted for that hand.
 */

import { HANDS } from '../constants';
import { OVERLAY_STRIDE } from './handmesh';

/** Floats per hand in the engine's packed aim buffer. */
const AIM_STRIDE = 5;

/** Dashes per beam, and the on/off split of each dash cell. */
const DASHES = 18;
const DASH_FILL = 0.55;

/** Beam length in normalised units — long enough to always leave the frame. */
const BEAM = 1.6;

/** Two vertices per dash, per hand. */
export const GUNSIGHT_CAPACITY = HANDS * DASHES * 2;

/**
 * Writes the dashed beams into `out` and returns the vertex count. The vertex
 * layout matches the hand overlay (`x, y, glow`) so both share one shader.
 */
export function buildGunSight(aim: Float32Array, out: Float32Array): number {
  const cap = Math.floor(out.length / OVERLAY_STRIDE);
  let n = 0;

  const push = (x: number, y: number, glow: number): void => {
    if (n >= cap) return;
    const o = n * OVERLAY_STRIDE;
    out[o] = x;
    out[o + 1] = y;
    out[o + 2] = glow;
    n++;
  };

  for (let h = 0; h < HANDS; h++) {
    const base = h * AIM_STRIDE;
    if (base + AIM_STRIDE > aim.length) break;
    if (!(aim[base] > 0.5)) continue;

    const x = aim[base + 1];
    const y = aim[base + 2];
    const dx = aim[base + 3];
    const dy = aim[base + 4];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const len = Math.hypot(dx, dy);
    if (!(len > 0.001)) continue;

    const ux = dx / len;
    const uy = dy / len;
    const cell = BEAM / DASHES;
    for (let i = 0; i < DASHES; i++) {
      const t0 = i * cell;
      const t1 = t0 + cell * DASH_FILL;
      // The beam fades along its length, so the fingertip end is clearly the
      // origin and the far end never competes with the simulation.
      const glow = 1 - i / DASHES;
      push(x + ux * t0, y + uy * t0, glow);
      push(x + ux * t1, y + uy * t1, glow);
    }
  }

  return n;
}
