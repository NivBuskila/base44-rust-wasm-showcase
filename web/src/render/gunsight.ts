/**
 * The finger gun's aim sight — a stream of energy, not a ruler.
 *
 * The engine reports an aim ray per hand (`[active, x, y, dx, dy]` in the same
 * normalised, already-mirrored view space as the hand overlay). A straight dash
 * train read as a technical drawing laid over the simulation, so the sight is
 * built instead as *flowing charge*: two strands braiding around the aim axis,
 * widening as they travel, with brightness pulses scrolling outwards from the
 * fingertip, over a short hot spine at the muzzle where the charge is dense.
 * Nothing about it is static — the motion is what makes it read as energy
 * rather than as a line — so it is driven by the frame clock.
 *
 * A ray with `dir == [0, 0]` is the gun aimed at the lens. There is no bearing
 * to draw along, so the same charge is staged head-on instead: arcs spiralling
 * inwards onto the fingertip, which is what a stream coming at the viewer looks
 * like in a flat frame.
 */

import { HANDS } from '../constants';
import { OVERLAY_STRIDE } from './handmesh';

/** Floats per hand in the engine's packed aim buffer. */
const AIM_STRIDE = 5;

/** Beam length in normalised units — long enough to always leave the frame. */
const BEAM = 1.6;

/** Segments per strand, and the strands braiding around the axis. */
const NODES = 26;
const STRANDS = 2;

/** Braid geometry: waves over the beam's length and how wide it opens up. */
const WAVES = 2.6;
const SPREAD_NEAR = 0.006;
const SPREAD_FAR = 0.04;

/** Cycles per second the braid and the brightness pulses travel outwards. */
const FLOW_HZ = 1.7;
const PULSE_HZ = 2.4;
/** Brightness pulses along the beam at any instant. */
const PULSES = 3;

/** The dense core at the muzzle: short segments right where the charge leaves. */
const SPINE = 6;
const SPINE_LEN = 0.11;

/** Head-on staging: spiral arms converging on the fingertip, and their reach. */
const ARMS = 3;
const ARM_NODES = 12;
const ARM_R = 0.085;
const ARM_TURNS = 0.55;
/** Revolutions per second the head-on spiral turns. */
const SPIN_HZ = 0.8;

const BEAM_VERTICES = (STRANDS * NODES + SPINE) * 2;
const HEAD_ON_VERTICES = (ARMS * ARM_NODES + 8) * 2;

/** Two vertices per segment, per hand; whichever staging needs more. */
export const GUNSIGHT_CAPACITY = HANDS * Math.max(BEAM_VERTICES, HEAD_ON_VERTICES);

type Push = (x: number, y: number, glow: number) => void;

/**
 * Writes the sight into `out` and returns the vertex count. The vertex layout
 * matches the hand overlay (`x, y, glow`) so both share one shader. `time` is
 * the frame clock in seconds; it animates the flow.
 */
export function buildGunSight(aim: Float32Array, out: Float32Array, time: number): number {
  const cap = Math.floor(out.length / OVERLAY_STRIDE);
  const t = Number.isFinite(time) ? time : 0;
  let n = 0;

  const push: Push = (x, y, glow) => {
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
    if (len > 0.001) {
      pushStream(push, x, y, dx / len, dy / len, t);
    } else {
      pushHeadOn(push, x, y, t);
    }
  }

  return n;
}

/** Brightness of the charge at `s` (0..1 along the beam) at time `t`. */
function flow(s: number, t: number): number {
  // A travelling sawtooth sharpened into blobs: the pulses run outwards, and
  // the whole beam fades with distance so the fingertip stays the origin.
  const phase = s * PULSES - t * PULSE_HZ;
  const pulse = 0.35 + 0.65 * Math.pow(0.5 + 0.5 * Math.cos(phase * Math.PI * 2), 3);
  return pulse * (1 - s * 0.85);
}

/** The braided stream travelling away from the fingertip. */
function pushStream(push: Push, x: number, y: number, ux: number, uy: number, t: number): void {
  // Normal to the aim axis: the braid swings across it.
  const nx = -uy;
  const ny = ux;

  for (let strand = 0; strand < STRANDS; strand++) {
    const twist = (strand / STRANDS) * Math.PI * 2;
    let px = x;
    let py = y;
    for (let i = 1; i <= NODES; i++) {
      const s = i / NODES;
      const spread = SPREAD_NEAR + (SPREAD_FAR - SPREAD_NEAR) * s;
      const swing = Math.sin(s * WAVES * Math.PI * 2 - t * FLOW_HZ * Math.PI * 2 + twist) * spread;
      const along = s * BEAM;
      const qx = x + ux * along + nx * swing;
      const qy = y + uy * along + ny * swing;
      push(px, py, flow(s - 1 / NODES, t));
      push(qx, qy, flow(s, t));
      px = qx;
      py = qy;
    }
  }

  // The muzzle core: the charge is densest where it leaves the fingertip, so a
  // few full-brightness segments sit straight on the axis there.
  for (let i = 0; i < SPINE; i++) {
    const a = (i / SPINE) * SPINE_LEN;
    const b = ((i + 0.6) / SPINE) * SPINE_LEN;
    const glow = 1 - (i / SPINE) * 0.5;
    push(x + ux * a, y + uy * a, glow);
    push(x + ux * b, y + uy * b, glow);
  }
}

/** The same stream aimed at the viewer: spiral arms falling into the fingertip. */
function pushHeadOn(push: Push, x: number, y: number, t: number): void {
  const spin = t * SPIN_HZ * Math.PI * 2;
  for (let arm = 0; arm < ARMS; arm++) {
    const off = (arm / ARMS) * Math.PI * 2 + spin;
    let px: number | null = null;
    let py = 0;
    let pg = 0;
    for (let i = 0; i <= ARM_NODES; i++) {
      const s = i / ARM_NODES;
      // Radius shrinks inwards while the angle winds, so each arm is a spiral
      // being swallowed by the fingertip; the pulse rides it like the beam's.
      const r = ARM_R * (0.15 + 0.85 * s);
      const a = off + s * ARM_TURNS * Math.PI * 2;
      const qx = x + Math.cos(a) * r;
      const qy = y + Math.sin(a) * r;
      const glow = flow(1 - s, t);
      if (px !== null) {
        push(px, py, pg);
        push(qx, qy, glow);
      }
      px = qx;
      py = qy;
      pg = glow;
    }
  }

  // A tight pulsing bead at the tip: the point the charge is arriving from.
  const bead = 0.012 + 0.006 * Math.sin(t * PULSE_HZ * Math.PI * 2);
  for (let i = 0; i < 4; i++) {
    const a0 = (i / 4) * Math.PI * 2;
    const a1 = ((i + 1) / 4) * Math.PI * 2;
    push(x + Math.cos(a0) * bead, y + Math.sin(a0) * bead, 1);
    push(x + Math.cos(a1) * bead, y + Math.sin(a1) * bead, 1);
  }
}
