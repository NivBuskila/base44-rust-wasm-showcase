/**
 * The arithmetic behind the particle compute pass, with no GPU in sight.
 *
 * Every frame the WebGPU backend has to fold the engine's parameters exactly the
 * way `aether_core::particles` folds them through `dt` — the decays, the inverse
 * step, the swirl and gravity pre-multiply, the integer respawn budget carried
 * between frames — and pack the result into one uniform block whose slot order
 * the WGSL depends on positionally. That is the part most likely to drift from
 * the Rust path, and it used to live inline in the middle of a method that also
 * uploaded buffers and dispatched a pass, where nothing could test it.
 *
 * It is pure here: numbers in, the same numbers written into two views over one
 * block, and the carried respawn credit returned. `sim-uniform.test.ts` pins the
 * slot order and the fold against the Rust constants, so a shader binding change
 * or a stray unit fails in CI instead of on someone's screen.
 */

import type { GpuSimFrame } from "../types";
import { SIM_UNIFORM_FLOATS } from "./shaders/particles";

/** Mirrors `aether_core::particles`: heat decay and rise rates. */
export const HEAT_DECAY = 0.85;
export const HEAT_RISE = 3.5;
/** `aether_core::particles::MAX_STEP`. */
export const MAX_SIM_STEP = 0.25;

export const clamp = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));
/** `aether_core::math::decay`. */
export const decay = (rate: number, dt: number): number => Math.exp(-rate * dt);
export const finite = (v: number, fallback: number): number =>
  Number.isFinite(v) ? v : fallback;

/** What the pass knows that the engine frame does not carry itself. */
export interface SimUniformInput {
  /** Grid and op-log geometry, straight off the engine's frame. */
  sim: GpuSimFrame;
  /** Particles actually stepped this frame, already clamped to the pool. */
  active: number;
  /** Ops to replay, already clamped to `MAX_PARTICLE_OPS`. */
  opCount: number;
  /** Obstacle field size, and whether it is worth sampling at all. */
  obstacleW: number;
  obstacleH: number;
  hasObstacle: boolean;
  /** Frame counter, the salt of the shader's PCG hash. */
  frameIndex: number;
  /** Fractional respawns carried over from the previous frame. */
  respawnCredit: number;
}

/** The two values the caller must keep for the next frame. */
export interface SimUniformOutput {
  /** Fractional respawn credit to carry; the integer part went to the shader. */
  respawnCredit: number;
  /** Timestep after clamping — zero means the pass may be skipped. */
  dt: number;
}

/**
 * Writes one `SimUniform` block. `f` and `u` must be a `Float32Array` and a
 * `Uint32Array` over the *same* buffer of `SIM_UNIFORM_FLOATS` floats: the
 * block mixes floats and counts, and the WGSL struct reads each slot by index.
 */
export function packSimUniform(
  f: Float32Array,
  u: Uint32Array,
  input: SimUniformInput,
): SimUniformOutput {
  if (f.length < SIM_UNIFORM_FLOATS || u.length < SIM_UNIFORM_FLOATS) {
    throw new Error("sim uniform scratch is too small");
  }
  const { sim, active, opCount, obstacleW, obstacleH, hasObstacle } = input;

  // Everything the Rust `Frame` folds through dt, folded the same way here.
  const dt = clamp(finite(sim.params[0], 0), 0, MAX_SIM_STEP);
  const maxx = sim.gridW - 1;
  const maxy = sim.gridH - 1;
  const life = clamp(finite(sim.params[1], 1), 0.05, 60);
  const drag = clamp(finite(sim.params[2], 1), 0, 8);
  const spawnRate = Math.max(0, finite(sim.params[3], 0));
  const damping = Math.max(0, finite(sim.params[4], 0));
  const swirl = finite(sim.params[5], 0) * dt;
  const gravity = finite(sim.params[6], 0) * dt;

  // Respawns are whole particles, so the fractional part is carried rather than
  // rounded away — otherwise a low spawn rate never respawns anything at all.
  // The carry is capped at one frame's worth so a paused tab cannot bank a
  // burst that lands the moment it resumes.
  const perFrame = spawnRate * dt;
  const credit = Math.min(input.respawnCredit + perFrame, perFrame + 1);
  const budget = Math.floor(credit);

  f[0] = maxx;
  f[1] = maxy;
  f[2] = (obstacleW - 1) / Math.max(1, maxx);
  f[3] = (obstacleH - 1) / Math.max(1, maxy);
  f[4] = dt;
  f[5] = decay(damping, dt);
  f[6] = decay(HEAT_DECAY, dt);
  f[7] = 1 - decay(HEAT_RISE, dt);
  f[8] = dt > 0 ? 1 / dt : 0;
  f[9] = swirl;
  f[10] = gravity;
  f[11] = drag;
  f[12] = life;
  f[13] = obstacleW;
  f[14] = obstacleH;
  f[15] = hasObstacle ? 1 : 0;
  u[16] = active;
  u[17] = opCount;
  u[18] = input.frameIndex;
  u[19] = budget;
  u[20] = sim.gridW;
  u[21] = sim.gridH;
  u[22] = 0;
  u[23] = 0;

  return { respawnCredit: credit - budget, dt };
}
