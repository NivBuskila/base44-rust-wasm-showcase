/**
 * The pool's byte layout and dispatch constants, plus the generated op-kind
 * WGSL. Its own module so the two shader halves can read it without importing
 * the `particles.ts` index that re-exports them (that cycle left the spliced
 * op constants undefined at module-eval time).
 */

import { PARTICLE_OP } from '../../constants';

export const PARTICLE_FLOATS = 8;
export const PARTICLE_BYTES = PARTICLE_FLOATS * 4;
export const STEP_WORKGROUP = 256;
export const SIM_UNIFORM_FLOATS = 24;
export const DRAW_UNIFORM_FLOATS = 8;

/**
 * The op kinds as WGSL constants, generated from `PARTICLE_OP` so the shader
 * cannot hold a third copy of the protocol. `PARTICLE_OP` itself is checked
 * against Rust's `OP_KINDS` at boot by `assertOpLayout`.
 */
export const OP_KINDS_WGSL = Object.entries(PARTICLE_OP)
  .map(([name, value]) => `const OP_${name}: f32 = ${value.toFixed(1)};`)
  .join('\n');
