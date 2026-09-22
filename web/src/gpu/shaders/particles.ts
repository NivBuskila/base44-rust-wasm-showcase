/**
 * The GPU particle pool: a compute kernel that is a line-for-line port of
 * `crates/aether-core/src/particles/{mod,lane}.rs`, and the vertex/fragment
 * pair that draws it, ported from `render/shaders/particles.ts`.
 *
 * ## Contract with the engine
 *
 * The Rust engine keeps the fluid, the spells and every bit of gameplay. When
 * the pool is offloaded it records what the spells *would* have done to the
 * particles — bursts, rings, impulses, damping fields, grips — as an op log
 * (`particles/offload.rs`), and `step_main` replays that log per particle in
 * call order before advancing, exactly as the CPU pool would have. Round-robin
 * spawn slots come from the engine too, so a burst overwrites the same
 * particles on either backend.
 *
 * Positions are in **grid space** like the CPU pool; the draw shader
 * normalises against `maxx, maxy`.
 *
 * ## Randomness
 *
 * The CPU pool draws from a per-lane xoshiro stream. A GPU thread cannot own a
 * stream, so every draw here is a PCG hash of `(particle, frame, salt)`. Same
 * distribution, different numbers — the two backends are statistically, not
 * bit-for-bit, identical.
 *
 * ## Respawn budget
 *
 * `counts.w` is the whole-particle credit this frame (the browser carries the
 * fraction). Dead particles race for it with one atomic each; the CPU pool
 * spends its credit front-to-back through the lanes, this one spends it in
 * whatever order the threads arrive. Both cap it at one frame plus one.
 */

export {
  DRAW_UNIFORM_FLOATS,
  OP_KINDS_WGSL,
  PARTICLE_BYTES,
  PARTICLE_FLOATS,
  SIM_UNIFORM_FLOATS,
  STEP_WORKGROUP,
} from './particles-layout';
export { PARTICLE_COMPUTE_WGSL } from './particles-compute';
export { PARTICLE_DRAW_WGSL } from './particles-draw';
