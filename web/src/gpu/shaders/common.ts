/**
 * WGSL pieces every pass in the WebGPU chain shares.
 *
 * Coordinate convention, stated once: `uv` in every fullscreen pass has y
 * *down*, matching WebGPU's texture origin at the top-left. The dye grid's row
 * 0 is the top of the screen, so it is sampled with `uv` directly, and every
 * intermediate target holds the image the way the screen shows it. The WebGL2
 * chain flips `v` at each read because GL's texture origin is at the bottom;
 * none of that is needed here.
 *
 * Intermediate targets are always `rgba16float`, which WebGPU guarantees to be
 * renderable and blendable, so there is no 8-bit encode/decode path and no
 * `EMIT`/`DECODE` contract to keep. Values written are linear radiance.
 */

import { NOISE_SIZE } from '../../render/noise';

/** Rec. 709 luminance, used identically by the camera treatment and the grade. */
export const LUMA_WGSL = /* wgsl */ `
fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }`;

/** Noise tile size, for the texel-exact grain fetches. */
export const NOISE_WGSL = /* wgsl */ `
const NOISE_SIZE: f32 = ${NOISE_SIZE.toFixed(1)};
const NOISE_MASK: i32 = ${NOISE_SIZE - 1};`;

/**
 * Fullscreen vertex stage: four vertices of a triangle strip synthesised from
 * the vertex index. Emits y-down `uv`.
 */
export const FULLSCREEN_VS = /* wgsl */ `
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  var out: VsOut;
  let p = vec2<f32>(f32((vi & 1u) << 1u), f32(vi & 2u)) - 1.0;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}`;
