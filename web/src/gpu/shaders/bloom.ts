/**
 * Bloom, WGSL port of `render/shaders/bloom.ts`: a 13-tap downsample chain
 * and a 9-tap tent fold-back (Jimenez 2014). The first downsample doubles as
 * the soft-knee bright pass.
 *
 * Down uniform: texel.xy, threshold, knee. Up uniform: texel.xy, amount, 0.
 */

import { FULLSCREEN_VS } from './common';

export const BLOOM_DOWN_WGSL = /* wgsl */ `
struct DownUniforms { p: vec4<f32> };
@group(0) @binding(0) var<uniform> U: DownUniforms;
@group(0) @binding(1) var u_src: texture_2d<f32>;
@group(0) @binding(2) var s_clamp: sampler;

${FULLSCREEN_VS}

fn tap(uv: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
  return textureSample(u_src, s_clamp, uv + U.p.xy * o).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let a = tap(uv, vec2<f32>(-2.0,  2.0));
  let b = tap(uv, vec2<f32>( 0.0,  2.0));
  let c = tap(uv, vec2<f32>( 2.0,  2.0));
  let d = tap(uv, vec2<f32>(-2.0,  0.0));
  let e = tap(uv, vec2<f32>( 0.0,  0.0));
  let f = tap(uv, vec2<f32>( 2.0,  0.0));
  let g = tap(uv, vec2<f32>(-2.0, -2.0));
  let h = tap(uv, vec2<f32>( 0.0, -2.0));
  let i = tap(uv, vec2<f32>( 2.0, -2.0));
  let j = tap(uv, vec2<f32>(-1.0,  1.0));
  let k = tap(uv, vec2<f32>( 1.0,  1.0));
  let l = tap(uv, vec2<f32>(-1.0, -1.0));
  let m = tap(uv, vec2<f32>( 1.0, -1.0));

  var sum = e * 0.125;
  sum += (a + c + g + i) * 0.03125;
  sum += (b + d + f + h) * 0.0625;
  sum += (j + k + l + m) * 0.125;

  let threshold = U.p.z;
  let knee = U.p.w;
  if (threshold > 0.0) {
    let lum = max(sum.r, max(sum.g, sum.b));
    var soft = clamp(lum - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1.0e-4);
    sum *= max(soft, lum - threshold) / max(lum, 1.0e-4);
  }
  return vec4<f32>(sum, 1.0);
}`;

export const BLOOM_UP_WGSL = /* wgsl */ `
struct UpUniforms { p: vec4<f32> };
@group(0) @binding(0) var<uniform> U: UpUniforms;
@group(0) @binding(1) var u_src: texture_2d<f32>;
@group(0) @binding(2) var s_clamp: sampler;

${FULLSCREEN_VS}

fn tap(uv: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
  return textureSample(u_src, s_clamp, uv + U.p.xy * o).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  var sum = tap(uv, vec2<f32>(-1.0,  1.0));
  sum += tap(uv, vec2<f32>( 0.0,  1.0)) * 2.0;
  sum += tap(uv, vec2<f32>( 1.0,  1.0));
  sum += tap(uv, vec2<f32>(-1.0,  0.0)) * 2.0;
  sum += tap(uv, vec2<f32>( 0.0,  0.0)) * 4.0;
  sum += tap(uv, vec2<f32>( 1.0,  0.0)) * 2.0;
  sum += tap(uv, vec2<f32>(-1.0, -1.0));
  sum += tap(uv, vec2<f32>( 0.0, -1.0)) * 2.0;
  sum += tap(uv, vec2<f32>( 1.0, -1.0));
  // Additively blended onto the level below, so no ping-pong target.
  return vec4<f32>(sum * (U.p.z / 16.0), 1.0);
}`;
