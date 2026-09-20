/**
 * Scene pass, WGSL port of `render/shaders/scene.ts`: treated camera behind a
 * cubic-resampled, domain-warped dye field, written as linear radiance.
 *
 * Uniform layout (`SceneUniforms`, six vec4s):
 *   0 dye:     dyeSize.xy, dyeTexel.xy
 *   1 cam:     camScale.xy, videoTexel.xy
 *   2 camTint: rgb, camToe
 *   3 camEdge: rgb, camRaw
 *   4 amounts: dyeAmount, bgAmount, hasVideo, intensity
 *   5 time:    time, 0, 0, 0
 */

import { FULLSCREEN_VS, LUMA_WGSL, NOISE_WGSL } from './common';

export const SCENE_UNIFORM_FLOATS = 24;

export const SCENE_WGSL = /* wgsl */ `
struct SceneUniforms {
  dye: vec4<f32>,
  cam: vec4<f32>,
  camTint: vec4<f32>,
  camEdge: vec4<f32>,
  amounts: vec4<f32>,
  time: vec4<f32>,
};

@group(0) @binding(0) var<uniform> U: SceneUniforms;
@group(0) @binding(1) var u_dye: texture_2d<f32>;
@group(0) @binding(2) var u_video: texture_2d<f32>;
@group(0) @binding(3) var u_noise: texture_2d<f32>;
@group(0) @binding(4) var s_clamp: sampler;
@group(0) @binding(5) var s_repeat: sampler;

${NOISE_WGSL}
${LUMA_WGSL}
${FULLSCREEN_VS}

/** Cubic B-spline resample in four bilinear taps (Sigg & Hadwiger). */
fn bspline(uv: vec2<f32>, size: vec2<f32>, texel: vec2<f32>) -> vec4<f32> {
  let coord = uv * size;
  let centre = floor(coord - 0.5) + 0.5;
  let f = coord - centre;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 / 6.0) * (-f3 + 3.0 * f2 - 3.0 * f + 1.0);
  let w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + 4.0);
  let w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0);
  let w3 = (1.0 / 6.0) * f3;
  let s0 = w0 + w1;
  let s1 = w2 + w3;
  let t0 = (centre - 1.0 + w1 / s0) * texel;
  let t1 = (centre + 1.0 + w3 / s1) * texel;
  let a = textureSample(u_dye, s_clamp, vec2<f32>(t0.x, t0.y));
  let b = textureSample(u_dye, s_clamp, vec2<f32>(t1.x, t0.y));
  let c = textureSample(u_dye, s_clamp, vec2<f32>(t0.x, t1.y));
  let d = textureSample(u_dye, s_clamp, vec2<f32>(t1.x, t1.y));
  return mix(mix(a, b, s1.x), mix(c, d, s1.x), s1.y);
}

/** Noise lookup expressed in dye cells: one tile texel per \`cells\` cells. */
fn cellNoise(cell: vec2<f32>, cells: f32) -> vec4<f32> {
  return textureSample(u_noise, s_repeat, cell / (cells * NOISE_SIZE));
}

fn dyeRadiance(uv: vec2<f32>, frag: vec2<f32>) -> vec3<f32> {
  let dyeSize = U.dye.xy;
  let dyeTexel = U.dye.zw;
  let time = U.time.x;
  let intensity = U.amounts.w;
  let cell = uv * dyeSize;

  let warp = cellNoise(cell + vec2<f32>(time * 1.7, -time * 1.3), 3.0).rg - 0.5;
  let warped = uv + warp * dyeTexel * 2.4;

  let s = bspline(warped, dyeSize, dyeTexel);

  // Undo the engine's 1 - exp(-x) packing.
  var lin = -log(max(vec3<f32>(1.0) - s.rgb * 0.996, vec3<f32>(1.0e-3)));

  let wcell = warped * dyeSize;
  let fine = cellNoise(wcell + vec2<f32>(0.0, time * 0.9), 0.55).g;
  let coarse = cellNoise(wcell - vec2<f32>(time * 0.5, 0.0), 26.0).r;
  let shift = i32(time * 24.0);
  let grainAt = (vec2<i32>(frag) + vec2<i32>(shift * 53, shift * 19)) & vec2<i32>(NOISE_MASK);
  let film = textureLoad(u_noise, grainAt, 0).b;
  lin *= (0.84 + 0.32 * fine * (0.4 + 1.2 * coarse)) * (0.92 + 0.16 * film);

  lin = max(mix(vec3<f32>(luma(lin)), lin, 1.24), vec3<f32>(0.0));
  lin += vec3<f32>(0.55, 0.78, 1.0) * pow(s.a, 10.0) * 0.9;

  return lin * (0.42 + 0.34 * intensity);
}

fn treatedCamera(iuv: vec2<f32>) -> vec3<f32> {
  let camTint = U.camTint.rgb;
  let camToe = U.camTint.w;
  let camEdge = U.camEdge.rgb;
  let camRaw = U.camEdge.w;
  if (camRaw >= 1.0) {
    return camTint * pow(textureSample(u_video, s_clamp, iuv).rgb, vec3<f32>(2.2));
  }
  let o = U.cam.zw * 2.0;
  let c = luma(textureSample(u_video, s_clamp, iuv).rgb);
  let xl = luma(textureSample(u_video, s_clamp, iuv - vec2<f32>(o.x, 0.0)).rgb);
  let xr = luma(textureSample(u_video, s_clamp, iuv + vec2<f32>(o.x, 0.0)).rgb);
  let yd = luma(textureSample(u_video, s_clamp, iuv - vec2<f32>(0.0, o.y)).rgb);
  let yu = luma(textureSample(u_video, s_clamp, iuv + vec2<f32>(0.0, o.y)).rgb);
  let soft = (2.0 * c + xl + xr + yd + yu) * (1.0 / 6.0);
  let edge = min(1.0, (abs(xr - xl) + abs(yu - yd)) * 1.7);
  let body = mix(soft, soft * soft * (0.40 + 0.60 * soft), camToe);
  return camTint * body + camEdge * edge;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let dyeAmount = U.amounts.x;
  let bgAmount = U.amounts.y;
  let hasVideo = U.amounts.z;
  let intensity = U.amounts.w;
  let time = U.time.x;
  var radiance = vec3<f32>(0.0);

  if (bgAmount > 0.0) {
    // The GL chain centres the gradient at y = 0.54 in a y-up frame; the same
    // point in this y-down frame is 0.46.
    let r = length((uv - vec2<f32>(0.5, 0.46)) * vec2<f32>(1.0, 0.82));
    var base = mix(vec3<f32>(0.00120, 0.00185, 0.00460), vec3<f32>(0.00016, 0.00024, 0.00070),
                   smoothstep(0.06, 0.78, r));
    let neb = textureSample(u_noise, s_repeat,
                            uv * vec2<f32>(0.22, 0.13) + vec2<f32>(time * 0.004, -time * 0.003)).r;
    base += vec3<f32>(0.00035, 0.00110, 0.00330) * neb * (0.3 + 1.2 * intensity);
    radiance += base * bgAmount;
  }

  if (hasVideo > 0.5) {
    // Mirrored horizontally to match the engine's flipped view convention.
    let iuv = vec2<f32>(1.0 - uv.x, uv.y);
    radiance += treatedCamera((iuv - 0.5) * U.cam.xy + 0.5);
  }

  if (dyeAmount > 0.0) {
    radiance += dyeRadiance(uv, in.pos.xy) * dyeAmount;
  }

  return vec4<f32>(radiance, 1.0);
}`;
