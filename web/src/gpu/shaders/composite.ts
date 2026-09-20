/**
 * Final pass, WGSL port of `render/shaders/composite.ts`: scene plus bloom,
 * chromatic aberration in linear light, ACES tone map, split tone, vignette,
 * sRGB transfer and a triangular dither, straight to the canvas.
 *
 * Uniform layout (three vec4s):
 *   0 amounts: bloomAmount, exposure, aberration, vignette
 *   1 misc:    grade, frame, rushProgress, rushPower
 *   2 rushAt:  x, y (y-down uv), 0, 0
 */

import { FULLSCREEN_VS, LUMA_WGSL, NOISE_WGSL } from './common';

export const COMPOSITE_UNIFORM_FLOATS = 12;

export const COMPOSITE_WGSL = /* wgsl */ `
struct CompositeUniforms {
  amounts: vec4<f32>,
  misc: vec4<f32>,
  rushAt: vec4<f32>,
};

@group(0) @binding(0) var<uniform> U: CompositeUniforms;
@group(0) @binding(1) var u_scene: texture_2d<f32>;
@group(0) @binding(2) var u_bloom: texture_2d<f32>;
@group(0) @binding(3) var u_noise: texture_2d<f32>;
@group(0) @binding(4) var s_clamp: sampler;

${NOISE_WGSL}
${LUMA_WGSL}
${FULLSCREEN_VS}

fn bloomAt(uv: vec2<f32>) -> vec3<f32> {
  return textureSample(u_bloom, s_clamp, uv).rgb * U.amounts.x;
}

/** Narkowicz's ACES fit. */
fn aces(x: vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

/** Shape of the rush over its life: fast attack, long release. */
fn rushEnvelope(p: f32) -> f32 {
  return (1.0 - smoothstep(0.18, 1.0, p)) * smoothstep(0.0, 0.10, p);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let exposure = U.amounts.y;
  let aberration = U.amounts.z;
  let vignette = U.amounts.w;
  let grade = U.misc.x;
  let frame = i32(U.misc.y);
  let rushProgress = U.misc.z;
  let rushPower = U.misc.w;
  let rushAt = U.rushAt.xy;

  let d = in.uv - 0.5;
  let r2 = dot(d, d);

  // --- lens rush: dolly + radial smear, skipped when nothing is flying
  var uv = in.uv;
  var rushLight = vec3<f32>(0.0);
  var rushAber = 0.0;
  var sceneColor = vec3<f32>(-1.0);
  if (rushPower > 0.0) {
    let p = clamp(rushProgress, 0.0, 1.0);
    let amt = rushPower * rushEnvelope(p);
    let rd = in.uv - rushAt;
    uv = rushAt + rd / (1.0 + 0.34 * amt);

    var smear = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0; i < 5; i++) {
      let k = f32(i) / 4.0;
      let w = 1.0 - 0.55 * k;
      smear += textureSample(u_scene, s_clamp, rushAt + rd * (1.0 - 0.16 * amt * k)).rgb * w;
      wsum += w;
    }
    smear /= wsum;

    let dist = length(rd);
    let shell = exp(-pow((dist - 1.05 * p) / (0.07 + 0.30 * p), 2.0));
    let core = exp(-dist * dist / (0.006 + 0.055 * p));
    let tint = mix(vec3<f32>(0.35, 0.85, 1.0), vec3<f32>(1.0), 0.45 * core);
    rushLight = tint * (core * 2.1 + shell * 0.85) * rushPower * (1.0 - smoothstep(0.25, 1.0, p));

    sceneColor = mix(textureSample(u_scene, s_clamp, uv).rgb, smear, 0.75 * amt);
    rushAber = amt;
  }

  // Two independent white-noise taps, offset per frame.
  let np = (vec2<i32>(in.pos.xy) + vec2<i32>(frame * 37, frame * 71)) & vec2<i32>(NOISE_MASK);
  let n = textureLoad(u_noise, np, 0).ba;

  let rushDir = (in.uv - rushAt) * (0.045 * rushAber);
  let off = d * r2 * aberration * 0.055 + rushDir;
  var scene: vec3<f32>;
  if (sceneColor.r >= 0.0) {
    scene = sceneColor;
  } else {
    scene = textureSample(u_scene, s_clamp, uv).rgb;
  }
  let c = scene + vec3<f32>(bloomAt(uv + off).r, bloomAt(uv).g, bloomAt(uv - off).b) + rushLight;

  var mapped: vec3<f32>;
  if (grade >= 1.0) {
    mapped = aces(c * exposure);
    let l = luma(mapped);
    mapped = max(mix(vec3<f32>(l), mapped, 1.14), vec3<f32>(0.0));
    mapped = mix(mapped * vec3<f32>(0.90, 0.97, 1.14), mapped * vec3<f32>(1.07, 1.00, 0.92),
                 smoothstep(0.22, 0.85, l));
  } else {
    mapped = clamp(c * exposure, vec3<f32>(0.0), vec3<f32>(1.0));
  }

  mapped *= 1.0 - vignette * smoothstep(0.10, 0.80, r2 * 1.7);

  let srgb = pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
  return vec4<f32>(srgb + (n.x + n.y - 1.0) * (1.0 / 255.0), 1.0);
}`;

/** Debug view: the raw obstacle/flow texture, ungraded, nearest-filtered. */
export const BLIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var u_src: texture_2d<f32>;
@group(0) @binding(1) var s_nearest: sampler;

${FULLSCREEN_VS}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSample(u_src, s_nearest, in.uv).rgb, 1.0);
}`;
