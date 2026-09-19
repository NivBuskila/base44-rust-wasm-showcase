/**
 * Final pass: scene plus bloom, tone mapped in linear light, graded, then
 * written to the 8-bit default framebuffer.
 *
 * The order matters and is not arbitrary. The chromatic aberration is applied
 * to linear radiance, before the tone map: a fringe added afterwards is flat
 * and reads as a coloured outline rather than as a lens. The tone map runs
 * before the sRGB transfer, because the ACES curve is defined on scene-referred
 * values. And the dither is the very last thing, in display space at exactly
 * one LSB, because the 8-bit write is the quantisation it exists to break up.
 */

import { LUMA } from './common';

export const COMPOSITE_FRAG = /* glsl */ `
precision highp float;

in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_noise;
uniform float u_bloomAmount;
uniform float u_exposure;
uniform float u_aberration;
uniform float u_vignette;
/** 1 applies the filmic tone map and grade; 0 passes linear light through. */
uniform float u_grade;
/** Frame counter, to reseed the dither every frame. */
uniform float u_frame;
/**
 * Lens rush: a bolt pushed at the camera. u_rushPower is 0 whenever nothing
 * is in flight and the whole block below is skipped; u_rushAt is the origin in
 * this pass's uv space and u_rushProgress runs 0 to 1 over its life.
 *
 * The field is two-dimensional, so depth here is entirely a camera trick: the
 * frame dollies in on the origin, the image smears radially outward from it,
 * and a core and shell of light grow through the smear. Those three together
 * are what reads as something arriving at the lens; any one alone reads as a
 * zoom, a blur or a flash.
 */
uniform vec2 u_rushAt;
uniform float u_rushProgress;
uniform float u_rushPower;
uniform float u_rushKind;
${LUMA}

vec3 bloomAt(vec2 uv, float jitter) {
  vec3 b = BLOOM_DECODE(texture(u_bloom, uv).rgb);
#if !HDR_FLOAT
  // On 8-bit targets the bloom chain quantises at one code per pass, and a
  // wide soft glow made of those steps is a set of concentric rings around
  // every highlight. Dithering by one bloom LSB on read fixes all of it in one
  // place instead of dithering six writes.
  b += (jitter - 0.5) * (BLOOM_RANGE / 255.0);
#endif
  return b * u_bloomAmount;
}

/** Narkowicz's ACES fit: one rational expression, close enough to the real
 * curve that the difference is invisible next to the grade below. */
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

/** Shape of the rush over its life: fast attack, long release. */
float rushEnvelope(float p) {
  return (1.0 - smoothstep(0.18, 1.0, p)) * smoothstep(0.0, 0.10, p);
}

/** A gunshot: instantaneous attack, then a hard fall. No approach at all. */
float shotEnvelope(float p) {
  return pow(1.0 - p, 3.0);
}

void main() {
  vec2 d = v_uv - 0.5;
  float r2 = dot(d, d);

  // --- lens rush: dolly + radial smear, both skipped when nothing is flying
  vec2 uv = v_uv;
  vec3 rushLight = vec3(0.0);
  float rushAber = 0.0;
  vec3 sceneColor = vec3(-1.0);
  if (u_rushPower > 0.0 && u_rushKind < 0.5) {
    float p = clamp(u_rushProgress, 0.0, 1.0);
    float amt = u_rushPower * rushEnvelope(p);
    vec2 rd = v_uv - u_rushAt;
    // Dolly: sampling a smaller window around the origin magnifies the image
    // out of it, which is the same screen motion as the origin coming closer.
    uv = u_rushAt + rd / (1.0 + 0.34 * amt);

    // Streaks: five taps walking back towards the origin along the same line
    // the dolly stretches. A radial smear is the one blur that reads as travel
    // through the lens rather than as a soft image.
    vec3 smear = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 5; i++) {
      float k = float(i) / 4.0;
      float w = 1.0 - 0.55 * k;
      smear += DECODE(texture(u_scene, u_rushAt + rd * (1.0 - 0.16 * amt * k)).rgb) * w;
      wsum += w;
    }
    smear /= wsum;

    // Light: a hot core at the origin plus a shell expanding past the frame,
    // the same cyan the bolt itself is made of, tinted white at the core so it
    // reads as intensity and not as a coloured blob.
    float dist = length(rd);
    float shell = exp(-pow((dist - 1.05 * p) / (0.07 + 0.30 * p), 2.0));
    float core = exp(-dist * dist / (0.006 + 0.055 * p));
    vec3 tint = mix(vec3(0.35, 0.85, 1.0), vec3(1.0), 0.45 * core);
    rushLight = tint * (core * 2.1 + shell * 0.85) * u_rushPower * (1.0 - smoothstep(0.25, 1.0, p));

    // The smear stands in for the scene under the dolly rather than being added
    // to it: a second copy of the image would only double the exposure.
    sceneColor = mix(DECODE(texture(u_scene, uv).rgb), smear, 0.75 * amt);
    rushAber = amt;
  } else if (u_rushPower > 0.0) {
    // --- gunshot at the lens: muzzle blast, recoil kick, powder-hot flash.
    //
    // Nothing here dollies in. A shot leaves the muzzle, so the frame is thrown
    // *back* from it and settles: a brief push of the whole image away from the
    // fingertip, decaying in a couple of bounces, which is what a recoiling
    // camera does. The light is a tight muzzle flare with a few radial spikes
    // rather than an expanding shell, and it is gone almost immediately.
    float p = clamp(u_rushProgress, 0.0, 1.0);
    float amt = u_rushPower * shotEnvelope(p);
    vec2 rd = v_uv - u_rushAt;
    float dist = length(rd);

    // Recoil: the image is shoved away from the muzzle and rings out, the
    // bounce riding the same decay so it never leaves the frame displaced.
    float ring = cos(p * 46.0) * exp(-p * 9.0);
    uv = v_uv + normalize(rd + 1e-5) * (0.05 * amt * ring);

    // Muzzle flare: a hot core, a thin ragged corona, and star spikes. Amber at
    // the edge, white at the centre, the colour of burning powder.
    float flare = exp(-dist * dist / 0.0016) * 2.6;
    float ang = atan(rd.y, rd.x);
    float spikes = pow(max(0.0, cos(ang * 6.0)), 10.0) * exp(-dist * 18.0) * 1.1;
    float crack = exp(-pow((dist - 0.10) / 0.035, 2.0)) * 0.7;
    vec3 tint = mix(vec3(1.0, 0.62, 0.18), vec3(1.0), 0.7 * exp(-dist * 30.0));
    rushLight = tint * (flare + spikes + crack) * u_rushPower * shotEnvelope(p);

    // Smoke shadow: the blast darkens what sits just around the muzzle, so the
    // flash reads as a discharge in front of the scene, not a glow inside it.
    vec3 base = DECODE(texture(u_scene, uv).rgb);
    sceneColor = base * (1.0 - 0.45 * amt * exp(-pow((dist - 0.07) / 0.06, 2.0)));
    rushAber = amt * 0.6;
  }

  // Two independent white-noise taps, offset per frame so the pattern does not
  // sit still. Shared by the bloom read and the final dither.
  ivec2 np = ivec2(gl_FragCoord.xy) + ivec2(int(u_frame) * 37, int(u_frame) * 71);
  vec2 n = texelFetch(u_noise, np & ivec2(NOISE_MASK), 0).ba;

  // Transverse chromatic aberration. The offset grows with r^2 like a real
  // lens, so the centre of the frame stays clean and only the edges fringe.
  //
  // Split on the bloom only, not on the scene: the fringe is only legible
  // where the image is bright and soft, which is exactly what the bloom holds,
  // and splitting the scene as well would cost two more full-resolution taps
  // in the most expensive pass of the frame for no visible gain.
  //
  // During a rush the fringe also grows along the rush axis, not just with r^2:
  // a lens flexes hardest where the image is moving through it.
  vec2 rushDir = (v_uv - u_rushAt) * (0.045 * rushAber);
  vec2 off = d * r2 * u_aberration * 0.055 + rushDir;
  vec3 scene = sceneColor.r >= 0.0 ? sceneColor : DECODE(texture(u_scene, uv).rgb);
  vec3 c = scene + vec3(
    bloomAt(uv + off, n.x).r,
    bloomAt(uv, n.y).g,
    bloomAt(uv - off, fract(n.x + n.y)).b) + rushLight;

  vec3 mapped;
  if (u_grade >= 1.0) {
    mapped = aces(c * u_exposure);

    // ACES pushes highlights toward white, which on a field this saturated eats
    // the hue right off the bright core of every arm. Winding a little chroma
    // back in afterwards is cheaper than a hue-preserving tone map and the
    // difference between the two is not visible here.
    float l = luma(mapped);
    mapped = max(mix(vec3(l), mapped, 1.14), vec3(0.0));

    // Split tone: cool shadows, warm highlights. Two multiplies and a mix, and
    // most of the distance between "blue fluid on black" and "graded image".
    mapped = mix(mapped * vec3(0.90, 0.97, 1.14), mapped * vec3(1.07, 1.00, 0.92),
                 smoothstep(0.22, 0.85, l));
  } else {
    // Camera view: no film look, the feed is shown as the sensor saw it.
    mapped = clamp(c * u_exposure, 0.0, 1.0);
  }

  mapped *= 1.0 - u_vignette * smoothstep(0.10, 0.80, r2 * 1.7);

  // sRGB transfer. The chain above is linear; writing it straight to an 8-bit
  // surface would crush every dark gradient in the image into a handful of
  // codes.
  vec3 srgb = pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));

  // Triangular-PDF dither at one LSB. Uniform noise of the same amplitude
  // leaves a visible texture on flat areas; the sum of two does not, and this
  // image is nothing but large flat dark gradients.
  o_color = vec4(srgb + (n.x + n.y - 1.0) * (1.0 / 255.0), 1.0);
}`;

/**
 * Debug view. The contract is that `frame.debug` is shown raw — red is the
 * obstacle field, green and blue are signed optical flow around a 0.5 bias —
 * so this deliberately skips the grade, the bloom and the tone map. Nearest
 * filtering is part of that: seeing the actual 256x144 cells is the point of
 * the view.
 */
export const BLIT_FRAG = /* glsl */ `
precision highp float;

in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_src;

void main() {
  o_color = vec4(texture(u_src, vec2(v_uv.x, 1.0 - v_uv.y)).rgb, 1.0);
}`;
