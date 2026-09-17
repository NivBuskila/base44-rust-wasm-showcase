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
/** Frame counter, to reseed the dither every frame. */
uniform float u_frame;
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

void main() {
  vec2 d = v_uv - 0.5;
  float r2 = dot(d, d);

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
  vec2 off = d * r2 * u_aberration * 0.055;
  vec3 c = DECODE(texture(u_scene, v_uv).rgb) + vec3(
    bloomAt(v_uv + off, n.x).r,
    bloomAt(v_uv, n.y).g,
    bloomAt(v_uv - off, fract(n.x + n.y)).b);

  vec3 mapped = aces(c * u_exposure);

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
