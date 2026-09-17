/**
 * Bloom: a mip chain built with a 13-tap downsample and folded back with a
 * 9-tap tent, the dual-filter scheme from Jimenez's Siggraph 2014 talk.
 *
 * Why not a separable Gaussian per level: a particle field is a cloud of
 * single-pixel highlights, and a plain box or two-pass Gaussian at half
 * resolution aliases them badly — the glow crawls and flickers frame to frame
 * as points cross texel boundaries. The 13-tap kernel is specifically shaped to
 * suppress that, and it does the blur and the downsample in one pass, so the
 * whole chain is five or six small draws instead of a dozen.
 */

export const BLOOM_DOWN_FRAG = /* glsl */ `
precision highp float;

in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_src;
/** Reciprocal of the *source* size; the taps are in source texels. */
uniform vec2 u_texel;
/** > 0 only on the first level, which doubles as the bright-pass. */
uniform float u_threshold;
uniform float u_knee;
/** 1 when reading the scene target (needs DECODE), 0 when reading a mip. */
uniform float u_fromScene;

void main() {
  vec3 a = texture(u_src, v_uv + u_texel * vec2(-2.0,  2.0)).rgb;
  vec3 b = texture(u_src, v_uv + u_texel * vec2( 0.0,  2.0)).rgb;
  vec3 c = texture(u_src, v_uv + u_texel * vec2( 2.0,  2.0)).rgb;
  vec3 d = texture(u_src, v_uv + u_texel * vec2(-2.0,  0.0)).rgb;
  vec3 e = texture(u_src, v_uv).rgb;
  vec3 f = texture(u_src, v_uv + u_texel * vec2( 2.0,  0.0)).rgb;
  vec3 g = texture(u_src, v_uv + u_texel * vec2(-2.0, -2.0)).rgb;
  vec3 h = texture(u_src, v_uv + u_texel * vec2( 0.0, -2.0)).rgb;
  vec3 i = texture(u_src, v_uv + u_texel * vec2( 2.0, -2.0)).rgb;
  vec3 j = texture(u_src, v_uv + u_texel * vec2(-1.0,  1.0)).rgb;
  vec3 k = texture(u_src, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
  vec3 l = texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
  vec3 m = texture(u_src, v_uv + u_texel * vec2( 1.0, -1.0)).rgb;

  vec3 sum = e * 0.125;
  sum += (a + c + g + i) * 0.03125;
  sum += (b + d + f + h) * 0.0625;
  sum += (j + k + l + m) * 0.125;

  // The scene target and the mips share one encoding, so this only exists
  // because the prefilter wants radiance to threshold against.
  if (u_fromScene > 0.5) sum = DECODE(sum);

  if (u_threshold > 0.0) {
    // Soft knee: a hard threshold makes the glow pop in and out as a highlight
    // crosses it, which on a fluid that is constantly brightening and dimming
    // looks like flicker.
    float lum = max(sum.r, max(sum.g, sum.b));
    float soft = clamp(lum - u_threshold + u_knee, 0.0, 2.0 * u_knee);
    soft = soft * soft / (4.0 * u_knee + 1.0e-4);
    sum *= max(soft, lum - u_threshold) / max(lum, 1.0e-4);
  }

  // Only the prefilter converts: it reads the scene's encoding and writes the
  // bloom chain's. Every level below it is already in bloom space, and because
  // both encodings are pure scales the tent blends stay linear.
  o_color = vec4(u_fromScene > 0.5 ? BLOOM_EMIT(sum) : sum, 1.0);
}`;

export const BLOOM_UP_FRAG = /* glsl */ `
precision highp float;

in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_src;
/** Reciprocal of the *source* size, scaled by the tent radius. */
uniform vec2 u_texel;
uniform float u_amount;

void main() {
  vec3 sum = texture(u_src, v_uv + u_texel * vec2(-1.0,  1.0)).rgb;
  sum += texture(u_src, v_uv + u_texel * vec2( 0.0,  1.0)).rgb * 2.0;
  sum += texture(u_src, v_uv + u_texel * vec2( 1.0,  1.0)).rgb;
  sum += texture(u_src, v_uv + u_texel * vec2(-1.0,  0.0)).rgb * 2.0;
  sum += texture(u_src, v_uv).rgb * 4.0;
  sum += texture(u_src, v_uv + u_texel * vec2( 1.0,  0.0)).rgb * 2.0;
  sum += texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
  sum += texture(u_src, v_uv + u_texel * vec2( 0.0, -1.0)).rgb * 2.0;
  sum += texture(u_src, v_uv + u_texel * vec2( 1.0, -1.0)).rgb;
  // Written with additive blending onto the level below, so the pass never
  // reads its own destination and no ping-pong target is needed.
  o_color = vec4(sum * (u_amount / 16.0), 1.0);
}`;
