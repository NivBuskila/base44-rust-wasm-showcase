/**
 * Hand landmark overlay: bones as `gl.LINES`, joints as `gl.POINTS`, both
 * additive into the scene target so the bloom chain picks them up and the
 * skeleton glows instead of looking like a wireframe pasted on top.
 *
 * One program serves both draws. `gl_PointCoord` is undefined for line
 * primitives, so the radial falloff sits behind a uniform branch rather than
 * being multiplied out — reading an undefined builtin is how you get a driver
 * that returns NaN and a screen full of black holes.
 */

export const OVERLAY_VERT = /* glsl */ `
layout(location = 0) in vec2 a_pos;   // normalised view space, y down
layout(location = 1) in float a_glow;

uniform float u_alpha;
uniform float u_size;

out float v_glow;

void main() {
  gl_Position = vec4(vec2(a_pos.x, 1.0 - a_pos.y) * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = max(1.0, u_size);
  v_glow = a_glow * u_alpha;
}`;

export const OVERLAY_FRAG = /* glsl */ `
precision highp float;

in float v_glow;
out vec4 o_color;

uniform vec3 u_tint;
uniform float u_round;

void main() {
  float shape = 1.0;
  if (u_round > 0.5) {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    shape = exp(-2.6 * r2);
  }
  o_color = vec4(EMIT(u_tint * v_glow * shape), shape * v_glow);
}`;
