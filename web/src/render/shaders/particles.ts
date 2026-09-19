/**
 * The particle pass: one `gl.POINTS` draw over the interleaved
 * `x, y, heat, life` buffer the engine streams in, blended additively into the
 * scene target.
 *
 * Everything that varies per particle is computed in the vertex shader from the
 * four floats already in the buffer. Nothing is precomputed on the CPU and
 * nothing is indexed per particle from JS — at 220k particles a single
 * `Math.*` call per particle per frame would be the frame budget.
 */

export const PARTICLE_VERT = /* glsl */ `
layout(location = 0) in vec4 a_particle; // x, y in [0,1]; heat in [0,1]; life in [0,1]

uniform float u_size;      // base point diameter, device pixels
uniform float u_gain;      // energy normalisation for the pool size
uniform float u_intensity;

out vec4 v_col;            // rgb = premultiplied radiance, a = spark weight

/**
 * Heat ramp: indigo -> electric cyan -> violet -> ember -> white hot.
 *
 * Sequential mixes rather than a cosine palette: the stops are placed by eye
 * so the mid range (where most particles live) lands on cyan rather than on
 * the muddy blue-violet a smooth analytic palette gives between those hues.
 */
vec3 heatRamp(float t) {
  vec3 c = mix(vec3(0.04, 0.09, 0.42), vec3(0.10, 0.66, 1.00), smoothstep(0.00, 0.30, t));
  c = mix(c, vec3(0.58, 0.34, 1.00), smoothstep(0.26, 0.56, t));
  c = mix(c, vec3(1.00, 0.44, 0.26), smoothstep(0.56, 0.84, t));
  c = mix(c, vec3(1.00, 0.93, 0.82), smoothstep(0.84, 1.00, t));
  return c;
}

void main() {
  float life = a_particle.w;
  vec2 p = a_particle.xy;
  // Dead slots stay in the buffer at life 0, and a NaN fails every comparison
  // below, so this one test covers both. Parking the vertex behind the far
  // plane is cheaper than a discard per covered fragment.
  bool live = life > 0.0 && abs(p.x - 0.5) <= 4.0 && abs(p.y - 0.5) <= 4.0;
  if (!live) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    v_col = vec4(0.0);
    return;
  }

  // Grid row 0 is the top of the screen; clip space has y up.
  gl_Position = vec4(vec2(p.x, 1.0 - p.y) * 2.0 - 1.0, 0.0, 1.0);

  float heat = clamp(a_particle.z, 0.0, 1.0);
  // Life runs 1 -> 0. The smoothstep gives the last instants a fast fade and
  // the power curve keeps the long tail dim; a linear ramp makes the whole
  // pool read as uniform grey haze.
  float fade = smoothstep(0.0, 0.16, life) * (0.22 + 0.78 * pow(life, 0.65));
  float want = u_size * (0.72 + 0.65 * heat) * (0.58 + 0.42 * life);
  gl_PointSize = max(1.0, want);
  // A point cannot be drawn smaller than one pixel, so on a dense small frame
  // the floor silently hands every mote extra area and the dust turns into
  // bright confetti. Dim it by the area it did not earn instead.
  float floorFade = clamp(want, 0.0, 1.0);
  floorFade *= floorFade;
  // The cold end stays dust-faint on purpose. 120k cold particles cover an
  // eighth of a 720p frame, and at even a quarter of the hot brightness they
  // stop reading as individual motes and become a flat blue veil that nothing
  // else can be seen against. Heat is what earns brightness here.
  float bright = floorFade * u_gain * fade * (0.11 + 1.8 * heat) * (0.85 + 0.55 * u_intensity);
  v_col = vec4(heatRamp(heat) * bright, heat * heat);
}`;

export const PARTICLE_FRAG = /* glsl */ `
precision highp float;

in vec4 v_col;
out vec4 o_color;

void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;

  // A smooth core plus an anisotropic cross on the hot end: bright points in a
  // real lens do not fall off radially, and that cross is the whole difference
  // between "spark" and "dot" once the bloom picks it up.
  //
  // Squared falloff rather than a Gaussian, and the cross behind a branch. This
  // shader runs on up to 220k overlapping points and is the single most
  // expensive pass in the frame; three transcendentals per fragment here cost
  // more than the entire rest of the chain, and at a 2-pixel sprite the
  // difference from a true Gaussian is not visible.
  float core = 1.0 - r2;
  core *= core;
  vec3 c = v_col.rgb * core;
  if (v_col.a > 0.03) {
    vec2 q = d * d * 22.0;
    float spark = (max(0.0, 1.0 - q.x) + max(0.0, 1.0 - q.y)) * (1.0 - r2);
    c += v_col.rgb * (v_col.a * spark * 0.4);
  }
  o_color = vec4(EMIT(c), core);
}`;
