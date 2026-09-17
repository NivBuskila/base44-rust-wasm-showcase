/**
 * Shader preamble and the pieces every stage shares.
 *
 * Shader bodies here carry no `#version` line: `buildShader` prepends it along
 * with the compile-time defines that select the HDR path. Branching on the
 * target format at compile time rather than on a uniform keeps the inner loops
 * free of dead arithmetic — and the two paths differ in more than a constant,
 * so a uniform would not have expressed it anyway.
 */

import { NOISE_SIZE } from '../noise';

/** Compile-time configuration shared by every program in the chain. */
export interface ShaderEnv {
  /** True when the intermediate targets are float and need no encoding. */
  float: boolean;
  /** Radiance that saturates an 8-bit intermediate; 1.0 on the float path. */
  range: number;
}

/**
 * Prepends the version line and the environment defines to a shader body.
 *
 * `DECODE`/`EMIT` are the contract between passes: everything written to an
 * intermediate target goes through `EMIT`, everything read back through
 * `DECODE`. Because the 8-bit encoding is a pure scale, additive blending
 * between the two stays linear, which is what lets the particle pass blend
 * into the scene target on either path.
 */
export function buildShader(body: string, env: ShaderEnv): string {
  const lines = [
    '#version 300 es',
    `#define HDR_FLOAT ${env.float ? 1 : 0}`,
    `#define HDR_RANGE ${env.range.toFixed(4)}`,
    // The bloom mips only ever hold energy above the bright-pass threshold, so
    // they get a tighter range than the scene: on 8-bit targets that is the
    // difference between visible contour rings around every highlight and none.
    `#define BLOOM_RANGE ${(env.float ? 1 : Math.min(env.range, 2.5)).toFixed(4)}`,
    `#define NOISE_SIZE ${NOISE_SIZE.toFixed(1)}`,
    `#define NOISE_MASK ${NOISE_SIZE - 1}`,
    '#if HDR_FLOAT',
    '#define DECODE(c) (c)',
    '#define EMIT(c) (c)',
    '#define BLOOM_DECODE(c) (c)',
    '#define BLOOM_EMIT(c) (c)',
    '#else',
    '#define DECODE(c) ((c) * HDR_RANGE)',
    '#define EMIT(c) ((c) * (1.0 / HDR_RANGE))',
    '#define BLOOM_DECODE(c) ((c) * BLOOM_RANGE)',
    '#define BLOOM_EMIT(c) ((c) * (1.0 / BLOOM_RANGE))',
    '#endif',
  ];
  return `${lines.join('\n')}\n${body}`;
}

/**
 * Fullscreen pass vertex shader: four vertices of a triangle strip synthesised
 * from `gl_VertexID`, so no vertex buffer and no VAO state are needed.
 */
export const FULLSCREEN_VERT = /* glsl */ `
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 1), float(gl_VertexID & 2)) - 1.0;
  v_uv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

/** Rec. 709 luminance, used identically by the camera treatment and the grade. */
export const LUMA = /* glsl */ `
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }`;
