/**
 * Hand landmark overlay, WGSL port of `render/shaders/overlay.ts`.
 *
 * Bones are a `line-list` draw over the packed `x, y, glow` vertex buffer
 * (WebGPU lines are one pixel wide, which is what the GL chain used). Joints
 * have no point-sprite primitive to fall back on, so they are an instanced
 * quad per joint reading the same buffer with `stepMode: 'instance'`.
 *
 * Uniform: alpha, size (device px), viewport.xy | tint.rgb, round | viewFit.xy, 0, 0.
 */

export const OVERLAY_UNIFORM_FLOATS = 12;

const OVERLAY_COMMON = /* wgsl */ `
struct OverlayUniforms {
  a: vec4<f32>, // alpha, size, viewport.x, viewport.y
  b: vec4<f32>, // tint.rgb, round
  c: vec4<f32>, // viewFit.x, viewFit.y (render/look.ts), 0, 0
};
@group(0) @binding(0) var<uniform> U: OverlayUniforms;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) glow: f32,
  @location(1) corner: vec2<f32>,
};

/**
 * Landmarks are normalised to the whole camera frame, so they take the feed's
 * crop: that is what keeps the skeleton on the hand the screen shows.
 */
fn toClip(pos: vec2<f32>) -> vec2<f32> {
  let view = (pos - 0.5) / U.c.xy + 0.5;
  return vec2<f32>(view.x, 1.0 - view.y) * 2.0 - 1.0;
}

fn shade(glow: f32, corner: vec2<f32>) -> vec4<f32> {
  var shape = 1.0;
  if (U.b.w > 0.5) {
    let r2 = dot(corner, corner);
    if (r2 > 1.0) { discard; }
    shape = exp(-2.6 * r2);
  }
  return vec4<f32>(U.b.rgb * glow * shape, shape * glow);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return shade(in.glow, in.corner);
}`;

export const OVERLAY_LINES_WGSL = /* wgsl */ `
${OVERLAY_COMMON}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) glow: f32) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(toClip(pos), 0.0, 1.0);
  out.glow = glow * U.a.x;
  out.corner = vec2<f32>(0.0);
  return out;
}`;

export const OVERLAY_POINTS_WGSL = /* wgsl */ `
${OVERLAY_COMMON}

const CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));

@vertex
fn vs_main(@builtin(vertex_index) vi: u32,
           @location(0) pos: vec2<f32>, @location(1) glow: f32) -> VsOut {
  var out: VsOut;
  let corner = CORNERS[vi];
  let centre = toClip(pos);
  let size = max(1.0, U.a.y);
  out.pos = vec4<f32>(centre + corner * size / U.a.zw, 0.0, 1.0);
  out.glow = glow * U.a.x;
  out.corner = corner;
  return out;
}`;
