/**
 * The draw half of the GPU pool: the instanced quad that renders one particle.
 * Byte-identical `Particle` struct to `particles-compute.ts` — the two sides of
 * the same buffer, pinned together by `shaders.test.ts`.
 */

/**
 * Draw: one instanced quad per particle, reading the pool directly. WebGPU
 * has no point sprites, so the six corners are synthesised from the vertex
 * index and the fragment gets the `[-1, 1]` corner the GL shader derived from
 * `gl_PointCoord`.
 *
 * Uniform: size, gain, intensity, 0 | viewport.xy, maxx, maxy.
 */
export const PARTICLE_DRAW_WGSL = /* wgsl */ `
struct Particle {
  x: f32, y: f32, vx: f32, vy: f32,
  life: f32, max_life: f32, heat: f32, pad: f32,
};

struct DrawUniforms {
  a: vec4<f32>, // size, gain, intensity, 0
  b: vec4<f32>, // viewport.x, viewport.y, maxx, maxy
};

@group(0) @binding(0) var<uniform> U: DrawUniforms;
@group(0) @binding(1) var<storage, read> P: array<Particle>;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) col: vec4<f32>,
  @location(1) corner: vec2<f32>,
};

const CORNERS = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));

fn heatRamp(t: f32) -> vec3<f32> {
  var c = mix(vec3<f32>(0.04, 0.09, 0.42), vec3<f32>(0.10, 0.66, 1.00), smoothstep(0.00, 0.30, t));
  c = mix(c, vec3<f32>(0.58, 0.34, 1.00), smoothstep(0.26, 0.56, t));
  c = mix(c, vec3<f32>(1.00, 0.44, 0.26), smoothstep(0.56, 0.84, t));
  c = mix(c, vec3<f32>(1.00, 0.93, 0.82), smoothstep(0.84, 1.00, t));
  return c;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
  var out: VsOut;
  let p = P[ii];
  let life = select(0.0, p.life / p.max_life, p.max_life > 0.0);
  let px = p.x / U.b.z;
  let py = p.y / U.b.w;
  let live = life > 0.0 && abs(px - 0.5) <= 4.0 && abs(py - 0.5) <= 4.0;
  if (!live) {
    // Degenerate quad: zero area, nothing rasterised.
    out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.col = vec4<f32>(0.0);
    out.corner = vec2<f32>(0.0);
    return out;
  }

  let corner = CORNERS[vi];
  let centre = vec2<f32>(px, 1.0 - py) * 2.0 - 1.0;
  let heat = clamp(p.heat, 0.0, 1.0);
  let fade = smoothstep(0.0, 0.16, life) * (0.22 + 0.78 * pow(life, 0.65));
  let want = U.a.x * (0.72 + 0.65 * heat) * (0.58 + 0.42 * life);
  let size = max(1.0, want);
  var floorFade = clamp(want, 0.0, 1.0);
  floorFade *= floorFade;
  let bright = floorFade * U.a.y * fade * (0.11 + 1.8 * heat) * (0.85 + 0.55 * U.a.z);

  out.pos = vec4<f32>(centre + corner * size / U.b.xy, 0.0, 1.0);
  out.col = vec4<f32>(heatRamp(heat) * bright, heat * heat);
  out.corner = corner;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let d = in.corner;
  let r2 = dot(d, d);
  if (r2 > 1.0) { discard; }
  var core = 1.0 - r2;
  core *= core;
  var c = in.col.rgb * core;
  if (in.col.a > 0.03) {
    let q = d * d * 22.0;
    let spark = (max(0.0, 1.0 - q.x) + max(0.0, 1.0 - q.y)) * (1.0 - r2);
    c += in.col.rgb * (in.col.a * spark * 0.4);
  }
  return vec4<f32>(c, core);
}`;
