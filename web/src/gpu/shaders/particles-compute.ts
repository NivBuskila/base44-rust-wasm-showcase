/**
 * The compute half of the GPU pool: the op-log replay and both compute entry
 * points. The protocol, the randomness and the respawn-budget rules it obeys
 * are documented in `particles.ts`, which owns the constants this reads.
 */

import { OP_KINDS_WGSL, STEP_WORKGROUP } from './particles-layout';

export const PARTICLE_COMPUTE_WGSL = /* wgsl */ `
struct Particle {
  x: f32, y: f32, vx: f32, vy: f32,
  life: f32, max_life: f32, heat: f32, pad: f32,
};

struct Sim {
  grid: vec4<f32>,    // maxx, maxy, osx, osy
  rates: vec4<f32>,   // dt, keep, heat_keep, heat_rise
  misc: vec4<f32>,    // inv_dt, swirl, gravity, drag
  life_ob: vec4<f32>, // life, obstacle_w, obstacle_h, has_obstacle
  counts: vec4<u32>,  // active, ops, frame, respawn_budget
  dims: vec4<u32>,    // grid_w, grid_h, 0, 0
};

struct Counters { respawns: atomic<u32>, alive: atomic<u32> };

@group(0) @binding(0) var<uniform> S: Sim;
@group(0) @binding(1) var<storage, read_write> P: array<Particle>;
@group(0) @binding(2) var<storage, read> vel_u: array<f32>;
@group(0) @binding(3) var<storage, read> vel_v: array<f32>;
@group(0) @binding(4) var<storage, read> obstacle: array<f32>;
@group(0) @binding(5) var<storage, read> ops: array<vec4<f32>>; // 3 vec4 per op
@group(0) @binding(6) var<storage, read_write> C: Counters;

const HEAT_SPEED_FULL: f32 = 220.0;
const MAX_OWN_SPEED: f32 = 2000.0;
const RESPAWN_TRIES: u32 = 3u;
const LIFE_JITTER_LO: f32 = 0.6;
const LIFE_JITTER_HI: f32 = 1.4;
const BURST_SCATTER: f32 = 1.5;
const TAU: f32 = 6.283185307179586;

${OP_KINDS_WGSL}

// ------------------------------------------------------------------ random

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

struct Rng { state: u32 };

fn rng_new(i: u32, salt: u32) -> Rng {
  var r: Rng;
  r.state = pcg(i ^ (S.counts.z * 0x9E3779B9u) ^ (salt * 0x85EBCA6Bu));
  return r;
}

fn rnd(r: ptr<function, Rng>) -> f32 {
  (*r).state = pcg((*r).state);
  return f32((*r).state >> 8u) * (1.0 / 16777216.0);
}

fn rnd_range(r: ptr<function, Rng>, lo: f32, hi: f32) -> f32 {
  return lo + (hi - lo) * rnd(r);
}

fn in_disc(r: ptr<function, Rng>) -> vec2<f32> {
  let theta = rnd(r) * TAU;
  let rad = sqrt(rnd(r));
  return vec2<f32>(rad * cos(theta), rad * sin(theta));
}

// ------------------------------------------------------------------- field

fn is_finite(v: f32) -> bool { return abs(v) < 3.0e38; }

fn tame(v: f32) -> f32 {
  if (is_finite(v)) { return clamp(v, -MAX_OWN_SPEED, MAX_OWN_SPEED); }
  return 0.0;
}

/** Hermite ease matching aether_core::math::smoothstep, edges in either order. */
fn sstep(edge0: f32, edge1: f32, x: f32) -> f32 {
  if (abs(edge1 - edge0) < 1.1920929e-7) {
    return select(1.0, 0.0, x < edge0);
  }
  let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn sample_uv(x_in: f32, y_in: f32) -> vec2<f32> {
  let w = S.dims.x;
  let h = S.dims.y;
  let x = clamp(x_in, 0.0, f32(w - 1u));
  let y = clamp(y_in, 0.0, f32(h - 1u));
  let ix0 = u32(x);
  let iy0 = u32(y);
  let tx = x - f32(ix0);
  let ty = y - f32(iy0);
  let ix1 = min(ix0 + 1u, w - 1u);
  let iy1 = min(iy0 + 1u, h - 1u);
  let r0 = iy0 * w;
  let r1 = iy1 * w;
  let u = mix(mix(vel_u[r0 + ix0], vel_u[r0 + ix1], tx), mix(vel_u[r1 + ix0], vel_u[r1 + ix1], tx), ty);
  let v = mix(mix(vel_v[r0 + ix0], vel_v[r0 + ix1], tx), mix(vel_v[r1 + ix0], vel_v[r1 + ix1], tx), ty);
  return vec2<f32>(u, v);
}

fn curl_at(x: f32, y: f32) -> f32 {
  let w = S.dims.x;
  let h = S.dims.y;
  let xi = min(u32(max(x + 0.5, 0.0)), w - 1u);
  let yi = min(u32(max(y + 0.5, 0.0)), h - 1u);
  let xm = select(xi - 1u, 0u, xi == 0u);
  let xp = min(xi + 1u, w - 1u);
  let ym = select(yi - 1u, 0u, yi == 0u);
  let yp = min(yi + 1u, h - 1u);
  let inv_dx = select(1.0, 0.5, xp - xm == 2u);
  let inv_dy = select(1.0, 0.5, yp - ym == 2u);
  let dvdx = (vel_v[yi * w + xp] - vel_v[yi * w + xm]) * inv_dx;
  let dudy = (vel_u[yp * w + xi] - vel_u[ym * w + xi]) * inv_dy;
  return dvdx - dudy;
}

fn is_solid(x: f32, y: f32) -> bool {
  if (S.life_ob.w < 0.5) { return false; }
  let ow = u32(S.life_ob.y);
  let oh = u32(S.life_ob.z);
  let xi = min(u32(max(x * S.grid.z + 0.5, 0.0)), ow - 1u);
  let yi = min(u32(max(y * S.grid.w + 0.5, 0.0)), oh - 1u);
  return obstacle[yi * ow + xi] >= 0.5;
}

// -------------------------------------------------------------------- ops

/** Round-robin membership: is particle i one of the \`count\` slots from \`slot\`? */
fn in_slots(i: u32, slot: f32, count: f32) -> bool {
  let n = S.counts.x;
  let s = u32(max(slot, 0.0)) % max(n, 1u);
  let k = (i + n - s) % n;
  return f32(k) < count;
}

fn apply_ops(i: u32, p_in: Particle) -> Particle {
  var p = p_in;
  let n_ops = S.counts.y;
  for (var j = 0u; j < n_ops; j++) {
    let a = ops[j * 3u];
    let b = ops[j * 3u + 1u];
    let c = ops[j * 3u + 2u];
    let kind = a.x;
    let cx = a.y;
    let cy = a.z;

    if (kind == OP_BURST) {
      // a: kind, x, y, count | b: speed, heat, life, slot
      if (in_slots(i, b.w, a.w)) {
        var r = rng_new(i, j + 1u);
        let d = in_disc(&r);
        p.x = cx + d.x * BURST_SCATTER;
        p.y = cy + d.y * BURST_SCATTER;
        p.vx = d.x * b.x;
        p.vy = d.y * b.x;
        p.max_life = b.z * rnd_range(&r, LIFE_JITTER_LO, LIFE_JITTER_HI);
        p.life = p.max_life;
        p.heat = b.y;
      }
    } else if (kind == OP_RING) {
      // a: kind, x, y, count | b: ring, speed, heat, life | c: slot
      if (in_slots(i, c.x, a.w)) {
        var r = rng_new(i, j + 1u);
        let theta = rnd(&r) * TAU;
        let d = vec2<f32>(cos(theta), sin(theta));
        let jit = in_disc(&r);
        p.x = cx + d.x * b.x + jit.x * BURST_SCATTER;
        p.y = cy + d.y * b.x + jit.y * BURST_SCATTER;
        p.vx = d.x * b.y;
        p.vy = d.y * b.y;
        p.max_life = b.w * rnd_range(&r, LIFE_JITTER_LO, LIFE_JITTER_HI);
        p.life = p.max_life;
        p.heat = b.z;
      }
    } else if (p.life > 0.0) {
      // a: kind, x, y, radius | b: strength/k, heat, -, -
      let radius = a.w;
      let dx = p.x - cx;
      let dy = p.y - cy;
      let d2 = dx * dx + dy * dy;
      if (radius > 0.0 && d2 <= radius * radius) {
        let falloff = sstep(radius, 0.0, sqrt(d2));
        if (kind == OP_IMPULSE) {
          let len = sqrt(d2);
          var nx = 0.0;
          var ny = 0.0;
          if (len > 1e-6) { nx = dx / len; ny = dy / len; }
          p.vx = tame(p.vx + nx * b.x * falloff);
          p.vy = tame(p.vy + ny * b.x * falloff);
          p.heat = max(p.heat, b.y * falloff);
        } else if (kind == OP_DAMP) {
          let w = b.x * falloff;
          p.vx = tame(p.vx * (1.0 - w));
          p.vy = tame(p.vy * (1.0 - w));
        } else if (kind == OP_GRIP) {
          let w = b.x * falloff;
          p.x = mix(p.x, cx, w);
          p.y = mix(p.y, cy, w);
          p.vx = tame(p.vx * (1.0 - w));
          p.vy = tame(p.vy * (1.0 - w));
        }
      }
    }
  }
  return p;
}

// ------------------------------------------------------------------- step

var<workgroup> wg_alive: atomic<u32>;

@compute @workgroup_size(${STEP_WORKGROUP})
fn step_main(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
  if (lid == 0u) { atomicStore(&wg_alive, 0u); }
  workgroupBarrier();

  let i = gid.x;
  let aliveCount = S.counts.x;
  var alive = 0u;

  if (i < aliveCount) {
    var p = apply_ops(i, P[i]);

    let dt = S.rates.x;
    let maxx = S.grid.x;
    let maxy = S.grid.y;
    let drag = S.misc.w;
    let life = p.life - dt;
    var committed = false;

    if (life > 0.0) {
      // advance(): RK2 midpoint through the fluid plus own velocity.
      let f1 = sample_uv(p.x, p.y);
      var ox = p.vx * S.rates.y;
      var oy = p.vy * S.rates.y;
      let swirl = S.misc.y;
      if (swirl != 0.0) {
        let tvx = drag * f1.x + ox;
        let tvy = drag * f1.y + oy;
        let len2 = tvx * tvx + tvy * tvy;
        if (len2 > 1e-12) {
          let k = curl_at(p.x, p.y) * swirl / sqrt(len2);
          ox -= tvy * k;
          oy += tvx * k;
        }
      }
      oy += S.misc.z;
      ox = tame(ox);
      oy = tame(oy);
      let hx = p.x + 0.5 * dt * (drag * f1.x + ox);
      let hy = p.y + 0.5 * dt * (drag * f1.y + oy);
      let f2 = sample_uv(hx, hy);
      let nx = p.x + dt * (drag * f2.x + ox);
      let ny = p.y + dt * (drag * f2.y + oy);
      let ok = is_finite(nx) && is_finite(ny)
        && nx >= 0.0 && nx <= maxx && ny >= 0.0 && ny <= maxy
        && !is_solid(nx, ny);

      if (ok) {
        let speed = length(vec2<f32>((nx - p.x) * S.misc.x, (ny - p.y) * S.misc.x));
        let floor_heat = min(speed * (1.0 / HEAT_SPEED_FULL), 1.0);
        p.x = nx;
        p.y = ny;
        p.vx = ox;
        p.vy = oy;
        let cooled = p.heat * S.rates.z;
        if (floor_heat > cooled) {
          p.heat = cooled + (floor_heat - cooled) * S.rates.w;
        } else {
          p.heat = cooled;
        }
        p.life = life;
        committed = true;
      }
    }

    if (!committed) {
      // Recycle: spend a unit of respawn credit if any is left this frame.
      if (atomicAdd(&C.respawns, 1u) < S.counts.w) {
        var r = rng_new(i, 0xC0FFEEu);
        var x = 0.0;
        var y = 0.0;
        var free = false;
        for (var t = 0u; t < RESPAWN_TRIES; t++) {
          x = rnd_range(&r, 0.0, maxx);
          y = rnd_range(&r, 0.0, maxy);
          if (!is_solid(x, y)) { free = true; break; }
        }
        p.x = x;
        p.y = y;
        p.vx = 0.0;
        p.vy = 0.0;
        p.heat = 0.0;
        p.max_life = S.life_ob.x * rnd_range(&r, LIFE_JITTER_LO, LIFE_JITTER_HI);
        p.life = select(0.0, p.max_life, free);
        committed = free;
      } else {
        p.life = 0.0;
        p.vx = 0.0;
        p.vy = 0.0;
        p.heat = 0.0;
        p.x = select(0.0, clamp(p.x, 0.0, maxx), is_finite(p.x));
        p.y = select(0.0, clamp(p.y, 0.0, maxy), is_finite(p.y));
      }
    }

    P[i] = p;
    alive = select(0u, 1u, committed);
  }

  atomicAdd(&wg_alive, alive);
  workgroupBarrier();
  if (lid == 0u) {
    atomicAdd(&C.alive, atomicLoad(&wg_alive));
  }
}

/** Particles::seed_uniform: scatter every active particle with a random life phase. */
@compute @workgroup_size(${STEP_WORKGROUP})
fn seed_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= S.counts.x) { return; }
  var r = rng_new(i, 0x5EEDu);
  var p: Particle;
  p.x = rnd_range(&r, 0.0, S.grid.x);
  p.y = rnd_range(&r, 0.0, S.grid.y);
  p.vx = 0.0;
  p.vy = 0.0;
  p.max_life = S.life_ob.x * rnd_range(&r, LIFE_JITTER_LO, LIFE_JITTER_HI);
  p.life = p.max_life * rnd(&r);
  p.heat = 0.0;
  p.pad = 0.0;
  P[i] = p;
}`;
