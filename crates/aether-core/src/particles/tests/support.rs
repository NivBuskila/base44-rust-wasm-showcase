//! Fixtures shared by every group: constant and rigid-rotation fields, the parameter/config builders and the obstacle frame.

pub(super) use crate::particles::*;

pub(super) const DT: f32 = 1.0 / 60.0;

pub(super) fn const_field(w: usize, h: usize, u: f32, v: f32) -> VecField {
    let mut f = VecField::new(w, h);
    f.u.data.fill(u);
    f.v.data.fill(v);
    f
}

/// Rigid rotation about `(cx, cy)`: `u = -omega * (y - cy)`,
/// `v = omega * (x - cx)`. Each component is linear, so bilinear sampling
/// reproduces it exactly and the analytic circle is the exact solution.
pub(super) fn rot_field(w: usize, h: usize, cx: f32, cy: f32, omega: f32) -> VecField {
    let mut f = VecField::new(w, h);
    for j in 0..h {
        for i in 0..w {
            f.u.set(i, j, -omega * (j as f32 - cy));
            f.v.set(i, j, omega * (i as f32 - cx));
        }
    }
    f
}

pub(super) fn params(drag: f32, life: f32, spawn: f32) -> Params {
    Params {
        particle_drag: drag,
        particle_life: life,
        spawn_rate: spawn,
        ..Params::default()
    }
}

pub(super) fn cfg(curl: f32, gravity: f32, damping: f32) -> ParticleConfig {
    ParticleConfig {
        curl_influence: curl,
        gravity,
        damping,
    }
}

/// Advection only: no swirl, no gravity, immortal particles.
pub(super) fn pure_advection() -> (Params, ParticleConfig) {
    (params(1.0, 10.0, 0.0), cfg(0.0, 0.0, 0.6))
}

/// A `Frame` that does nothing but look up obstacles, for unit-testing the
/// cull predicate directly.
pub(super) fn obstacle_frame<'a>(vel: &'a VecField, obs: &'a Grid) -> Frame<'a> {
    Frame {
        vel,
        obstacle: Some(obs),
        osx: (obs.w - 1) as f32 / (vel.w() - 1) as f32,
        osy: (obs.h - 1) as f32 / (vel.h() - 1) as f32,
        maxx: (vel.w() - 1) as f32,
        maxy: (vel.h() - 1) as f32,
        life: 1.0,
        dt: DT,
        drag: 1.0,
        keep: 1.0,
        heat_keep: 1.0,
        heat_rise: 0.0,
        inv_dt: 1.0 / DT,
        swirl: 0.0,
        gravity: 0.0,
    }
}
