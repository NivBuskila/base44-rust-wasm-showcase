//! Structure-of-arrays particle system advected by the fluid velocity field.
//!
//! SoA rather than array-of-structs so each `step` pass strides linearly over
//! one `Vec<f32>`, which is both cache friendly and auto-vectorisable.
//!
//! ## Contract
//!
//! `step` must, for each of the first `active` particles:
//! - advect by the fluid velocity (RK2 midpoint; plain Euler visibly smears at
//!   the speeds hand gestures produce),
//! - add a curl-driven tangential swirl scaled by `curl_influence`,
//! - age by `dt` and respawn from the edges when `life` runs out or when the
//!   particle lands inside an obstacle or leaves the grid,
//! - keep `heat` as the 0..1 colour-ramp coordinate, decaying toward 0.
//!
//! Positions are in **grid space**; `build_render_buffer` converts to the
//! normalised `[0, 1]` coordinates the vertex shader wants.

use crate::config::{Params, PARTICLE_STRIDE};
use crate::field::{Grid, VecField};
use crate::rng::Rng;

/// Per-step tunables that do not belong in the global [`Params`].
#[derive(Clone, Copy, Debug)]
pub struct ParticleConfig {
    /// Tangential swirl from the fluid's curl.
    pub curl_influence: f32,
    /// Constant downward acceleration in grid units per second squared.
    pub gravity: f32,
    /// Fraction of velocity kept per second when outside the fluid's influence.
    pub damping: f32,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            curl_influence: 2.2,
            gravity: 0.0,
            damping: 0.6,
        }
    }
}

/// A fixed-capacity particle pool.
pub struct Particles {
    capacity: usize,
    active: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    life: Vec<f32>,
    max_life: Vec<f32>,
    heat: Vec<f32>,
    render: Vec<f32>,
    rng: Rng,
    /// Round-robin cursor so bursts overwrite the oldest slots.
    cursor: usize,
    alive: usize,
}

impl Particles {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            active: 0,
            x: vec![0.0; capacity],
            y: vec![0.0; capacity],
            vx: vec![0.0; capacity],
            vy: vec![0.0; capacity],
            life: vec![0.0; capacity],
            max_life: vec![1.0; capacity],
            heat: vec![0.0; capacity],
            render: vec![0.0; capacity * PARTICLE_STRIDE],
            rng: Rng::new(seed),
            cursor: 0,
            alive: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Number of particles with `life > 0`, recomputed each `step`.
    #[inline]
    pub fn alive(&self) -> usize {
        self.alive
    }

    pub fn set_active(&mut self, n: usize) {
        self.active = n.min(self.capacity);
    }

    /// Scatters every active particle uniformly over a `w * h` grid with a
    /// randomised life phase, so the pool does not die in lockstep.
    pub fn seed_uniform(&mut self, w: usize, h: usize, life: f32) {
        for i in 0..self.active {
            self.x[i] = self.rng.range(0.0, (w - 1) as f32);
            self.y[i] = self.rng.range(0.0, (h - 1) as f32);
            self.vx[i] = 0.0;
            self.vy[i] = 0.0;
            self.max_life[i] = life * self.rng.range(0.6, 1.4);
            self.life[i] = self.max_life[i] * self.rng.next_f32();
            self.heat[i] = 0.0;
        }
        self.alive = self.active;
    }

    /// Emits `count` particles from `(x, y)` in grid space with radial speed
    /// `speed` and the given colour-ramp coordinate.
    ///
    /// AGENT-IMPLEMENT: the placeholder writes position but no velocity.
    pub fn spawn_burst(&mut self, x: f32, y: f32, count: usize, speed: f32, heat: f32, life: f32) {
        if self.active == 0 {
            return;
        }
        let _ = speed;
        for _ in 0..count.min(self.active) {
            let i = self.cursor % self.active;
            self.cursor = self.cursor.wrapping_add(1);
            self.x[i] = x;
            self.y[i] = y;
            self.max_life[i] = life.max(0.05);
            self.life[i] = self.max_life[i];
            self.heat[i] = heat;
        }
    }

    /// Advances every active particle.
    ///
    /// AGENT-IMPLEMENT: the placeholder only ages particles.
    pub fn step(
        &mut self,
        vel: &VecField,
        obstacle: &Grid,
        dt: f32,
        params: &Params,
        cfg: &ParticleConfig,
    ) {
        let _ = (vel, obstacle, cfg, params);
        let mut alive = 0;
        for i in 0..self.active {
            self.life[i] -= dt;
            if self.life[i] > 0.0 {
                alive += 1;
            }
        }
        self.alive = alive;
    }

    /// Packs `x, y, heat, life` per particle, with position normalised to
    /// `[0, 1]` against the grid the simulation runs on and `life` normalised
    /// against that particle's own lifetime.
    pub fn build_render_buffer(&mut self, grid_w: usize, grid_h: usize) {
        let sx = 1.0 / (grid_w - 1) as f32;
        let sy = 1.0 / (grid_h - 1) as f32;
        for i in 0..self.active {
            let o = i * PARTICLE_STRIDE;
            let norm_life = if self.max_life[i] > 0.0 {
                (self.life[i] / self.max_life[i]).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.render[o] = self.x[i] * sx;
            self.render[o + 1] = self.y[i] * sy;
            self.render[o + 2] = self.heat[i];
            self.render[o + 3] = norm_life;
        }
    }

    /// The interleaved buffer, truncated to the active particle count.
    #[inline]
    pub fn render_buffer(&self) -> &[f32] {
        &self.render[..self.active * PARTICLE_STRIDE]
    }

    /// Raw pointer to the render buffer, for zero-copy access from JS.
    #[inline]
    pub fn render_ptr(&self) -> *const f32 {
        self.render.as_ptr()
    }

    // --- Test and spell accessors ---

    #[inline]
    pub fn position(&self, i: usize) -> (f32, f32) {
        (self.x[i], self.y[i])
    }

    #[inline]
    pub fn velocity(&self, i: usize) -> (f32, f32) {
        (self.vx[i], self.vy[i])
    }

    #[inline]
    pub fn life_of(&self, i: usize) -> f32 {
        self.life[i]
    }

    #[inline]
    pub fn heat_of(&self, i: usize) -> f32 {
        self.heat[i]
    }

    /// Places a particle explicitly. Tests use this to build exact scenarios.
    pub fn place(&mut self, i: usize, x: f32, y: f32, vx: f32, vy: f32, life: f32) {
        self.x[i] = x;
        self.y[i] = y;
        self.vx[i] = vx;
        self.vy[i] = vy;
        self.max_life[i] = life.max(1e-4);
        self.life[i] = life;
    }

    /// Adds a radial impulse to every particle within `radius` of `(cx, cy)`.
    ///
    /// AGENT-IMPLEMENT (spells rely on this).
    pub fn impulse(&mut self, cx: f32, cy: f32, radius: f32, strength: f32, heat: f32) {
        let _ = (cx, cy, radius, strength, heat);
    }
}
