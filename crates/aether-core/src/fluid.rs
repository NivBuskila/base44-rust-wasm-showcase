//! Incompressible 2-D fluid solver (Stam, *Stable Fluids*, SIGGRAPH 1999) with
//! vorticity confinement and interior obstacles.
//!
//! ## Contract
//!
//! One [`Fluid::step`] performs, in order:
//! 1. **Advection** — semi-Lagrangian backtrace of velocity and of the three
//!    dye channels through the velocity field.
//! 2. **Viscous diffusion** — Jacobi relaxation, skipped when viscosity is 0.
//! 3. **Vorticity confinement** — re-injects the curl that numerical
//!    dissipation eats, which is what makes the smoke look alive.
//! 4. **Projection** — divergence, Jacobi pressure solve, gradient subtract,
//!    leaving the field (numerically) divergence free.
//! 5. **Dissipation** — frame-rate independent exponential decay of velocity
//!    and dye.
//!
//! Obstacle cells (`obstacle >= 0.5`) must end the step with zero velocity, and
//! the domain border is a no-slip wall. Post-step the field must satisfy
//! `max |div u| << max |u|` in the fluid interior.

use crate::config::Params;
use crate::field::{Grid, VecField};
use crate::math::decay;

/// Dye channel count (RGB).
pub const DYE_CHANNELS: usize = 3;

/// A velocity + dye field on a fixed grid.
pub struct Fluid {
    w: usize,
    h: usize,
    vel: VecField,
    vel_tmp: VecField,
    dye: [Grid; DYE_CHANNELS],
    dye_tmp: [Grid; DYE_CHANNELS],
    pressure: Grid,
    pressure_tmp: Grid,
    divergence: Grid,
    curl: Grid,
    obstacle: Grid,
    scratch: Grid,
    /// Absolute maximum divergence measured at the end of the last step; the
    /// HUD shows it and the tests assert on it.
    last_divergence: f32,
}

impl Fluid {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            vel: VecField::new(w, h),
            vel_tmp: VecField::new(w, h),
            dye: [Grid::new(w, h), Grid::new(w, h), Grid::new(w, h)],
            dye_tmp: [Grid::new(w, h), Grid::new(w, h), Grid::new(w, h)],
            pressure: Grid::new(w, h),
            pressure_tmp: Grid::new(w, h),
            divergence: Grid::new(w, h),
            curl: Grid::new(w, h),
            obstacle: Grid::new(w, h),
            scratch: Grid::new(w, h),
            last_divergence: 0.0,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    /// Adds a velocity impulse, in grid units per second, around `(x, y)`.
    pub fn add_force(&mut self, x: f32, y: f32, fx: f32, fy: f32, radius: f32) {
        self.vel.splat(x, y, radius, fx, fy);
    }

    /// Injects coloured dye around `(x, y)`.
    pub fn add_dye(&mut self, x: f32, y: f32, rgb: [f32; 3], radius: f32) {
        for c in 0..DYE_CHANNELS {
            self.dye[c].splat(x, y, radius, rgb[c]);
        }
    }

    /// Replaces the obstacle field. Values `>= 0.5` are solid.
    pub fn set_obstacle(&mut self, mask: &Grid) {
        self.obstacle.resample_from(mask);
    }

    #[inline]
    pub fn obstacle(&self) -> &Grid {
        &self.obstacle
    }

    #[inline]
    pub fn velocity(&self) -> &VecField {
        &self.vel
    }

    #[inline]
    pub fn velocity_mut(&mut self) -> &mut VecField {
        &mut self.vel
    }

    #[inline]
    pub fn dye(&self) -> &[Grid; DYE_CHANNELS] {
        &self.dye
    }

    #[inline]
    pub fn curl(&self) -> &Grid {
        &self.curl
    }

    /// Curl (vorticity) sampled at a grid-space point; drives particle spin.
    pub fn curl_at(&self, x: f32, y: f32) -> f32 {
        self.curl.sample(x, y)
    }

    pub fn energy(&self) -> f32 {
        self.vel.energy()
    }

    pub fn max_speed(&self) -> f32 {
        self.vel.max_speed()
    }

    /// Absolute maximum divergence left over after the last projection.
    pub fn last_divergence(&self) -> f32 {
        self.last_divergence
    }

    pub fn reset(&mut self) {
        self.vel.zero();
        self.pressure.zero();
        for c in 0..DYE_CHANNELS {
            self.dye[c].zero();
        }
        self.curl.zero();
        self.last_divergence = 0.0;
    }

    /// Advances the simulation by `dt` seconds.
    ///
    /// AGENT-IMPLEMENT: replace this placeholder with the full solver described
    /// in the module docs. The placeholder only dissipates so the app runs
    /// end-to-end while the real solver is being written.
    pub fn step(&mut self, dt: f32, params: &Params) {
        let dt = dt.clamp(1.0 / 480.0, 1.0 / 20.0);
        self.vel.scale(decay(params.velocity_dissipation, dt));
        for c in 0..DYE_CHANNELS {
            self.dye[c].scale(decay(params.dye_dissipation, dt));
        }
        let _ = (&mut self.vel_tmp, &mut self.dye_tmp, &mut self.pressure_tmp);
        let _ = (&mut self.divergence, &mut self.scratch);
        self.enforce_boundaries();
        self.last_divergence = 0.0;
    }

    /// Zeroes velocity inside obstacles and applies no-slip at the border.
    /// Shared by the solver stages, so it lives here rather than inline.
    pub(crate) fn enforce_boundaries(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let solid = self.obstacle.data[i] >= 0.5;
                let border = x == 0 || y == 0 || x == self.w - 1 || y == self.h - 1;
                if solid || border {
                    self.vel.u.data[i] = 0.0;
                    self.vel.v.data[i] = 0.0;
                }
            }
        }
    }

    /// Clears any non-finite cell. Returns the number of cells repaired.
    pub fn sanitize(&mut self) -> usize {
        let mut fixed = self.vel.sanitize() + self.pressure.sanitize();
        for c in 0..DYE_CHANNELS {
            fixed += self.dye[c].sanitize();
        }
        fixed
    }
}
