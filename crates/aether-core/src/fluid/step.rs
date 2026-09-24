//! The solver's frame: the stage order in [`Fluid::step`], plus the boundary
//! work it is the only caller of.
//!
//! Split out of `mod.rs` so the struct, its constants and its accessors read as
//! the module's contract while the per-frame order lives on its own. That order
//! is load-bearing and commented inline — do not reorder it for tidiness.

use crate::config::Params;
use crate::math::decay;

use super::{sane_dt, Fluid, MAX_CELL_SPEED, VISCOSITY_EPSILON};

impl Fluid {
    /// Advances the simulation by `dt` seconds.
    ///
    /// External forces and dye are expected to have been injected already (the
    /// spell layer runs first), so this is the pure solver: advect, diffuse,
    /// confine, project, dissipate.
    pub fn step(&mut self, dt: f32, params: &Params) {
        let dt = sane_dt(dt);

        // The wall mask was refreshed by set_obstacle (or at construction).
        // A silhouette cell that only just turned solid still holds the
        // velocity it had as fluid, and every stage below assumes otherwise.
        self.enforce_boundaries();

        self.advect(dt);

        if params.viscosity > VISCOSITY_EPSILON {
            self.diffuse(dt, params.viscosity);
        }

        // Recomputed every step even when confinement is off, so that the
        // `curl` / `curl_at` accessors are never a frame stale. (The particle
        // system does *not* read them — it re-derives curl from four velocity
        // loads around the particle, which is cheaper than an interpolated
        // fetch — so this pass is one full grid sweep for the HUD and for
        // whatever reads the accessors next.)
        self.compute_curl();
        if params.vorticity > 0.0 {
            self.confine_vorticity(dt, params.vorticity);
        }

        // Nothing else bounds the field: every gesture adds momentum, and with a
        // gentle dissipation a person standing in front of the camera drives it
        // up until dye is advected far further than a cell per step and the
        // frame reads as one saturated smear. The ceiling is well above the
        // speeds a single push produces, so it only catches that runaway — and
        // it goes *before* the projection, whose whole job is to remove the
        // divergence a per-cell rescale leaves behind.
        self.vel.clamp_speed(MAX_CELL_SPEED);

        self.close_solid_faces();
        self.project(params.pressure_iters);

        // A uniform scale cannot introduce divergence, so dissipation is safe
        // to apply after the projection rather than before it.
        self.vel.scale(decay(params.velocity_dissipation, dt));
        let dye_decay = decay(params.dye_dissipation, dt);
        for channel in &mut self.dye {
            channel.scale(dye_decay);
        }
        self.fade_dye_at_body(dt);

        self.last_divergence = self.max_divergence();
    }

    // -------------------------------------------------------------- boundary

    /// Rebuilds the solver's wall mask from the obstacle field.
    pub(super) fn refresh_solid(&mut self) {
        let (w, h) = (self.w, self.h);
        self.solid.data.copy_from_slice(&self.obstacle.data);
        self.has_obstacle = self.obstacle.data.iter().any(|&o| o >= 0.5);
        for x in 0..w {
            self.solid.data[x] = 1.0;
            self.solid.data[(h - 1) * w + x] = 1.0;
        }
        for y in 0..h {
            self.solid.data[y * w] = 1.0;
            self.solid.data[y * w + w - 1] = 1.0;
        }
    }

    /// Zeroes velocity inside obstacles and applies no-slip at the border.
    /// Shared by the solver stages, so it lives here rather than inline.
    pub(crate) fn enforce_boundaries(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                if self.solid.data[i] >= 0.5 {
                    self.vel.u.data[i] = 0.0;
                    self.vel.v.data[i] = 0.0;
                }
            }
        }
    }
}
