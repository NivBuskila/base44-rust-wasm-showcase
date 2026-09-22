//! The resolved per-step [`Frame`] and the RK2 [`Advance`] of one particle.
//!
//! Split from the lane itself because these two are pure: `advance` reads the
//! shared, immutable frame and returns a candidate state, which is what lets a
//! lane run two of them back to back and lets the core interleave them.

use crate::particles::*;

/// Everything one `step` needs that is not per-particle, resolved once: the
/// fields, the bounds, and every tunable already folded through `dt`. Bundling
/// them keeps the per-particle helpers pure and short-signatured, and keeps the
/// clamping and `decay` calls out of the loop.
pub(in crate::particles) struct Frame<'a> {
    pub(in crate::particles) vel: &'a VecField,
    /// `None` when there is no usable obstacle field, which skips the test
    /// entirely for the common no-body case.
    pub(in crate::particles) obstacle: Option<&'a Grid>,
    /// Grid-space -> obstacle-index scale, for the case where the obstacle
    /// field is not the same resolution as the velocity field.
    pub(in crate::particles) osx: f32,
    pub(in crate::particles) osy: f32,
    /// Inclusive upper bounds of the velocity grid, `(w - 1, h - 1)`.
    pub(in crate::particles) maxx: f32,
    pub(in crate::particles) maxy: f32,
    /// Base lifetime for respawns, before jitter.
    pub(in crate::particles) life: f32,
    pub(in crate::particles) dt: f32,
    /// `params.particle_drag`, clamped: the multiplier on the fluid velocity.
    pub(in crate::particles) drag: f32,
    /// Fraction of a particle's own velocity surviving this step.
    pub(in crate::particles) keep: f32,
    /// Fraction of `heat` surviving this step.
    pub(in crate::particles) heat_keep: f32,
    /// Fraction of the gap to the speed-derived heat floor closed this step.
    pub(in crate::particles) heat_rise: f32,
    /// `1 / dt`, for turning this step's displacement into a speed.
    pub(in crate::particles) inv_dt: f32,
    /// `curl_influence * dt`.
    pub(in crate::particles) swirl: f32,
    /// `gravity * dt`.
    pub(in crate::particles) gravity: f32,
}

/// One particle's candidate next state.
#[derive(Clone, Copy)]
pub(in crate::particles) struct Advance {
    pub(in crate::particles) nx: f32,
    pub(in crate::particles) ny: f32,
    pub(in crate::particles) ox: f32,
    pub(in crate::particles) oy: f32,
    /// False when the particle left the grid, landed in an obstacle, or went
    /// non-finite — all of which mean "recycle it".
    pub(in crate::particles) ok: bool,
}

impl Advance {
    /// The state of a particle that is not worth integrating at all.
    pub(in crate::particles) const RECYCLE: Self = Self {
        nx: 0.0,
        ny: 0.0,
        ox: 0.0,
        oy: 0.0,
        ok: false,
    };
}

impl Frame<'_> {
    /// Integrates one particle for a step: RK2 midpoint advection through the
    /// fluid, plus the particle's own velocity carrying gravity, curl swirl and
    /// whatever gestures kicked into it.
    ///
    /// Pure, so the caller can run two of these back to back and let the core
    /// interleave them.
    #[inline(always)]
    pub(in crate::particles) fn advance(&self, px: f32, py: f32, ovx: f32, ovy: f32) -> Advance {
        let vel = self.vel;
        let (fu1, fv1) = sample_uv(vel, px, py);

        let mut ox = ovx * self.keep;
        let mut oy = ovy * self.keep;

        // Swirl: perpendicular to the direction of travel, signed by the local
        // curl, so a particle entering a vortex is bent onto a spiral instead
        // of crossing it in a straight line. Scaled off the *unit* travel
        // direction, which keeps the magnitude proportional to omega alone —
        // taken off the unscaled velocity instead, fast particles spin and slow
        // ones (the ones actually trapped in the vortex) barely turn at all.
        if self.swirl != 0.0 {
            let tvx = self.drag * fu1 + ox;
            let tvy = self.drag * fv1 + oy;
            let len2 = tvx * tvx + tvy * tvy;
            // Same guard as `math::normalize` (len > 1e-6), but folding the
            // normalisation into the scale keeps one division off the critical
            // path instead of two.
            if len2 > 1e-12 {
                let k = curl_at(vel, px, py) * self.swirl / len2.sqrt();
                ox -= tvy * k;
                oy += tvx * k;
            }
        }
        oy += self.gravity;
        ox = tame(ox);
        oy = tame(oy);

        // RK2 midpoint: sample, step half, sample again, apply. The own
        // velocity is constant across the step so it needs no correction; only
        // the field sample does, and it is exactly the correction that keeps a
        // fast curved gesture from cutting the corner.
        let dt = self.dt;
        let hx = px + 0.5 * dt * (self.drag * fu1 + ox);
        let hy = py + 0.5 * dt * (self.drag * fv1 + oy);
        let (fu2, fv2) = sample_uv(vel, hx, hy);
        let nx = px + dt * (self.drag * fu2 + ox);
        let ny = py + dt * (self.drag * fv2 + oy);

        let ok = nx.is_finite()
            && ny.is_finite()
            && nx >= 0.0
            && nx <= self.maxx
            && ny >= 0.0
            && ny <= self.maxy
            && !self.is_solid(nx, ny);
        Advance { nx, ny, ox, oy, ok }
    }

    /// Whether grid-space `(x, y)` falls in an obstacle cell.
    ///
    /// Nearest-cell rather than bilinear: the obstacle is a hard binary
    /// predicate, the caller only needs to know which side of 0.5 it is on,
    /// and this is one load instead of four in a loop that runs 220k times a
    /// frame.
    #[inline(always)]
    pub(in crate::particles) fn is_solid(&self, x: f32, y: f32) -> bool {
        match self.obstacle {
            None => false,
            Some(o) => {
                // `as usize` saturates: NaN and negatives land on 0, huge
                // values on usize::MAX, and `min` brings both into range.
                let xi = ((x * self.osx + 0.5) as usize).min(o.w - 1);
                let yi = ((y * self.osy + 0.5) as usize).min(o.h - 1);
                o.data[yi * o.w + xi] >= 0.5
            }
        }
    }
}
