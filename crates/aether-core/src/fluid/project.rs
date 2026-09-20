//! Pressure projection: divergence, Jacobi pressure solve with solid faces
//! closed, gradient subtraction, and the residual the HUD reports.

use super::*;

impl Fluid {
    pub(super) fn project(&mut self, iters: usize) {
        self.compute_divergence();
        let (w, h) = (self.w, self.h);

        // `self.pressure` still holds last frame's solution and the field
        // barely changes between frames, so Jacobi starts far closer to the
        // answer than a zeroed guess: same iteration count, visibly lower
        // residual divergence.
        for _ in 0..iters.clamp(1, 200) {
            jacobi_pressure(
                &self.pressure.data,
                &mut self.pressure_tmp.data,
                &self.divergence.data,
                &self.solid.data,
                w,
                h,
            );
            core::mem::swap(&mut self.pressure.data, &mut self.pressure_tmp.data);
        }

        self.recentre_pressure();
        self.subtract_pressure_gradient();
    }

    /// Kills the flux through every face that touches a wall.
    ///
    /// A face between fluid and solid is not a degree of freedom: the pressure
    /// solve treats it as Neumann and so can never correct it, which means an
    /// uncorrected inflow there is mass vanishing *into* the body — the leak
    /// this whole boundary treatment exists to prevent. Only the component
    /// normal to the face is removed, so fluid still slides along a silhouette.
    pub(super) fn close_solid_faces(&mut self) {
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w && self.solid.data[i + 1] >= 0.5 {
                    self.vel.u.data[i] = 0.0;
                }
                if i + w < w * h && self.solid.data[i + w] >= 0.5 {
                    self.vel.v.data[i] = 0.0;
                }
            }
        }
    }

    // `x` also builds the flat index `i`; an iterator over `row` would hide that.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn compute_divergence(&mut self) {
        let (w, h) = (self.w, self.h);
        let mut div = core::mem::take(&mut self.divergence.data);
        div.fill(0.0);
        let this: &Self = self;
        interior_rows(&mut div, w, h, |y, row| {
            for x in 1..w - 1 {
                let i = y * w + x;
                if this.solid.data[i] >= 0.5 {
                    continue;
                }
                // Faces touching a wall carry no flux and have already been
                // zeroed, so the no-penetration condition needs no branch here.
                let du = this.vel.u.data[i] - this.vel.u.data[i - 1];
                let dv = this.vel.v.data[i] - this.vel.v.data[i - w];
                row[x] = du + dv;
            }
        });
        self.divergence.data = div;
    }

    /// Removes the constant component of the pressure.
    ///
    /// An all-Neumann Poisson problem only fixes pressure up to a constant,
    /// and every Jacobi sweep shifts that constant by a multiple of
    /// `mean(div)`, which is never exactly zero in floating point. Warm
    /// starting feeds the shift back in each frame, so over a long session the
    /// field drifts without bound — invisible in the velocity, since the
    /// gradient of a constant is zero, right up until it overflows.
    pub(super) fn recentre_pressure(&mut self) {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (p, s) in self.pressure.data.iter().zip(&self.solid.data) {
            if *s < 0.5 {
                sum += *p as f64;
                count += 1;
            }
        }
        if count == 0 {
            return;
        }
        let mean = (sum / count as f64) as f32;
        if !mean.is_finite() {
            self.pressure.zero();
            return;
        }
        for (p, s) in self.pressure.data.iter_mut().zip(&self.solid.data) {
            if *s < 0.5 {
                *p -= mean;
            } else {
                *p = 0.0;
            }
        }
    }

    pub(super) fn subtract_pressure_gradient(&mut self) {
        let (w, h) = (self.w, self.h);
        let mut u = core::mem::take(&mut self.vel.u.data);
        let mut v = core::mem::take(&mut self.vel.v.data);
        let this: &Self = self;
        interior_row_pairs(&mut u, &mut v, w, h, |y, ur, vr| {
            for x in 1..w - 1 {
                let i = y * w + x;
                if this.solid.data[i] >= 0.5 {
                    continue;
                }
                let pc = this.pressure.data[i];
                // Forward difference, matching the backward-difference
                // divergence: `wall_pick` makes the gradient vanish across a
                // closed face, leaving that face's zero flux untouched.
                let r = wall_pick(this.pressure.data[i + 1], this.solid.data[i + 1], pc);
                let d = wall_pick(this.pressure.data[i + w], this.solid.data[i + w], pc);
                ur[x] -= r - pc;
                vr[x] -= d - pc;
            }
        });
        self.vel.u.data = u;
        self.vel.v.data = v;
    }

    /// `max |div u|` over the non-solid interior — the residual the projection
    /// failed to remove.
    pub(super) fn max_divergence(&self) -> f32 {
        let (w, h) = (self.w, self.h);
        let mut worst = 0.0f32;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                let du = self.vel.u.data[i] - self.vel.u.data[i - 1];
                let dv = self.vel.v.data[i] - self.vel.v.data[i - w];
                // `max` drops NaN rather than latching it, so a poisoned field
                // still reports a usable number to the HUD.
                worst = worst.max((du + dv).abs());
            }
        }
        worst
    }
}
