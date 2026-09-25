//! Semi-Lagrangian advection: one shared bilinear trace map carries the
//! velocity and the three dye channels, and the dye is faded where it meets the
//! body.

mod reach;
mod sampling;

pub(in crate::fluid) use reach::Reach;
pub(in crate::fluid) use sampling::{corner_of, fetch, Advector, Stencil};

use super::*;

impl Fluid {
    pub(super) fn advect(&mut self, dt: f32) {
        self.build_traces(dt);
        let (w, h) = (self.w, self.h);

        self.advector.run(&mut self.vel.u.data, w, h);
        self.advector.run(&mut self.vel.v.data, w, h);
        self.enforce_boundaries();

        for channel in &mut self.dye {
            self.advector.run(&mut channel.data, w, h);
        }
    }

    /// Fills the backward and forward trace maps for this step.
    ///
    /// Materialised rather than recomputed because velocity and all three dye
    /// channels trace identically: five fetches per cell share one RK2 trace,
    /// one obstacle walk and one clamp-floor-fraction conversion.
    pub(super) fn build_traces(&mut self, dt: f32) {
        let (w, h) = (self.w, self.h);
        // With a body in frame every trace endpoint is tested against it, and
        // many cells are nowhere near it. `reach` marks the cells some wall is
        // close enough for a trace this step to touch; the rest skip the test,
        // whose answer for them is already known to be "clear".
        let skip_far = self.has_obstacle
            && match self.trace_reach(dt) {
                Some(radius) => {
                    self.reach.build(&self.obstacle.data, w, h, radius);
                    true
                }
                None => false,
            };
        // The trace maps are written while the rest of `self` is read through
        // `integrate`, so they are taken out for the duration: `Advector::new(0)`
        // allocates nothing.
        let mut advector = core::mem::replace(&mut self.advector, Advector::new(0));
        let this: &Self = self;
        let rows = h.min(advector.back.cells.len() / w.max(1));
        let mut lanes: Vec<_> = advector
            .back
            .cells
            .chunks_mut(w)
            .zip(advector.fwd.cells.chunks_mut(w))
            .take(rows)
            .collect();
        // Rows are placed by the band length asked for, never by `lanes.len()`:
        // the last band is usually shorter, and its own length would drop it
        // over rows an earlier band owns, leaving the bottom rows untraced.
        let per = par::chunk_len(rows, 4);
        par::chunks_mut(&mut lanes, per, |band, lanes| {
            for (k, (back, fwd)) in lanes.iter_mut().enumerate() {
                let y = band * per + k;
                let fy = y as f32;
                for x in 0..w {
                    let i = y * w + x;
                    let fx = x as f32;
                    let u0 = this.vel.u.data[i];
                    let v0 = this.vel.v.data[i];

                    // Midpoint (RK2) rather than Euler: at the 200+ cells/s a
                    // swipe produces, first-order backtracing rounds the dye
                    // off into mush inside a second and no number of pressure
                    // iterations brings the structure back.
                    let near = !skip_far || this.reach.near(i);
                    let (bx, by) = this.integrate(fx, fy, -dt, u0, v0);
                    back[x] = this.clear_stencil(fx, fy, bx, by, near);

                    let (gx, gy) = this.integrate(fx, fy, dt, u0, v0);
                    fwd[x] = this.clear_stencil(fx, fy, gx, gy, near);
                }
            }
        });
        drop(lanes);
        self.advector = advector;
    }

    /// The backward and forward trace maps the last `build_traces` produced.
    #[cfg(test)]
    pub(in crate::fluid) fn trace_stencils(&self) -> (&[Stencil], &[Stencil]) {
        (&self.advector.back.cells, &self.advector.fwd.cells)
    }

    /// Chebyshev radius, in cells, within which every stencil corner of every
    /// trace this step must fall; `None` when the field holds a non-finite
    /// value and no bound can be promised.
    ///
    /// A trace moves `hs * um` per substep with `|hs|` summing to `|dt|`, and a
    /// bilinear fetch never exceeds the largest component it blends, so each
    /// axis of the endpoint lies within `|dt| * max|component|` of the start.
    /// The stencil adds one cell for its `+1` corner; one more absorbs rounding.
    pub(super) fn trace_reach(&self, dt: f32) -> Option<usize> {
        let mut largest = 0.0f32;
        for &c in self.vel.u.data.iter().chain(&self.vel.v.data) {
            if !c.is_finite() {
                return None;
            }
            largest = largest.max(c.abs());
        }
        let span = dt.abs() * largest;
        if !span.is_finite() {
            return None;
        }
        // The cast saturates, so a huge finite speed still yields a radius
        // covering the whole grid rather than wrapping round to a small one.
        Some((span.ceil() as usize).saturating_add(2))
    }

    /// [`Stencil::at`] for a trace from `(x0, y0)` to `(tx, ty)`, backed out
    /// of the walls as [`Fluid::trace_to_fluid`] does.
    ///
    /// `near == false` promises no wall is within the trace's reach, so the
    /// obstacle test is a known miss and is skipped. Otherwise the stencil the
    /// endpoint needs anyway doubles as the test, instead of resolving the same
    /// clamp-floor-fraction twice.
    #[inline]
    pub(super) fn clear_stencil(&self, x0: f32, y0: f32, tx: f32, ty: f32, near: bool) -> Stencil {
        let s = Stencil::at(tx, ty, self.w, self.h);
        if !near || !self.has_obstacle || !self.stencil_hits_obstacle(s) {
            return s;
        }
        let (px, py) = self.trace_to_fluid(x0, y0, tx, ty);
        Stencil::at(px, py, self.w, self.h)
    }

    /// Traces `(x, y)` along the velocity field for `dt` seconds — negative to
    /// backtrace — with the midpoint rule, subdivided so no substep crosses
    /// much more than [`TRACE_CELLS_PER_STEP`].
    ///
    /// `u0`/`v0` are the cell's own lattice velocity, which is exact, so the
    /// first substep needs no fetch and a short trace costs exactly what the
    /// single midpoint step cost. Each later substep predicts with the previous
    /// substep's midpoint velocity rather than re-reading the field at its own
    /// start: that estimate is stale by `O(hs)`, which only moves the midpoint
    /// by `O(hs^2)` and leaves the step second order, for one gather per
    /// substep instead of two.
    #[inline]
    pub(super) fn integrate(&self, x: f32, y: f32, dt: f32, u0: f32, v0: f32) -> (f32, f32) {
        let span = dt.abs() * length(u0, v0) / TRACE_CELLS_PER_STEP;
        // `max` drops NaN and a float-to-int cast saturates, so a poisoned
        // velocity yields a count in `1..=MAX_TRACE_STEPS` and never a
        // zero-length or unbounded loop.
        let steps = (span.ceil().max(1.0) as usize).min(MAX_TRACE_STEPS);
        let hs = dt / steps as f32;
        let (mut px, mut py) = (x, y);
        let (mut u, mut v) = (u0, v0);
        for _ in 0..steps {
            let (um, vm) = self.velocity_at(px + 0.5 * hs * u, py + 0.5 * hs * v);
            px += hs * um;
            py += hs * vm;
            u = um;
            v = vm;
        }
        (px, py)
    }

    /// Bilinear velocity fetch. Both components share one stencil, so this is
    /// half the index arithmetic of sampling the two grids independently.
    #[inline]
    pub(super) fn velocity_at(&self, x: f32, y: f32) -> (f32, f32) {
        let (ix, fx) = corner_of(x, self.w);
        let (iy, fy) = corner_of(y, self.h);
        let corner = iy * self.w + ix;
        (
            fetch(&self.vel.u.data, self.w, corner, fx, fy),
            fetch(&self.vel.v.data, self.w, corner, fx, fy),
        )
    }

    /// Fades dye in the band of fluid cells around the body silhouette.
    ///
    /// The band is the obstacle mask dilated by [`BODY_CONTACT_BAND`], built
    /// by [`Reach`] with running counts, so its cost does not depend on the band.
    ///
    /// Only runs while a body is in frame; with none the obstacle mask is empty
    /// and the whole sweep would be a no-op.
    pub(super) fn fade_dye_at_body(&mut self, dt: f32) {
        if !self.has_obstacle {
            return;
        }
        let (w, h) = (self.w, self.h);
        let band = BODY_CONTACT_BAND;
        let fade = decay(BODY_CONTACT_FADE, dt);
        if w <= 2 * band || h <= 2 * band {
            return;
        }

        self.reach.build(&self.obstacle.data, w, h, band);
        for i in 0..w * h {
            if self.obstacle.data[i] >= 0.5 {
                // Dye that ended up inside the silhouette can never advect
                // out again, so it is cleared outright instead of faded.
                for channel in &mut self.dye {
                    channel.data[i] = 0.0;
                }
                continue;
            }
            if !self.reach.near(i) {
                continue;
            }
            for channel in &mut self.dye {
                channel.data[i] *= fade;
            }
        }
    }

    /// Shortens a trace until its endpoint is out of the walls.
    ///
    /// Without this the bilinear fetch mixes the stale values sitting inside
    /// the body silhouette into the fluid, which reads on screen as dye
    /// bleeding straight through the body.
    pub(super) fn trace_to_fluid(&self, x0: f32, y0: f32, tx: f32, ty: f32) -> (f32, f32) {
        // With no body in frame there is nothing to back away from, so the
        // whole walk — a random read into the mask per cell, a third of the
        // trace stage — is skipped outright.
        if !self.has_obstacle || !self.obstacle_in_stencil(tx, ty) {
            return (tx, ty);
        }
        for &t in &TRACE_RETRIES {
            let px = x0 + (tx - x0) * t;
            let py = y0 + (ty - y0) * t;
            if !self.obstacle_in_stencil(px, py) {
                return (px, py);
            }
        }
        // Every point along the ray would fetch from the body, which is the
        // normal case for the rim of cells pressed against it: stay put. At the
        // lattice point itself the body's corners carry weight zero, so this
        // fallback is always a clean fetch of the cell's own value.
        (x0, y0)
    }

    /// Whether a bilinear fetch at `(x, y)` would blend in an obstacle cell
    /// with more than negligible weight.
    ///
    /// Testing only the nearest cell is not enough: a trace that stops just
    /// clear of a silhouette still has the body as a stencil corner, and at
    /// 40% weight that is dye visibly bleeding out of the body. See
    /// [`MIN_STENCIL_WEIGHT`] for why the test is a weight and not a flag.
    ///
    /// Deliberately the *obstacle* mask and not `solid`: the domain border is a
    /// wall too, but the fetch already clamps against it, and backing away
    /// from it as well would make the border behave differently depending on
    /// whether a body happens to be in frame.
    #[inline]
    pub(super) fn obstacle_in_stencil(&self, x: f32, y: f32) -> bool {
        self.stencil_hits_obstacle(Stencil::at(x, y, self.w, self.h))
    }

    /// [`Fluid::obstacle_in_stencil`] for an already resolved stencil.
    #[inline]
    pub(super) fn stencil_hits_obstacle(&self, s: Stencil) -> bool {
        let c = s.corner as usize;
        let (fx, fy) = (s.fx, s.fy);
        let o = &self.obstacle.data;
        let (gx, gy) = (1.0 - fx, 1.0 - fy);
        (gx * gy > MIN_STENCIL_WEIGHT && o[c] >= 0.5)
            || (fx * gy > MIN_STENCIL_WEIGHT && o[c + 1] >= 0.5)
            || (gx * fy > MIN_STENCIL_WEIGHT && o[c + self.w] >= 0.5)
            || (fx * fy > MIN_STENCIL_WEIGHT && o[c + self.w + 1] >= 0.5)
    }
}
