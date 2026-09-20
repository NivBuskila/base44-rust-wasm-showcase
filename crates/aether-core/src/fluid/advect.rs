//! Semi-Lagrangian advection: one shared bilinear trace map carries the
//! velocity and the three dye channels, and the dye is faded where it meets the
//! body.

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
        par::chunks_mut(&mut lanes, par::chunk_len(rows, 4), |band, lanes| {
            let per = lanes.len();
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
                    let (bx, by) = this.integrate(fx, fy, -dt, u0, v0);
                    let (bx, by) = this.trace_to_fluid(fx, fy, bx, by);
                    back[x] = Stencil::at(bx, by, w, h);

                    let (gx, gy) = this.integrate(fx, fy, dt, u0, v0);
                    let (gx, gy) = this.trace_to_fluid(fx, fy, gx, gy);
                    fwd[x] = Stencil::at(gx, gy, w, h);
                }
            }
        });
        drop(lanes);
        self.advector = advector;
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
    /// The band is the obstacle mask dilated by [`BODY_CONTACT_BAND`], computed
    /// separably (a horizontal max pass into `scratch`, then a vertical max read
    /// while the fade is applied) so the cost is `O(band)` per cell rather than
    /// `O(band^2)`.
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

        // scratch[i] = 1 when any cell within `band` on this row is solid.
        for y in 0..h {
            for x in 0..w {
                let lo = x.saturating_sub(band);
                let hi = (x + band).min(w - 1);
                let row = y * w;
                let hit = self.obstacle.data[row + lo..=row + hi]
                    .iter()
                    .any(|&o| o >= 0.5);
                self.scratch.data[row + x] = if hit { 1.0 } else { 0.0 };
            }
        }

        for y in 0..h {
            let lo = y.saturating_sub(band);
            let hi = (y + band).min(h - 1);
            for x in 0..w {
                let i = y * w + x;
                if self.obstacle.data[i] >= 0.5 {
                    // Dye that ended up inside the silhouette can never advect
                    // out again, so it is cleared outright instead of faded.
                    for channel in &mut self.dye {
                        channel.data[i] = 0.0;
                    }
                    continue;
                }
                let near = (lo..=hi).any(|k| self.scratch.data[k * w + x] >= 0.5);
                if !near {
                    continue;
                }
                for channel in &mut self.dye {
                    channel.data[i] *= fade;
                }
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
        let (ix, fx) = corner_of(x, self.w);
        let (iy, fy) = corner_of(y, self.h);
        let c = iy * self.w + ix;
        let o = &self.obstacle.data;
        let (gx, gy) = (1.0 - fx, 1.0 - fy);
        (gx * gy > MIN_STENCIL_WEIGHT && o[c] >= 0.5)
            || (fx * gy > MIN_STENCIL_WEIGHT && o[c + 1] >= 0.5)
            || (gx * fy > MIN_STENCIL_WEIGHT && o[c + self.w] >= 0.5)
            || (fx * fy > MIN_STENCIL_WEIGHT && o[c + self.w + 1] >= 0.5)
    }
}

/// A resolved bilinear fetch: the index of the stencil's top-left corner plus
/// the two blend fractions.
///
/// Storing the resolved stencil instead of the raw trace position hoists the
/// clamp, floor and cast out of five per-cell fetches (`u`, `v` and three dye
/// channels all trace identically), and lets the fetch read a `w + 2` window
/// so the bounds check collapses to one per corner group. Packing the three
/// fields into one struct rather than three parallel arrays cuts the trace
/// build from six write streams to two, which is worth ~15% of the stage.
#[derive(Clone, Copy)]
pub(super) struct Stencil {
    corner: u32,
    fx: f32,
    fy: f32,
}

impl Stencil {
    /// The stencil for a trace endpoint `(x, y)`, clamped into the grid.
    #[inline]
    pub(super) fn at(x: f32, y: f32, w: usize, h: usize) -> Self {
        let (ix, fx) = corner_of(x, w);
        let (iy, fy) = corner_of(y, h);
        Self {
            corner: (iy * w + ix) as u32,
            fx,
            fy,
        }
    }
}

/// One [`Stencil`] per cell: where this cell's value comes from this step.
pub(super) struct TraceMap {
    cells: Vec<Stencil>,
}

impl TraceMap {
    pub(super) fn new(n: usize) -> Self {
        Self {
            cells: vec![
                Stencil {
                    corner: 0,
                    fx: 0.0,
                    fy: 0.0,
                };
                n
            ],
        }
    }

    #[inline]
    pub(super) fn fetch(&self, g: &[f32], w: usize, i: usize) -> f32 {
        let s = self.cells[i];
        fetch(g, w, s.corner as usize, s.fx, s.fy)
    }

    /// Fetch plus the min and max of the four corners it blended — the range
    /// the MacCormack limiter is allowed to keep, for free from the same loads.
    #[inline]
    pub(super) fn fetch_limited(&self, g: &[f32], w: usize, i: usize) -> (f32, f32, f32) {
        let s = self.cells[i];
        let start = s.corner as usize;
        let end = start + w + 2;
        if end > g.len() {
            return (0.0, 0.0, 0.0);
        }
        let q = &g[start..end];
        let (a, b, c, d) = (q[0], q[1], q[w], q[w + 1]);
        let top = a + (b - a) * s.fx;
        let bottom = c + (d - c) * s.fx;
        (
            top + (bottom - top) * s.fy,
            a.min(b).min(c).min(d),
            a.max(b).max(c).max(d),
        )
    }
}

/// Trace maps plus the workspace the error-corrected advection needs.
pub(super) struct Advector {
    back: TraceMap,
    fwd: TraceMap,
    /// The backward fetch, kept whole because the forward pass gathers from it.
    hat: Vec<f32>,
    /// Limiter bounds from the backward fetch. Stored rather than re-derived:
    /// re-deriving them means a second random four-corner gather per cell,
    /// which is the most expensive thing in the advection.
    lo: Vec<f32>,
    hi: Vec<f32>,
}

impl Advector {
    pub(super) fn new(n: usize) -> Self {
        Self {
            back: TraceMap::new(n),
            fwd: TraceMap::new(n),
            hat: vec![0.0; n],
            lo: vec![0.0; n],
            hi: vec![0.0; n],
        }
    }

    /// Advects `field` in place through the current trace maps.
    ///
    /// MacCormack: the backward fetch `hat`, the forward fetch `bar` of that
    /// result, then `hat + (field - bar) / 2` clamped to the range the backward
    /// fetch could legitimately have returned. The limiter is not optional —
    /// unlimited MacCormack overshoots at every sharp dye edge and the
    /// overshoot compounds into ringing and then into NaN within seconds.
    ///
    /// Two passes, two gathers per cell. The correction can be written straight
    /// back into `field` because it only ever reads `field` at its own index.
    pub(super) fn run(&mut self, field: &mut [f32], w: usize, h: usize) {
        let n = w * h;
        if field.len() != n || self.hat.len() != n {
            return;
        }

        // Both passes are gathers into private output ranges, so each runs
        // over row bands in parallel; the barrier between them is the one the
        // MacCormack correction needs anyway (`bar` gathers from all of `hat`).
        let band = par::chunk_len(h, 4) * w;
        let back = &self.back;
        let field_ro: &[f32] = field;
        let mut outs: Vec<_> = self
            .hat
            .chunks_mut(band)
            .zip(self.lo.chunks_mut(band))
            .zip(self.hi.chunks_mut(band))
            .collect();
        par::chunks_mut(&mut outs, 1, |c, one| {
            let ((hat, lo), hi) = &mut one[0];
            let base = c * band;
            for k in 0..hat.len() {
                let (value, l, h) = back.fetch_limited(field_ro, w, base + k);
                hat[k] = value;
                lo[k] = l;
                hi[k] = h;
            }
        });
        drop(outs);

        let (fwd, hat, lo, hi) = (&self.fwd, &self.hat, &self.lo, &self.hi);
        par::chunks_mut(field, band, |c, cells| {
            let base = c * band;
            for (k, cell) in cells.iter_mut().enumerate() {
                let i = base + k;
                let bar = fwd.fetch(hat, w, i);
                let corrected = hat[i] + 0.5 * (*cell - bar);
                // `max`/`min` rather than `clamp`: they return the bound when
                // one operand is not a number, so a stray NaN dies here
                // instead of spreading one cell further every frame.
                *cell = corrected.max(lo[i]).min(hi[i]);
            }
        });
    }
}

/// Corner index and blend fraction for a clamped bilinear fetch on one axis.
///
/// The index is clamped to `n - 2` rather than `n - 1` so the `+1` neighbour
/// always exists; the fraction absorbs the difference, which reproduces
/// [`Grid::sample`]'s clamped behaviour including for out-of-range and NaN
/// inputs.
#[inline]
pub(super) fn corner_of(x: f32, n: usize) -> (usize, f32) {
    if x.is_nan() || n < 2 {
        return (0, 0.0);
    }
    let clamped = x.clamp(0.0, (n - 1) as f32);
    let i = (clamped as usize).min(n - 2);
    (i, clamped - i as f32)
}

/// Bilinear blend of the four values at `corner`, `corner + 1`, `corner + w`
/// and `corner + w + 1`.
#[inline]
pub(super) fn fetch(g: &[f32], w: usize, corner: usize, fx: f32, fy: f32) -> f32 {
    let end = corner + w + 2;
    if end > g.len() {
        return 0.0;
    }
    let q = &g[corner..end];
    let top = q[0] + (q[1] - q[0]) * fx;
    let bottom = q[w] + (q[w + 1] - q[w]) * fx;
    top + (bottom - top) * fy
}
