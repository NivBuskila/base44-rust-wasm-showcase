//! The bilinear machinery the advection stage gathers through: resolved
//! stencils, the per-cell trace maps, and the MacCormack workspace that runs
//! a field through them.

use crate::par;

/// A resolved bilinear fetch: the index of the stencil's top-left corner plus
/// the two blend fractions.
///
/// Storing the resolved stencil instead of the raw trace position hoists the
/// clamp, floor and cast out of five per-cell fetches (`u`, `v` and three dye
/// channels all trace identically), and lets the fetch read a `w + 2` window
/// so the bounds check collapses to one per corner group. Packing the three
/// fields into one struct rather than three parallel arrays cuts the trace
/// build from six write streams to two, which is worth ~15% of the stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::fluid) struct Stencil {
    pub(super) corner: u32,
    pub(super) fx: f32,
    pub(super) fy: f32,
}

impl Stencil {
    /// The stencil for a trace endpoint `(x, y)`, clamped into the grid.
    #[inline]
    pub(in crate::fluid) fn at(x: f32, y: f32, w: usize, h: usize) -> Self {
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
pub(in crate::fluid) struct TraceMap {
    pub(super) cells: Vec<Stencil>,
}

impl TraceMap {
    pub(in crate::fluid) fn new(n: usize) -> Self {
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
    pub(in crate::fluid) fn fetch(&self, g: &[f32], w: usize, i: usize) -> f32 {
        let s = self.cells[i];
        fetch(g, w, s.corner as usize, s.fx, s.fy)
    }

    /// Fetch plus the min and max of the four corners it blended — the range
    /// the MacCormack limiter is allowed to keep, for free from the same loads.
    #[inline]
    pub(in crate::fluid) fn fetch_limited(&self, g: &[f32], w: usize, i: usize) -> (f32, f32, f32) {
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
pub(in crate::fluid) struct Advector {
    pub(super) back: TraceMap,
    pub(super) fwd: TraceMap,
    /// The backward fetch, kept whole because the forward pass gathers from it.
    hat: Vec<f32>,
    /// Limiter bounds from the backward fetch. Stored rather than re-derived:
    /// re-deriving them means a second random four-corner gather per cell,
    /// which is the most expensive thing in the advection.
    lo: Vec<f32>,
    hi: Vec<f32>,
}

impl Advector {
    pub(in crate::fluid) fn new(n: usize) -> Self {
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
    pub(in crate::fluid) fn run(&mut self, field: &mut [f32], w: usize, h: usize) {
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
/// `Grid::sample`'s clamped behaviour including for out-of-range and NaN
/// inputs.
#[inline]
pub(in crate::fluid) fn corner_of(x: f32, n: usize) -> (usize, f32) {
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
pub(in crate::fluid) fn fetch(g: &[f32], w: usize, corner: usize, fx: f32, fy: f32) -> f32 {
    let end = corner + w + 2;
    if end > g.len() {
        return 0.0;
    }
    let q = &g[corner..end];
    let top = q[0] + (q[1] - q[0]) * fx;
    let bottom = q[w] + (q[w + 1] - q[w]) * fx;
    top + (bottom - top) * fy
}
