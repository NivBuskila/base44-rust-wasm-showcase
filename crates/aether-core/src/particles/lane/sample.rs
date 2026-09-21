//! The two field reads on the per-particle critical path: the shared bilinear
//! gather and the nearest-lattice curl.
//!
//! Both are deliberately hand-rolled rather than routed through `Grid`/
//! `VecField`, and both comment why: this runs 220k times a frame, twice per
//! particle for the gather.

use crate::particles::*;

/// Bilinear sample of both velocity components at once.
///
/// Identical in result to two [`Grid::sample`] calls, but the clamp, floor and
/// index arithmetic are shared: the compiler cannot prove `u.w == v.w`, so
/// going through `VecField::sample` computes all of it twice, and `step` does
/// two of these per particle.
#[inline(always)]
pub(in crate::particles) fn sample_uv(vel: &VecField, x: f32, y: f32) -> (f32, f32) {
    let w = vel.u.w;
    let h = vel.u.h;
    // Clamp and integer split exactly as `Grid::sample` does them, for the
    // reasons documented there: `max`/`min` fold NaN to the origin without a
    // branch, and truncation avoids `f32::floor`, which is a libm call on the
    // SSE2 baseline. Four of those per particle measured at 25 ns here, more
    // than the rest of the step put together.
    let x = x.max(0.0).min((w - 1) as f32);
    let y = y.max(0.0).min((h - 1) as f32);

    let ix0 = x as usize;
    let iy0 = y as usize;
    let tx = x - ix0 as f32;
    let ty = y - iy0 as f32;

    let ix1 = (ix0 + 1).min(w - 1);
    let iy1 = (iy0 + 1).min(h - 1);
    let row0 = iy0 * w;
    let row1 = iy1 * w;

    // Nested `lerp`, bit-for-bit what `Grid::sample` computes: the fluid
    // backtraces through the same field, and a particle that disagreed with
    // the solver in the last bit would be a needless source of drift between
    // what the dye shows and where the particles are. (A flattened
    // weighted-sum form measured no faster here, so there is nothing to trade
    // that agreement for.)
    let u = lerp(
        lerp(vel.u.data[row0 + ix0], vel.u.data[row0 + ix1], tx),
        lerp(vel.u.data[row1 + ix0], vel.u.data[row1 + ix1], tx),
        ty,
    );
    let v = lerp(
        lerp(vel.v.data[row0 + ix0], vel.v.data[row0 + ix1], tx),
        lerp(vel.v.data[row1 + ix0], vel.v.data[row1 + ix1], tx),
        ty,
    );
    (u, v)
}

/// Curl `omega = dv/dx - du/dy` at the lattice point nearest `(x, y)`.
///
/// The fluid solver keeps its own curl grid, but `step` is handed only the
/// velocity field, and recomputing it here from four loads is cheaper than the
/// eight bilinear samples an interpolated curl would cost. Matches
/// `Fluid::compute_curl` at interior lattice points, and degrades to a
/// one-sided difference on the border.
#[inline(always)]
pub(in crate::particles) fn curl_at(vel: &VecField, x: f32, y: f32) -> f32 {
    let w = vel.u.w;
    let h = vel.u.h;
    let xi = ((x + 0.5) as usize).min(w - 1);
    let yi = ((y + 0.5) as usize).min(h - 1);
    let xm = xi.saturating_sub(1);
    let xp = (xi + 1).min(w - 1);
    let ym = yi.saturating_sub(1);
    let yp = (yi + 1).min(h - 1);
    // The span is 2 in the interior and 1 against a border, so the per-cell
    // normalisation is a select on an integer, not a divide. Two `divss` here
    // measured at ~7 ns per particle, a tenth of the entire step: this runs
    // once per particle per frame, 220k times, on the critical path.
    let inv_dx = if xp - xm == 2 { 0.5 } else { 1.0 };
    let inv_dy = if yp - ym == 2 { 0.5 } else { 1.0 };
    let dvdx = (vel.v.data[yi * w + xp] - vel.v.data[yi * w + xm]) * inv_dx;
    let dudy = (vel.u.data[yp * w + xi] - vel.u.data[ym * w + xi]) * inv_dy;
    dvdx - dudy
}
