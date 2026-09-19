//! The hot loops: five-point Jacobi sweeps for pressure and diffusion, in a
//! wasm32 SIMD128 path and a scalar fallback that stays compiled everywhere,
//! plus the small scalar helpers the stages share.

use super::*;

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
use core::arch::wasm32::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_ge, f32x4_mul, f32x4_splat, f32x4_sub, v128,
    v128_bitselect,
};



/// Clamps a timestep into the range the explicit stages stay stable over.
///
/// A backgrounded tab hands back a `dt` of several seconds. Semi-Lagrangian
/// advection would survive that, but the trace would jump most of the way
/// across the grid and confinement would inject a frame's worth of energy
/// several hundred times over.
#[inline]
pub(super) fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 480.0, 1.0 / 20.0)
}

/// Clamps a velocity impulse to `+/- limit`, mapping a non-finite one to zero.
///
/// The zero case is not theoretical: a vanishing vorticity gradient yields a
/// normal of exactly `(0, 0)`, and `0 * NaN` is NaN, so a single poisoned curl
/// cell would otherwise seed the whole velocity field.
#[inline]
pub(super) fn bounded_impulse(v: f32, limit: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-limit, limit)
    } else {
        0.0
    }
}

/// Substitutes the centre value for a wall neighbour.
///
/// For pressure this is the Neumann condition `dp/dn = 0` on a solid face.
/// Reading the wall cell's own pressure instead is precisely the bug that lets
/// fluid pour through the body silhouette.
#[inline]
pub(super) fn wall_pick(value: f32, solid: f32, centre: f32) -> f32 {
    if solid >= 0.5 {
        centre
    } else {
        value
    }
}

// ------------------------------------------------------------------ kernels
//
// The two Jacobi sweeps are the hot loops of the whole engine: at 256x144 with
// 28 pressure iterations they are ~1M cell updates per frame. Both have a
// wasm32 SIMD128 path and a scalar fallback that stays compiled everywhere, so
// the two can be diffed against each other.

/// The row windows one output row of a five-point sweep reads.
///
/// Slicing the rows out up front is not cosmetic: indexing `w`-long slices with
/// `x in 1..w - 1` lets the compiler discharge the bounds checks, which is
/// worth roughly 3x on the pressure sweep over indexing the flat grid.
pub(super) struct StencilRows<'a> {
    up: &'a [f32],
    mid: &'a [f32],
    down: &'a [f32],
    solid_up: &'a [f32],
    solid_mid: &'a [f32],
    solid_down: &'a [f32],
    rhs: &'a [f32],
}

impl<'a> StencilRows<'a> {
    /// Windows around row `y`. Requires `1 <= y <= h - 2` and slices of at
    /// least `w * h`.
    #[inline]
    pub(super) fn new(field: &'a [f32], solid: &'a [f32], rhs: &'a [f32], w: usize, y: usize) -> Self {
        let row = y * w;
        Self {
            up: &field[row - w..row],
            mid: &field[row..row + w],
            down: &field[row + w..row + 2 * w],
            solid_up: &solid[row - w..row],
            solid_mid: &solid[row..row + w],
            solid_down: &solid[row + w..row + 2 * w],
            rhs: &rhs[row..row + w],
        }
    }

    /// Damped Jacobi pressure update for one cell of this row.
    ///
    /// The wall test comes last, as a select on an already-computed value,
    /// rather than as an early return. That is not a style choice: an early
    /// return here costs 4.2 ns/cell against 1.2 ns/cell for the select,
    /// because the branch is data-dependent and blocks vectorisation.
    #[inline]
    pub(super) fn pressure(&self, x: usize) -> f32 {
        let pc = self.mid[x];
        let l = wall_pick(self.mid[x - 1], self.solid_mid[x - 1], pc);
        let r = wall_pick(self.mid[x + 1], self.solid_mid[x + 1], pc);
        let u = wall_pick(self.up[x], self.solid_up[x], pc);
        let d = wall_pick(self.down[x], self.solid_down[x], pc);
        // Grouped exactly as the SIMD path adds it, so the two are bit-equal
        // and the cross-check test can use a tight tolerance.
        let sweep = ((l + r) + (u + d) - self.rhs[x]) * 0.25;
        let relaxed = pc + PRESSURE_OMEGA * (sweep - pc);
        if self.solid_mid[x] >= 0.5 {
            0.0
        } else {
            relaxed
        }
    }

    /// Jacobi diffusion update for one cell of this row.
    #[inline]
    pub(super) fn diffuse(&self, x: usize, a: f32, inv: f32) -> f32 {
        // A wall neighbour contributes its own (zero) velocity rather than a
        // mirrored copy of the centre: a wall *drags* the fluid beside it,
        // which is the entire physical content of no-slip.
        let sum = (self.mid[x - 1] + self.mid[x + 1]) + (self.up[x] + self.down[x]);
        let relaxed = (self.rhs[x] + a * sum) * inv;
        if self.solid_mid[x] >= 0.5 {
            0.0
        } else {
            relaxed
        }
    }
}

/// One Jacobi sweep of `laplacian(p) = div`, with walls as Neumann boundaries.
///
/// Writes *every* cell of `out`, walls included, because the caller ping-pongs
/// the two buffers and stale values would otherwise survive forever.
#[inline]
pub(super) fn jacobi_pressure(p: &[f32], out: &mut [f32], div: &[f32], solid: &[f32], w: usize, h: usize) {
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    jacobi_pressure_simd(p, out, div, solid, w, h);
    #[cfg(not(all(target_arch = "wasm32", feature = "simd")))]
    jacobi_pressure_scalar(p, out, div, solid, w, h);
}

/// One Jacobi sweep of `(I - a * laplacian) u = rhs`.
#[inline]
pub(super) fn jacobi_diffuse(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    jacobi_diffuse_simd(x, out, rhs, solid, w, h, a);
    #[cfg(not(all(target_arch = "wasm32", feature = "simd")))]
    jacobi_diffuse_scalar(x, out, rhs, solid, w, h, a);
}

/// True when the kernels cannot run over these buffers at all.
#[inline]
pub(super) fn kernel_unusable(n: usize, w: usize, h: usize, lens: [usize; 4]) -> bool {
    w < 3 || h < 3 || lens.iter().any(|&l| l < n)
}

/// Runs `f(y, row)` over every interior row of a full-grid buffer, in parallel.
///
/// The buffer is passed separately from `&Self` on purpose: a stage that writes
/// one grid while reading its neighbours out of the others cannot hold `&mut
/// self`, so the caller `mem::take`s the target's storage, hands it here, and
/// puts it back. `row` is indexed by `x`, so a stencil can keep using the flat
/// `y * w + x` index for its reads.
pub(super) fn interior_rows<F>(data: &mut [f32], w: usize, h: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if w < 3 || h < 3 || data.len() < w * h {
        return;
    }
    par::rows_mut(&mut data[w..(h - 1) * w], w, h - 2, |k, row| f(k + 1, row));
}

/// [`interior_rows`] over two grids at once — the velocity components, which
/// every force stage writes together.
pub(super) fn interior_row_pairs<F>(a: &mut [f32], b: &mut [f32], w: usize, h: usize, f: F)
where
    F: Fn(usize, &mut [f32], &mut [f32]) + Sync + Send,
{
    if w < 3 || h < 3 || a.len() < w * h || b.len() < w * h {
        return;
    }
    let rows = h - 2;
    let per = par::chunk_len(rows, 4);
    let mut lanes: Vec<_> = a[w..(h - 1) * w]
        .chunks_mut(w)
        .zip(b[w..(h - 1) * w].chunks_mut(w))
        .collect();
    par::chunks_mut(&mut lanes, per, |band, lanes| {
        for (k, (ar, br)) in lanes.iter_mut().enumerate() {
            f(band * per + k + 1, ar, br);
        }
    });
}

#[inline]
pub(super) fn zero_walls(out: &mut [f32], w: usize, h: usize) {
    if w < 2 || h < 2 || out.len() < w * h {
        out.fill(0.0);
        return;
    }
    out[..w].fill(0.0);
    out[(h - 1) * w..h * w].fill(0.0);
    for y in 1..h - 1 {
        out[y * w] = 0.0;
        out[y * w + w - 1] = 0.0;
    }
}

#[cfg_attr(all(target_arch = "wasm32", feature = "simd"), allow(dead_code))]
pub(super) fn jacobi_pressure_scalar(
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [p.len(), out.len(), div.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    zero_walls(out, w, h);
    par::rows_mut(&mut out[w..(h - 1) * w], w, h - 2, |k, o| {
        let rows = StencilRows::new(p, solid, div, w, k + 1);
        for (x, cell) in o.iter_mut().enumerate().take(w - 1).skip(1) {
            *cell = rows.pressure(x);
        }
    });
}

#[cfg_attr(all(target_arch = "wasm32", feature = "simd"), allow(dead_code))]
pub(super) fn jacobi_diffuse_scalar(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [x.len(), out.len(), rhs.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let inv = 1.0 / (1.0 + 4.0 * a);
    zero_walls(out, w, h);
    par::rows_mut(&mut out[w..(h - 1) * w], w, h - 2, |k, o| {
        let rows = StencilRows::new(x, solid, rhs, w, k + 1);
        for (i, cell) in o.iter_mut().enumerate().take(w - 1).skip(1) {
            *cell = rows.diffuse(i, a, inv);
        }
    });
}

// --- wasm32 SIMD128 paths -------------------------------------------------
//
// Lanes are assembled with the safe `f32x4` constructor rather than
// `v128_load`, which takes a raw pointer and would need `unsafe`; this crate
// forbids it. The arithmetic below is still genuine packed f32x4 work, which is
// where the time goes.

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
pub(super) fn lanes(s: &[f32], i: usize) -> v128 {
    f32x4(s[i], s[i + 1], s[i + 2], s[i + 3])
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
pub(super) fn store_lanes(s: &mut [f32], i: usize, v: v128) {
    s[i] = f32x4_extract_lane::<0>(v);
    s[i + 1] = f32x4_extract_lane::<1>(v);
    s[i + 2] = f32x4_extract_lane::<2>(v);
    s[i + 3] = f32x4_extract_lane::<3>(v);
}

/// Lane-wise [`wall_pick`].
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
pub(super) fn wall_pick_v(value: v128, solid: v128, centre: v128) -> v128 {
    v128_bitselect(centre, value, f32x4_ge(solid, f32x4_splat(0.5)))
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
pub(super) fn jacobi_pressure_simd(
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [p.len(), out.len(), div.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let zero = f32x4_splat(0.0);
    let quarter = f32x4_splat(0.25);
    let half = f32x4_splat(0.5);
    let omega = f32x4_splat(PRESSURE_OMEGA);

    zero_walls(out, w, h);
    par::rows_mut(&mut out[w..(h - 1) * w], w, h - 2, |k, o| {
        let rows = StencilRows::new(p, solid, div, w, k + 1);
        let mut x = 1;
        // A row is contiguous, so the left/right neighbour vectors are just
        // this row shifted by one element — no shuffles needed.
        while x + 4 <= w - 1 {
            let pc = lanes(rows.mid, x);
            let l = wall_pick_v(lanes(rows.mid, x - 1), lanes(rows.solid_mid, x - 1), pc);
            let r = wall_pick_v(lanes(rows.mid, x + 1), lanes(rows.solid_mid, x + 1), pc);
            let u = wall_pick_v(lanes(rows.up, x), lanes(rows.solid_up, x), pc);
            let d = wall_pick_v(lanes(rows.down, x), lanes(rows.solid_down, x), pc);
            let sum = f32x4_add(f32x4_add(l, r), f32x4_add(u, d));
            let sweep = f32x4_mul(f32x4_sub(sum, lanes(rows.rhs, x)), quarter);
            let res = f32x4_add(pc, f32x4_mul(omega, f32x4_sub(sweep, pc)));
            let solid_centre = f32x4_ge(lanes(rows.solid_mid, x), half);
            store_lanes(o, x, v128_bitselect(zero, res, solid_centre));
            x += 4;
        }
        while x < w - 1 {
            o[x] = rows.pressure(x);
            x += 1;
        }
    });
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
pub(super) fn jacobi_diffuse_simd(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [x.len(), out.len(), rhs.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let inv_scalar = 1.0 / (1.0 + 4.0 * a);
    let zero = f32x4_splat(0.0);
    let half = f32x4_splat(0.5);
    let av = f32x4_splat(a);
    let inv = f32x4_splat(inv_scalar);

    zero_walls(out, w, h);
    par::rows_mut(&mut out[w..(h - 1) * w], w, h - 2, |k, o| {
        let rows = StencilRows::new(x, solid, rhs, w, k + 1);
        let mut i = 1;
        while i + 4 <= w - 1 {
            let sum = f32x4_add(
                f32x4_add(lanes(rows.mid, i - 1), lanes(rows.mid, i + 1)),
                f32x4_add(lanes(rows.up, i), lanes(rows.down, i)),
            );
            let res = f32x4_mul(f32x4_add(lanes(rows.rhs, i), f32x4_mul(av, sum)), inv);
            let solid_centre = f32x4_ge(lanes(rows.solid_mid, i), half);
            store_lanes(o, i, v128_bitselect(zero, res, solid_centre));
            i += 4;
        }
        while i < w - 1 {
            o[i] = rows.diffuse(i, a, inv_scalar);
            i += 1;
        }
    });
}
