//! The wasm32 SIMD128 Jacobi sweeps.
//!
//! Lanes are assembled with the safe `f32x4` constructor rather than
//! `v128_load`, which takes a raw pointer and would need `unsafe`; this crate
//! forbids it. The arithmetic below is still genuine packed f32x4 work, which is
//! where the time goes.

use super::{kernel_unusable, zero_walls, StencilRows};
use crate::fluid::PRESSURE_OMEGA;
use crate::par;

use core::arch::wasm32::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_ge, f32x4_mul, f32x4_splat, f32x4_sub, v128,
    v128_bitselect,
};

#[inline]
pub(in crate::fluid) fn lanes(s: &[f32], i: usize) -> v128 {
    f32x4(s[i], s[i + 1], s[i + 2], s[i + 3])
}

#[inline]
pub(in crate::fluid) fn store_lanes(s: &mut [f32], i: usize, v: v128) {
    s[i] = f32x4_extract_lane::<0>(v);
    s[i + 1] = f32x4_extract_lane::<1>(v);
    s[i + 2] = f32x4_extract_lane::<2>(v);
    s[i + 3] = f32x4_extract_lane::<3>(v);
}

/// Lane-wise [`super::wall_pick`].
#[inline]
pub(in crate::fluid) fn wall_pick_v(value: v128, solid: v128, centre: v128) -> v128 {
    v128_bitselect(centre, value, f32x4_ge(solid, f32x4_splat(0.5)))
}

pub(in crate::fluid) fn jacobi_pressure_simd(
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

pub(in crate::fluid) fn jacobi_diffuse_simd(
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
