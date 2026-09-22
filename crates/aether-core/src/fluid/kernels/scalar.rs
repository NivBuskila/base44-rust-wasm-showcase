//! The portable Jacobi sweeps. They stay compiled everywhere, including in the
//! SIMD build, so the two paths can be diffed against each other in tests.

use super::{kernel_unusable, zero_walls, StencilRows};
use crate::par;

#[cfg_attr(all(target_arch = "wasm32", feature = "simd"), allow(dead_code))]
pub(in crate::fluid) fn jacobi_pressure_scalar(
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
pub(in crate::fluid) fn jacobi_diffuse_scalar(
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
