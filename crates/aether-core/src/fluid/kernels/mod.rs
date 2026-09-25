//! The hot loops: five-point Jacobi sweeps for pressure and diffusion, in a
//! wasm32 SIMD128 path and a scalar fallback that stays compiled everywhere,
//! plus the small scalar helpers the stages share.
//!
//! The two Jacobi sweeps are the hot loops of the whole engine: at 256x144 with
//! 28 pressure iterations they are ~1M cell updates per frame. `stencil.rs` is
//! the row window and the per-cell arithmetic, `scalar.rs` the portable sweeps,
//! `simd.rs` the packed f32x4 ones; this module is the dispatch plus the row
//! plumbing and guards every stage shares.

mod scalar;
mod stencil;

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
mod simd;

// The scalar sweeps are the fallback dispatch target, and the tests compare the
// SIMD path against them, so the re-export is unused only in a SIMD release build.
#[allow(unused_imports)]
pub(super) use scalar::{jacobi_diffuse_scalar, jacobi_pressure_scalar};
pub(super) use stencil::{wall_pick, StencilRows};

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
pub(super) use simd::{jacobi_diffuse_simd, jacobi_pressure_simd};

use super::*;

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

/// One Jacobi sweep of `laplacian(p) = div`, with walls as Neumann boundaries.
///
/// Writes *every* cell of `out`, walls included, because the caller ping-pongs
/// the two buffers and stale values would otherwise survive forever.
#[inline]
pub(super) fn jacobi_pressure(
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) {
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

/// Runs `f(k, row)` over the interior rows of one Jacobi sweep's output, where
/// row `k` is grid row `k + 1`, on the calling thread.
///
/// Serial in the threaded build too, on purpose. One pressure sweep is ~50 µs
/// of work (all 28 cost ~1.3 ms native), which is about what a fork/join costs
/// in worker wake-ups, and the sweeps are the one stage that pays it dozens of
/// times a frame. Measured in the browser on 4 threads, sweeping serially was
/// as fast or faster at every pool size, and most so with a large pool.
#[inline]
pub(super) fn sweep_rows<F>(out: &mut [f32], w: usize, h: usize, mut f: F)
where
    F: FnMut(usize, &mut [f32]),
{
    for (k, row) in out[w..(h - 1) * w].chunks_mut(w).enumerate() {
        f(k, row);
    }
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
