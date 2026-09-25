//! Fixtures: inert parameters, dye mass/centroid and curl measures, the bar mask and the reference Jacobi sweeps the kernels are checked against.

pub(super) use crate::fluid::advect::{corner_of, Stencil};
pub(super) use crate::fluid::*;

pub(super) use crate::config::Params;
pub(super) use crate::config::{FLUID_H, FLUID_W};
pub(super) use crate::rng::Rng;
pub(super) const DT: f32 = 1.0 / 60.0;
// --- kernel cross-checks ---------------------------------------------
// --- trace CFL ---------------------------------------------------------
// --- boundaries --------------------------------------------------------
// --- projection --------------------------------------------------------

/// Params with every decay and side effect off, so a test measures exactly
/// the stage it is aiming at.
pub(super) fn inert_params() -> Params {
    Params {
        velocity_dissipation: 0.0,
        dye_dissipation: 0.0,
        vorticity: 0.0,
        viscosity: 0.0,
        pressure_iters: 30,
        ..Params::default()
    }
}

pub(super) fn dye_mass(f: &Fluid) -> f32 {
    f.dye[0].sum()
}

/// Dye centroid in grid space; `None` when there is no dye to speak of.
pub(super) fn dye_centroid(f: &Fluid) -> Option<(f32, f32)> {
    let mut wsum = 0.0;
    let mut xs = 0.0;
    let mut ys = 0.0;
    for y in 0..f.h {
        for x in 0..f.w {
            let m = f.dye[0].data[y * f.w + x];
            wsum += m;
            xs += m * x as f32;
            ys += m * y as f32;
        }
    }
    if wsum <= 1e-6 {
        return None;
    }
    Some((xs / wsum, ys / wsum))
}

pub(super) fn total_curl(f: &Fluid) -> f32 {
    f.curl.data.iter().map(|c| c.abs()).sum()
}

/// A vertical solid bar spanning the full height.
pub(super) fn bar_mask(w: usize, h: usize, x0: usize, x1: usize) -> Grid {
    let mut m = Grid::new(w, h);
    for y in 0..h {
        for x in x0..x1 {
            m.data[y * w + x] = 1.0;
        }
    }
    m
}

/// Deliberately naive pressure sweep, written from the 2-D indices with an
/// explicit per-neighbour wall test. Independent of the shipping kernels,
/// so agreement means something.
pub(super) fn reference_pressure(
    p: &[f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let is_solid = |x: usize, y: usize| solid[y * w + x] >= 0.5;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if is_solid(x, y) {
                continue;
            }
            let pc = p[y * w + x];
            let at = |x: usize, y: usize| if is_solid(x, y) { pc } else { p[y * w + x] };
            let sweep =
                (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - div[y * w + x]) / 4.0;
            out[y * w + x] = (1.0 - PRESSURE_OMEGA) * pc + PRESSURE_OMEGA * sweep;
        }
    }
    out
}

pub(super) fn reference_diffuse(
    x: &[f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for j in 1..h - 1 {
        for i in 1..w - 1 {
            if solid[j * w + i] >= 0.5 {
                continue;
            }
            let sum = x[j * w + i - 1] + x[j * w + i + 1] + x[(j - 1) * w + i] + x[(j + 1) * w + i];
            out[j * w + i] = (rhs[j * w + i] + a * sum) / (1.0 + 4.0 * a);
        }
    }
    out
}

/// Pseudo-random pressure, divergence and wall fields.
pub(super) fn kernel_fixture(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(0xBEEF_1234);
    let n = w * h;
    let p: Vec<f32> = (0..n).map(|_| rng.range(-5.0, 5.0)).collect();
    let div: Vec<f32> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
    let mut solid = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            // A blob of interior walls plus the border, so both branches of
            // every neighbour select get exercised.
            let blob = (x as i32 - (w as i32) / 2).abs() < 3 && (y as i32 - 4).abs() < 2;
            if border || blob {
                solid[y * w + x] = 1.0;
            }
        }
    }
    (p, div, solid)
}
