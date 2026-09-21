use super::*;
use crate::config::Params;
use crate::config::{FLUID_H, FLUID_W};
use crate::rng::Rng;

const DT: f32 = 1.0 / 60.0;

/// Params with every decay and side effect off, so a test measures exactly
/// the stage it is aiming at.
fn inert_params() -> Params {
    Params {
        velocity_dissipation: 0.0,
        dye_dissipation: 0.0,
        vorticity: 0.0,
        viscosity: 0.0,
        pressure_iters: 30,
        ..Params::default()
    }
}

fn dye_mass(f: &Fluid) -> f32 {
    f.dye[0].sum()
}

/// Dye centroid in grid space; `None` when there is no dye to speak of.
fn dye_centroid(f: &Fluid) -> Option<(f32, f32)> {
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

fn total_curl(f: &Fluid) -> f32 {
    f.curl.data.iter().map(|c| c.abs()).sum()
}

/// A vertical solid bar spanning the full height.
fn bar_mask(w: usize, h: usize, x0: usize, x1: usize) -> Grid {
    let mut m = Grid::new(w, h);
    for y in 0..h {
        for x in x0..x1 {
            m.data[y * w + x] = 1.0;
        }
    }
    m
}

#[test]
fn the_stepped_field_is_incompressible() {
    let mut f = Fluid::new(FLUID_W, FLUID_H);
    let params = Params::default();
    // Driven the way the engine drives it: a wandering impulse *rate*
    // re-injected every frame, not one huge one-off splat. Speed
    // accumulates over frames while the divergence each frame has to
    // remove stays bounded, which is the regime the HUD number lives in.
    //
    // The path is a fraction of the grid, not absolute cells: this is the
    // one test that deliberately runs at the app's real resolution, so
    // hard-coded coordinates would drift into the domain walls the next
    // time that resolution changes — and a wall is exactly where the
    // projection has the least freedom to correct divergence, so the test
    // would fail for a reason that has nothing to do with the solver.
    let (w, h) = (FLUID_W as f32, FLUID_H as f32);
    for k in 0..120 {
        let t = k as f32 / 60.0;
        let x = w * (0.5 + 0.27 * (t * 1.7).sin());
        let y = h * (0.5 + 0.28 * (t * 2.3).cos());
        let dir = t * 3.1;
        f.add_force(x, y, 12.0 * dir.cos(), 12.0 * dir.sin(), 12.0);
        f.add_dye(x, y, [0.6, 0.3, 0.1], 9.0);
        f.step(DT, &params);
    }
    let speed = f.max_speed();
    assert!(speed > 30.0, "the drive did nothing: max speed {speed}");
    let ratio = f.last_divergence() / speed;
    assert!(
        ratio < 0.02,
        "field is not incompressible: max|div| {} vs max|u| {speed} (ratio {ratio})",
        f.last_divergence()
    );
}

#[test]
fn the_projection_removes_the_divergence_it_is_handed() {
    // Straight at the solver: a wildly divergent field, one projection,
    // and no other stage to muddy the measurement.
    let (w, h) = (64usize, 48usize);
    let mut f = Fluid::new(w, h);
    f.set_obstacle(&bar_mask(w, h, 28, 33));
    let mut rng = Rng::new(0x0FF1_CE11);
    for i in 0..w * h {
        f.vel.u.data[i] = rng.signed() * 50.0;
        f.vel.v.data[i] = rng.signed() * 50.0;
    }
    f.enforce_boundaries();
    f.close_solid_faces();

    let before = f.max_divergence();
    assert!(before > 50.0, "the fixture was not divergent: {before}");
    f.project(Params::default().pressure_iters);
    let after = f.max_divergence();
    assert!(
        after < 0.03 * before,
        "projection left {after} of {before} ({:.1}%)",
        100.0 * after / before
    );
    // And it keeps improving with iterations, i.e. it is converging on the
    // right answer rather than sitting on a fixed point of the wrong one.
    f.project(200);
    assert!(f.max_divergence() < after, "more sweeps did not help");
}

#[test]
fn uniform_velocity_survives_advection() {
    let mut f = Fluid::new(48, 32);
    f.vel.u.data.fill(30.0);
    f.vel.v.data.fill(0.0);
    f.advect(1.0 / 30.0);
    for y in 4..28 {
        for x in 4..44 {
            let i = y * 48 + x;
            assert!(
                (f.vel.u.data[i] - 30.0).abs() < 1e-3,
                "u drifted at {x},{y}: {}",
                f.vel.u.data[i]
            );
            assert!(f.vel.v.data[i].abs() < 1e-3, "v appeared at {x},{y}");
        }
    }
}

#[test]
fn a_dye_blob_rides_the_flow_by_v_times_dt() {
    let mut f = Fluid::new(64, 48);
    f.vel.u.data.fill(20.0);
    f.add_dye(16.0, 24.0, [1.0, 0.0, 0.0], 6.0);
    let (x0, y0) = dye_centroid(&f).expect("dye was injected");

    let dt = 0.05;
    let steps = 4;
    for _ in 0..steps {
        f.advect(dt);
    }
    let (x1, y1) = dye_centroid(&f).expect("dye survived");

    let expected = 20.0 * dt * steps as f32;
    assert!(
        (x1 - x0 - expected).abs() < 0.2,
        "centroid moved {} cells, expected {expected}",
        x1 - x0
    );
    assert!(
        (y1 - y0).abs() < 0.05,
        "centroid drifted in y by {}",
        y1 - y0
    );
}

#[test]
fn advection_alone_conserves_dye_mass() {
    let mut f = Fluid::new(64, 48);
    f.vel.u.data.fill(10.0);
    f.vel.v.data.fill(4.0);
    f.add_dye(16.0, 16.0, [1.0, 0.0, 0.0], 7.0);
    let before = dye_mass(&f);
    assert!(before > 0.0);
    for _ in 0..30 {
        f.advect(0.05);
    }
    let after = dye_mass(&f);
    let drift = (after - before).abs() / before;
    assert!(drift < 0.03, "dye mass drifted {:.2}%", drift * 100.0);
}

#[test]
fn obstacle_cells_are_exactly_at_rest_after_a_step() {
    let mut f = Fluid::new(64, 48);
    f.set_obstacle(&bar_mask(64, 48, 28, 34));
    let params = Params::default();
    for _ in 0..20 {
        // Aim straight at the bar.
        f.add_force(20.0, 24.0, 300.0, 0.0, 9.0);
        f.step(DT, &params);
    }
    for y in 0..48 {
        for x in 0..64 {
            let i = y * 64 + x;
            if f.solid.data[i] >= 0.5 {
                assert_eq!(f.vel.u.data[i], 0.0, "u nonzero in wall at {x},{y}");
                assert_eq!(f.vel.v.data[i], 0.0, "v nonzero in wall at {x},{y}");
            }
        }
    }
}

#[test]
fn fluid_does_not_leak_through_a_solid_wall() {
    let (w, h) = (64, 48);
    let mut f = Fluid::new(w, h);
    f.set_obstacle(&bar_mask(w, h, 30, 34));
    let params = Params::default();
    for _ in 0..300 {
        f.add_force(12.0, 24.0, 320.0, 0.0, 8.0);
        f.add_dye(12.0, 24.0, [0.4, 0.4, 0.4], 6.0);
        f.step(DT, &params);
    }
    assert!(f.max_speed() > 10.0, "the push never took");

    let mut far_speed = 0.0f32;
    let mut far_dye = 0.0f32;
    for y in 0..h {
        for x in 38..w {
            let i = y * w + x;
            far_speed = far_speed.max(
                (f.vel.u.data[i] * f.vel.u.data[i] + f.vel.v.data[i] * f.vel.v.data[i]).sqrt(),
            );
            far_dye += f.dye[0].data[i];
        }
    }
    assert!(
        far_speed < 1e-4,
        "velocity leaked past the wall: {far_speed} (near side {})",
        f.max_speed()
    );
    assert!(far_dye < 1e-6, "dye leaked past the wall: {far_dye}");
}

#[test]
fn advection_never_fetches_from_inside_an_obstacle() {
    let (w, h) = (48usize, 32usize);
    let mut f = Fluid::new(w, h);
    f.set_obstacle(&bar_mask(w, h, 20, 26));
    // Dye parked inside the bar, fluid streaming right alongside it. The
    // column just clear of the bar backtraces into it, so without the
    // stencil walk the wall's dye bleeds straight out.
    for y in 0..h {
        for x in 20..26 {
            f.dye[0].data[y * w + x] = 1.0;
        }
    }
    f.vel.u.data.fill(25.0);
    f.enforce_boundaries();
    for _ in 0..8 {
        f.advect(1.0 / 30.0);
    }
    for y in 1..h - 1 {
        for x in 26..w - 1 {
            let v = f.dye[0].data[y * w + x];
            assert!(v < 1e-6, "dye bled out of the obstacle at {x},{y}: {v}");
        }
    }
}

#[test]
fn vorticity_confinement_adds_curl() {
    // A shear layer: two opposed jets with a thin gap between them.
    fn run(vorticity: f32) -> f32 {
        let mut f = Fluid::new(96, 64);
        let params = Params {
            vorticity,
            ..inert_params()
        };
        for _ in 0..90 {
            f.add_force(40.0, 28.0, 120.0, 0.0, 8.0);
            f.add_force(40.0, 36.0, -120.0, 0.0, 8.0);
            f.step(DT, &params);
        }
        total_curl(&f)
    }
    let without = run(0.0);
    let with = run(20.0);
    assert!(without > 0.0, "the shear layer produced no curl at all");
    assert!(
        with > without * 1.05,
        "confinement did not add curl: {with} vs {without}"
    );
}

#[test]
fn six_hundred_aggressive_steps_stay_finite_and_bounded() {
    let mut f = Fluid::new(128, 72);
    f.set_obstacle(&bar_mask(128, 72, 60, 64));
    let params = Params::default();
    let mut rng = Rng::new(0x51DE_5EED);
    for k in 0..600 {
        let x = rng.range(2.0, 126.0);
        let y = rng.range(2.0, 70.0);
        f.add_force(x, y, rng.signed() * 400.0, rng.signed() * 400.0, 10.0);
        f.add_dye(x, y, [1.0, 0.6, 0.2], 8.0);
        // Alternate 30 and 240 fps to exercise the whole dt range.
        f.step(if k % 2 == 0 { 1.0 / 30.0 } else { 1.0 / 240.0 }, &params);
        assert_eq!(f.sanitize(), 0, "non-finite cell appeared at step {k}");
    }
    let speed = f.max_speed();
    assert!(
        speed.is_finite() && speed < 4000.0,
        "max speed blew up: {speed}"
    );
    assert!(
        f.pressure.max_abs() < 1e6,
        "pressure drifted: {}",
        f.pressure.max_abs()
    );
    // Looser than the steady-state bound on purpose: a 400 cells/s impulse
    // splatted in the very frame being measured dominates the residual.
    assert!(
        f.last_divergence() < 0.15 * speed,
        "divergence blew up: {} vs speed {speed}",
        f.last_divergence()
    );
}

#[test]
fn extreme_dt_and_degenerate_forces_are_harmless() {
    let mut f = Fluid::new(32, 24);
    let params = Params::default();
    // Zero-size and non-finite forces must be ignored, not propagated.
    f.add_force(16.0, 12.0, 50.0, 50.0, 0.0);
    f.add_force(f32::NAN, 12.0, 50.0, 0.0, 5.0);
    f.add_force(16.0, 12.0, f32::INFINITY, 0.0, 5.0);
    f.add_dye(16.0, 12.0, [f32::NAN, 1.0, 0.0], 0.0);
    for dt in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e9, 1e-9] {
        f.step(dt, &params);
    }
    assert_eq!(f.sanitize(), 0, "degenerate input poisoned the field");
    assert!(f.last_divergence().is_finite());
}

#[test]
fn a_stray_nan_does_not_spread_through_advection() {
    let mut f = Fluid::new(32, 24);
    f.vel.u.data.fill(12.0);
    f.dye[0].data.fill(1.0);
    f.dye[0].data[12 * 32 + 16] = f32::NAN;
    f.advect(DT);
    // The extremum limiter clamps to real neighbouring values, so the NaN
    // cannot survive its own advection step.
    let bad = f.dye[0].data.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(bad, 0, "{bad} non-finite dye cells survived");
}

#[test]
fn a_degenerate_grid_does_not_panic() {
    let mut f = Fluid::new(2, 2);
    let params = Params::default();
    f.add_force(1.0, 1.0, 10.0, 10.0, 3.0);
    f.add_dye(1.0, 1.0, [1.0, 1.0, 1.0], 3.0);
    f.step(DT, &params);
    assert_eq!(f.last_divergence(), 0.0, "a 2x2 grid has no interior");
    assert_eq!(f.max_speed(), 0.0, "every cell of a 2x2 grid is a wall");
}

#[test]
fn dissipation_is_frame_rate_independent() {
    let params = Params {
        velocity_dissipation: 2.0,
        dye_dissipation: 2.0,
        vorticity: 0.0,
        viscosity: 0.0,
        pressure_iters: 1,
        ..Params::default()
    };
    // Same simulated duration, four times the frame rate.
    let mut slow = Fluid::new(32, 24);
    slow.dye[0].data.fill(1.0);
    for _ in 0..12 {
        slow.step(1.0 / 24.0, &params);
    }
    let mut fast = Fluid::new(32, 24);
    fast.dye[0].data.fill(1.0);
    for _ in 0..48 {
        fast.step(1.0 / 96.0, &params);
    }
    let (a, b) = (
        slow.dye[0].data[12 * 32 + 16],
        fast.dye[0].data[12 * 32 + 16],
    );
    assert!(
        (a - b).abs() < 1e-3,
        "dye decay differs with frame rate: {a} vs {b}"
    );
}

#[test]
fn default_viscosity_skips_diffusion_entirely() {
    assert!(
        Params::default().viscosity <= VISCOSITY_EPSILON,
        "the default viscosity should be below the skip threshold"
    );
    // A raised viscosity must actually smooth: a single spike spreads.
    let mut f = Fluid::new(32, 24);
    let i = 12 * 32 + 16;
    f.vel.u.data[i] = 100.0;
    f.diffuse(DT, 0.01);
    assert!(f.vel.u.data[i] < 100.0, "the spike did not flatten");
    assert!(
        f.vel.u.data[i + 1] > 0.1,
        "nothing diffused to the neighbour"
    );
    let moved = f.vel.u.data.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(moved > 5, "diffusion touched only {moved} cells");
}

#[test]
fn diffusion_leaves_walls_at_rest() {
    let mut f = Fluid::new(32, 24);
    f.set_obstacle(&bar_mask(32, 24, 14, 18));
    f.vel.u.data.fill(50.0);
    f.enforce_boundaries();
    f.diffuse(DT, 0.01);
    for (i, s) in f.solid.data.iter().enumerate() {
        if *s >= 0.5 {
            assert_eq!(f.vel.u.data[i], 0.0, "diffusion moved a wall cell at {i}");
        }
    }
}

// --- kernel cross-checks ---------------------------------------------

/// Deliberately naive pressure sweep, written from the 2-D indices with an
/// explicit per-neighbour wall test. Independent of the shipping kernels,
/// so agreement means something.
fn reference_pressure(p: &[f32], div: &[f32], solid: &[f32], w: usize, h: usize) -> Vec<f32> {
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

fn reference_diffuse(
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
fn kernel_fixture(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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

#[test]
fn pressure_kernel_matches_the_reference() {
    // 19 leaves a non-multiple-of-4 tail, 18 does not: both SIMD lanes and
    // the scalar tail get covered.
    for (w, h) in [(19usize, 11usize), (18, 12), (5, 5)] {
        let (p, div, solid) = kernel_fixture(w, h);
        let mut got = vec![0.0f32; w * h];
        jacobi_pressure(&p, &mut got, &div, &solid, w, h);
        let want = reference_pressure(&p, &div, &solid, w, h);
        for i in 0..w * h {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "{w}x{h} cell {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn diffusion_kernel_matches_the_reference() {
    for (w, h) in [(19usize, 11usize), (18, 12)] {
        let (rhs, x, solid) = kernel_fixture(w, h);
        let mut got = vec![0.0f32; w * h];
        jacobi_diffuse(&x, &mut got, &rhs, &solid, w, h, 0.37);
        let want = reference_diffuse(&x, &rhs, &solid, w, h, 0.37);
        for i in 0..w * h {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "{w}x{h} cell {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn dispatched_kernels_match_the_scalar_fallback() {
    // On wasm32 with `simd` this diffs SIMD128 against scalar; elsewhere it
    // pins the dispatcher to the fallback it is supposed to select.
    let (w, h) = (23usize, 13usize);
    let (p, div, solid) = kernel_fixture(w, h);
    let mut dispatched = vec![0.0f32; w * h];
    let mut scalar = vec![0.0f32; w * h];

    jacobi_pressure(&p, &mut dispatched, &div, &solid, w, h);
    jacobi_pressure_scalar(&p, &mut scalar, &div, &solid, w, h);
    for i in 0..w * h {
        assert!(
            (dispatched[i] - scalar[i]).abs() <= 1e-6 * scalar[i].abs().max(1.0),
            "pressure cell {i}: {} vs {}",
            dispatched[i],
            scalar[i]
        );
    }

    jacobi_diffuse(&p, &mut dispatched, &div, &solid, w, h, 0.25);
    jacobi_diffuse_scalar(&p, &mut scalar, &div, &solid, w, h, 0.25);
    for i in 0..w * h {
        assert!(
            (dispatched[i] - scalar[i]).abs() <= 1e-6 * scalar[i].abs().max(1.0),
            "diffusion cell {i}: {} vs {}",
            dispatched[i],
            scalar[i]
        );
    }
}

#[test]
fn kernels_refuse_undersized_buffers_instead_of_panicking() {
    let (w, h) = (8usize, 6usize);
    let short = vec![0.0f32; 4];
    let full = vec![0.0f32; w * h];
    let mut out = vec![9.0f32; w * h];
    jacobi_pressure(&short, &mut out, &full, &full, w, h);
    assert!(out.iter().all(|v| *v == 0.0));
    out.fill(9.0);
    jacobi_diffuse(&short, &mut out, &full, &full, w, h, 0.5);
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn walls_are_neumann_not_dirichlet_in_the_pressure_solve() {
    // One fluid column beside a wall: a Dirichlet (read-the-wall) update
    // would pull the pressure towards the wall's stored value instead of
    // mirroring it, so the two differ by a wide margin here.
    let (w, h) = (7usize, 5usize);
    let n = w * h;
    let mut solid = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 || x == 4 {
                solid[y * w + x] = 1.0;
            }
        }
    }
    let mut p = vec![0.0f32; n];
    // Poison the wall column: a correct solve never reads it.
    for y in 0..h {
        p[y * w + 4] = 1000.0;
    }
    let div = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    jacobi_pressure(&p, &mut out, &div, &solid, w, h);
    for y in 1..h - 1 {
        assert!(
            out[y * w + 3].abs() < 1e-6,
            "wall pressure leaked into the fluid: {}",
            out[y * w + 3]
        );
    }
}

#[test]
fn warm_started_pressure_stays_centred() {
    // Hundreds of frames of warm starting must not let the Neumann null
    // space drift the pressure off to infinity.
    let mut f = Fluid::new(64, 48);
    let params = Params::default();
    for _ in 0..200 {
        f.add_force(20.0, 24.0, 200.0, 40.0, 8.0);
        f.step(DT, &params);
    }
    let mean: f32 = f.pressure.sum() / f.pressure.len() as f32;
    assert!(mean.abs() < 1.0, "pressure mean drifted to {mean}");
    assert!(
        f.pressure.max_abs() < 1e4,
        "pressure magnitude {}",
        f.pressure.max_abs()
    );
}

// --- trace CFL ---------------------------------------------------------

#[test]
fn a_narrow_fast_impulse_actually_advects() {
    // The shape a hand swipe makes: `MIN_RADIUS`-ish and near
    // `spells::MAX_IMPULSE`. One midpoint step backtraces such a cell to
    // *itself* — the midpoint lands outside the impulse, where the fluid is
    // still — so the spike never moves and the next frame stacks onto it.
    let (w, h) = (96usize, 72usize);
    let mut f = Fluid::new(w, h);
    f.add_force(30.0, 36.0, 800.0, 0.0, 4.0);
    let peak = f.vel.u.data[36 * w + 30];
    assert!((peak - 800.0).abs() < 1.0, "fixture peak is {peak}");

    let dt = 1.0 / 30.0;
    let (tx, _) = f.integrate(30.0, 36.0, -dt, peak, 0.0);
    assert!(
        30.0 - tx > 2.0,
        "the backtrace only moved {} cells upwind of a {peak} cells/s jet",
        30.0 - tx
    );

    f.advect(dt);
    assert!(
        f.vel.u.data[36 * w + 30] < 0.5 * peak,
        "the spike stayed put: {} of {peak}",
        f.vel.u.data[36 * w + 30]
    );
}

#[test]
fn a_pinned_hard_drive_settles_at_the_same_speed_at_30_and_240_fps() {
    // The user-visible face of the trace CFL: the same impulse *rate* must
    // not reach a different speed because the frames are wider.
    fn run(dt: f32) -> f32 {
        let steps = (2.0 / dt).round() as usize;
        let mut f = Fluid::new(96, 72);
        let params = Params {
            vorticity: 0.0,
            viscosity: 0.0,
            velocity_dissipation: 0.0,
            ..Params::default()
        };
        for _ in 0..steps {
            f.add_force(30.0, 36.0, 6000.0 * dt, 0.0, 9.0);
            f.step(dt, &params);
        }
        f.max_speed()
    }
    let slow = run(1.0 / 30.0);
    let fast = run(1.0 / 240.0);
    assert!(slow > 50.0 && fast > 50.0, "the drive did nothing");
    let ratio = slow.max(fast) / slow.min(fast);
    assert!(
        ratio < 1.6,
        "30 fps settled at {slow} where 240 fps settled at {fast} (ratio {ratio})"
    );
}

#[test]
fn the_trace_reproduces_a_uniform_flow_at_every_substep_count() {
    // Subdivision must not change the answer where the single step was
    // already exact, at any speed, in either direction.
    let mut f = Fluid::new(64, 48);
    for speed in [3.0f32, 60.0, 300.0, 3000.0] {
        f.vel.u.data.fill(speed);
        f.vel.v.data.fill(-0.5 * speed);
        let dt = 1.0 / 60.0;
        let (bx, by) = f.integrate(32.0, 24.0, -dt, speed, -0.5 * speed);
        assert!(
            (bx - (32.0 - dt * speed)).abs() < 1e-2 * speed.max(1.0) * dt,
            "speed {speed}: backtrace x {bx}"
        );
        assert!(
            (by - (24.0 + dt * 0.5 * speed)).abs() < 1e-2 * speed.max(1.0) * dt,
            "speed {speed}: backtrace y {by}"
        );
        let (gx, _) = f.integrate(32.0, 24.0, dt, speed, -0.5 * speed);
        assert!(
            (gx - (32.0 + dt * speed)).abs() < 1e-2 * speed.max(1.0) * dt,
            "speed {speed}: forward x {gx}"
        );
    }
}

#[test]
fn the_trace_survives_a_poisoned_velocity() {
    let mut f = Fluid::new(32, 24);
    f.vel.u.data.fill(5.0);
    for (u0, v0) in [
        (f32::NAN, 0.0),
        (f32::INFINITY, f32::NEG_INFINITY),
        (0.0, f32::NAN),
        (1e38, 1e38),
    ] {
        let (x, y) = f.integrate(16.0, 12.0, -1.0 / 60.0, u0, v0);
        // Must terminate and must not be usable as an out-of-range index;
        // `TraceMap::set` clamps, so NaN is acceptable, a hang is not.
        assert!(x.is_nan() || (-1e9..1e9).contains(&x), "x {x}");
        assert!(y.is_nan() || (-1e9..1e9).contains(&y), "y {y}");
    }
}

// --- boundaries --------------------------------------------------------

#[test]
fn an_enclosed_cavity_stays_at_rest() {
    // Stronger than the full-height-bar leak test: the far side there is
    // shielded by the domain border as well, so it can only catch a gross
    // leak. Here the only thing between a violent drive and the cavity is
    // the solver's own wall treatment, on all four sides and on corners.
    let (w, h) = (96usize, 72usize);
    let mut f = Fluid::new(w, h);
    let mut m = Grid::new(w, h);
    for y in 28..=48 {
        for x in 40..=60 {
            if x == 40 || x == 60 || y == 28 || y == 48 {
                m.data[y * w + x] = 1.0;
            }
        }
    }
    f.set_obstacle(&m);
    let params = Params::default();
    for _ in 0..300 {
        f.add_force(20.0, 38.0, 400.0, 30.0, 9.0);
        f.add_dye(20.0, 38.0, [1.0, 0.0, 0.0], 7.0);
        f.step(DT, &params);
    }
    assert!(f.max_speed() > 10.0, "the drive did nothing");
    for y in 29..48 {
        for x in 41..60 {
            let i = y * w + x;
            assert!(f.solid.data[i] < 0.5, "cavity cell {x},{y} is solid");
            assert_eq!(f.vel.u.data[i], 0.0, "u leaked into the cavity at {x},{y}");
            assert_eq!(f.vel.v.data[i], 0.0, "v leaked into the cavity at {x},{y}");
            assert_eq!(f.dye[0].data[i], 0.0, "dye leaked in at {x},{y}");
        }
    }
}

#[test]
fn a_one_cell_thick_wall_still_seals() {
    // A one-cell wall is the case a face-based no-penetration treatment
    // gets wrong: both of its faces belong to fluid cells on opposite
    // sides, so neither is closed unless the wall cell's own stored
    // velocity is also trusted to be zero.
    let (w, h) = (64usize, 48usize);
    let mut f = Fluid::new(w, h);
    let mut m = Grid::new(w, h);
    for y in 0..h {
        m.data[y * w + 32] = 1.0;
    }
    f.set_obstacle(&m);
    let params = Params::default();
    for _ in 0..200 {
        f.add_force(12.0, 24.0, 320.0, 0.0, 8.0);
        f.add_dye(12.0, 24.0, [1.0, 0.0, 0.0], 6.0);
        f.step(DT, &params);
    }
    assert!(f.max_speed() > 10.0, "the push never took");
    for y in 0..h {
        for x in 33..w {
            let i = y * w + x;
            assert_eq!(f.vel.u.data[i], 0.0, "u crossed the wall at {x},{y}");
            assert_eq!(f.dye[0].data[i], 0.0, "dye crossed the wall at {x},{y}");
        }
    }
}

// --- projection --------------------------------------------------------

#[test]
fn the_projection_converges_on_a_smooth_divergence_blob() {
    // The white-noise fixture is the easy case for Jacobi — high-frequency
    // error is exactly what it kills fastest. A smooth, wide divergence
    // blob is the shape a real impulse makes, and it is also the case an
    // *inconsistent* divergence/gradient pair would plateau on instead of
    // converging. The bound is loose; the point is that it keeps falling.
    let (w, h) = (65usize, 65usize);
    let mut f = Fluid::new(w, h);
    let c = 32.0;
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            let r = (dx * dx + dy * dy).sqrt();
            if r > 0.5 && r < 14.0 {
                let i = y * w + x;
                f.vel.u.data[i] = 30.0 * dx / r;
                f.vel.v.data[i] = 30.0 * dy / r;
            }
        }
    }
    f.enforce_boundaries();
    f.close_solid_faces();
    let start = f.max_divergence();
    assert!(start > 20.0, "fixture was not divergent: {start}");

    f.project(200);
    let mid = f.max_divergence();
    for _ in 0..12 {
        f.project(200);
    }
    let end = f.max_divergence();
    assert!(mid < 0.1 * start, "200 sweeps left {mid} of {start}");
    assert!(
        end < 0.02 * mid,
        "the solve plateaued at {end} instead of converging (was {mid})"
    );
}

#[test]
fn a_still_field_stays_still() {
    let mut f = Fluid::new(64, 48);
    let params = Params::default();
    for _ in 0..10 {
        f.step(DT, &params);
    }
    assert_eq!(
        f.max_speed(),
        0.0,
        "the solver invented motion from nothing"
    );
    assert_eq!(f.last_divergence(), 0.0);
}
