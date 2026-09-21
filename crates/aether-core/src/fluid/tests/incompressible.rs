//! The projection: divergence removed, walls Neumann rather than Dirichlet, a warm start that does not drift and a cavity that stays at rest.

use super::support::*;

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
