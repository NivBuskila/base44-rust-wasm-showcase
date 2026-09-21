//! Advection and its trace: a uniform flow survives, a blob rides at `v * dt`, dye mass is conserved, and nothing is ever fetched from inside an obstacle.

use super::support::*;

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
