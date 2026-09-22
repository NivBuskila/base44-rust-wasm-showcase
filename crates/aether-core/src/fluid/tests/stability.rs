//! What must hold over long hostile runs: finite and bounded, frame-rate independent dissipation, and diffusion that respects the walls.

use super::support::*;

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
