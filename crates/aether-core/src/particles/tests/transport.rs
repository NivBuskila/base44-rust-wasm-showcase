//! Transport: that a particle goes where the field says, at the rate the field says, on a curve rather than a chord, and identically at any frame rate.

use super::support::*;

#[test]
fn constant_field_transports_at_the_field_velocity() {
    let vel = const_field(64, 32, 5.0, 0.0);
    let obs = Grid::new(64, 32);
    let (p_params, p_cfg) = pure_advection();
    let mut p = Particles::new(4, 1);
    p.set_active(1);
    p.place(0, 10.0, 16.0, 0.0, 0.0, 100.0);

    let steps = 60;
    for _ in 0..steps {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
    }

    let (x, y) = p.position(0);
    let expect = 10.0 + 5.0 * DT * steps as f32;
    assert!(
        (x - expect).abs() < 1e-3,
        "x drifted: {x} vs {expect} (sign or scale error)"
    );
    assert!((y - 16.0).abs() < 1e-4, "y should not have moved: {y}");
    assert_eq!(p.alive(), 1);
}

#[test]
fn negative_field_transports_left_and_down() {
    let vel = const_field(64, 64, -3.0, -4.0);
    let obs = Grid::new(64, 64);
    let (p_params, p_cfg) = pure_advection();
    let mut p = Particles::new(1, 2);
    p.set_active(1);
    p.place(0, 40.0, 40.0, 0.0, 0.0, 100.0);
    for _ in 0..30 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
    }
    let (x, y) = p.position(0);
    assert!((x - (40.0 - 3.0 * DT * 30.0)).abs() < 1e-3, "x {x}");
    assert!((y - (40.0 - 4.0 * DT * 30.0)).abs() < 1e-3, "y {y}");
}

#[test]
fn rk2_tracks_the_analytic_circle_far_better_than_euler() {
    let (w, h) = (128, 128);
    let (cx, cy) = (64.0, 64.0);
    let omega = 1.0;
    let radius = 30.0;
    let vel = rot_field(w, h, cx, cy, omega);
    let obs = Grid::new(w, h);
    let (p_params, p_cfg) = pure_advection();

    let mut p = Particles::new(1, 3);
    p.set_active(1);
    p.place(0, cx + radius, cy, 0.0, 0.0, 1000.0);

    // The same field integrated with one Euler step per frame, as the
    // reference for "what the naive integrator would have done".
    let (mut ex, mut ey) = (cx + radius, cy);

    let steps = 600;
    for _ in 0..steps {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        let (eu, ev) = (-omega * (ey - cy), omega * (ex - cx));
        ex += DT * eu;
        ey += DT * ev;
    }

    let (x, y) = p.position(0);
    let r_rk2 = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
    let r_euler = ((ex - cx).powi(2) + (ey - cy).powi(2)).sqrt();

    // RK2's radius error is O(dt^4) per step, Euler's is O(dt^2).
    assert!(
        (r_rk2 - radius).abs() < 0.01,
        "RK2 radius drifted {} cells",
        r_rk2 - radius
    );
    assert!(
        r_euler - radius > 1.0,
        "reference Euler should have visibly spiralled out, drift {}",
        r_euler - radius
    );

    // Phase must also track: total rotation is omega * t radians.
    let want = omega * DT * steps as f32;
    let got = (y - cy).atan2(x - cx);
    let err = ((got - want) + core::f32::consts::PI).rem_euclid(core::f32::consts::TAU)
        - core::f32::consts::PI;
    assert!(err.abs() < 0.01, "phase error {err} rad");
}

#[test]
fn curl_swirl_turns_the_particle_with_the_vortex() {
    let (w, h) = (64, 64);
    let (cx, cy) = (32.0, 32.0);
    let omega = 3.0;
    let vel = rot_field(w, h, cx, cy, omega);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 10.0, 0.0);

    // At the centre the fluid velocity is zero, so the only thing that can
    // deflect a radially moving particle is the curl swirl.
    let mut with = Particles::new(1, 4);
    with.set_active(1);
    with.place(0, cx, cy, 5.0, 0.0, 100.0);
    with.step(&vel, &obs, DT, &p_params, &cfg(2.0, 0.0, 0.6));

    let (_, vy) = with.velocity(0);
    // curl = dv/dx - du/dy = 2 * omega for a rigid rotation.
    let expect = 2.0 * omega * 2.0 * DT;
    assert!(
        (vy - expect).abs() < 1e-4,
        "swirl magnitude {vy} vs {expect}"
    );
    assert!(vy > 0.0, "swirl must follow the vortex sense");

    let mut without = Particles::new(1, 4);
    without.set_active(1);
    without.place(0, cx, cy, 5.0, 0.0, 100.0);
    without.step(&vel, &obs, DT, &p_params, &cfg(0.0, 0.0, 0.6));
    assert_eq!(
        without.velocity(0).1,
        0.0,
        "curl_influence 0 must not swirl"
    );
}

#[test]
fn own_velocity_decay_is_framerate_independent() {
    // drag 0 removes the fluid entirely, leaving only the damping term.
    let vel = const_field(32, 32, 0.0, 0.0);
    let obs = Grid::new(32, 32);
    let p_params = params(0.0, 10.0, 0.0);
    let p_cfg = cfg(0.0, 0.0, 1.5);

    let run = |dt: f32, steps: usize| {
        let mut p = Particles::new(1, 5);
        p.set_active(1);
        p.place(0, 16.0, 16.0, 10.0, 0.0, 100.0);
        for _ in 0..steps {
            p.step(&vel, &obs, dt, &p_params, &p_cfg);
        }
        p.velocity(0).0
    };

    let slow = run(1.0 / 30.0, 30);
    let fast = run(1.0 / 240.0, 240);
    let analytic = 10.0 * (-1.5f32).exp();
    assert!((slow - fast).abs() < 1e-3, "{slow} vs {fast}");
    assert!((slow - analytic).abs() < 1e-3, "{slow} vs {analytic}");
}

#[test]
fn transport_scales_linearly_with_particle_drag() {
    // Every other advection test runs at drag 1.0, where a factor of one is
    // invisible. `particle_drag` is a HUD slider over 0..4, so the multiplier
    // has to be exact across the range, not just non-zero.
    let vel = const_field(64, 64, 10.0, 0.0);
    let obs = Grid::new(64, 64);
    for drag in [0.0, 0.5, 1.0, 2.0] {
        let p_params = params(drag, 10.0, 0.0);
        let mut p = Particles::new(1, 1);
        p.set_active(1);
        p.place(0, 5.0, 32.0, 0.0, 0.0, 100.0);
        for _ in 0..60 {
            p.step(&vel, &obs, DT, &p_params, &cfg(0.0, 0.0, 0.6));
        }
        let want = 5.0 + drag * 10.0 * DT * 60.0;
        let got = p.position(0).0;
        assert!((got - want).abs() < 1e-3, "drag {drag}: {got} vs {want}");
    }
}

#[test]
fn the_whole_step_agrees_at_30_and_144_fps() {
    // `own_velocity_decay_is_framerate_independent` isolates the damping with
    // the fluid switched off. This is the composite: advection, curl swirl and
    // damping together at the shipped defaults, which is the only combination
    // a player ever sees.
    let (w, h) = (256, 144);
    let mut vel = VecField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - 128.0, y as f32 - 72.0);
            vel.u.set(x, y, -0.35 * dy + 6.0);
            vel.v.set(x, y, 0.35 * dx);
        }
    }
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 10.0, 0.0);
    let p_cfg = ParticleConfig::default();
    let run = |dt: f32, steps: usize| {
        let mut p = Particles::new(1, 3);
        p.set_active(1);
        p.place(0, 100.0, 40.0, 0.0, 0.0, 100.0);
        for _ in 0..steps {
            p.step(&vel, &obs, dt, &p_params, &p_cfg);
        }
        p.position(0)
    };
    // One second of travel is ~19 cells here, so the tolerance is 0.3% of
    // the path: a swirl or gravity term missing its `dt`, or an advection
    // step that integrated per frame instead of per second, lands orders of
    // magnitude outside it. (The damping form is pinned separately, by
    // `own_velocity_decay_is_framerate_independent`, which isolates it.)
    let (ax, ay) = run(1.0 / 30.0, 30);
    let (bx, by) = run(1.0 / 144.0, 144);
    assert!(
        (ax - bx).abs() < 0.05 && (ay - by).abs() < 0.05,
        "30 fps ({ax}, {ay}) vs 144 fps ({bx}, {by})"
    );
}
