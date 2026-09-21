//! Per-particle heat: it decays, motion earns it back, and it never leaves `[0, 1]`.

use super::support::*;

#[test]
fn heat_decays_toward_zero() {
    let (w, h) = (16, 16);
    let vel = const_field(w, h, 0.0, 0.0);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 100.0, 0.0);
    let p_cfg = cfg(0.0, 0.0, 0.6);

    let mut p = Particles::new(1, 21);
    p.set_active(1);
    p.place(0, 8.0, 8.0, 0.0, 0.0, 100.0);
    p.impulse(8.0, 8.0, 4.0, 0.0, 1.0);
    assert!((p.heat_of(0) - 1.0).abs() < 1e-6);

    let mut prev = p.heat_of(0);
    for _ in 0..60 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        let now = p.heat_of(0);
        assert!(now < prev, "heat did not decay: {now} >= {prev}");
        prev = now;
    }
    let analytic = (-HEAT_DECAY).exp();
    assert!((prev - analytic).abs() < 1e-3, "{prev} vs {analytic}");
    assert!(prev > 0.0);
}

#[test]
fn a_fast_flow_makes_particles_glow() {
    // The counterpart to `heat_decays_toward_zero`: heat is the colour-ramp
    // coordinate, so a particle the fluid is carrying quickly has to earn
    // colour on its own. Without this, the only coloured particles are the
    // ones a spell touched, and the flow renders as near-black filaments.
    //
    // The grid is wide enough that neither particle reaches the far edge in
    // the measured second. A narrow one recycles the fast particle
    // mid-test, and respawn resets heat — which is how this test failed
    // the first time, reporting 0 for a particle that was in fact glowing.
    let (w, h) = (256, 32);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 100.0, 0.0);
    let p_cfg = cfg(0.0, 0.0, 0.6);
    let steps = 60;

    let run = |speed: f32, seed: u64| {
        let field = const_field(w, h, speed, 0.0);
        let mut p = Particles::new(1, seed);
        p.set_active(1);
        p.place(0, 1.0, 16.0, 0.0, 0.0, 100.0);
        for _ in 0..steps {
            p.step(&field, &obs, DT, &p_params, &p_cfg);
        }
        // Confirm it never recycled, or the heat reading means nothing.
        assert!(p.position(0).0 > 1.0, "particle did not move");
        p.heat_of(0)
    };

    let hot = run(HEAT_SPEED_FULL, 31);
    let warm = run(HEAT_SPEED_FULL * 0.25, 32);

    // Well into the hot half of the ramp. Not tighter than that: the
    // actual figure depends on how far the rise converges in the measured
    // second and on the particle's own velocity relaxing, and pinning the
    // bar to the current output would make this a test of today's
    // arithmetic rather than of the intended behaviour.
    assert!(
        hot > 0.6,
        "a particle at the full-heat speed stayed cold: {hot}"
    );
    // A quarter of the speed must read as distinctly cooler, or the ramp
    // carries no information and every moving particle is one colour.
    assert!(
        warm < hot - 0.4,
        "heat does not discriminate speed: {warm} vs {hot}"
    );
    assert!(
        warm > 0.0,
        "a moving particle should have some heat: {warm}"
    );
}

#[test]
fn heat_never_leaves_the_unit_interval() {
    // The renderer indexes a colour ramp with this, so a value outside
    // [0, 1] is a garbage pixel. An absurd field is the stress case.
    let (w, h) = (32, 32);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 100.0, 0.0);
    let p_cfg = cfg(8.0, 0.0, 0.6);
    let violent = const_field(w, h, 50_000.0, -50_000.0);

    let mut p = Particles::new(64, 33);
    p.set_active(64);
    p.seed_uniform(w, h, 100.0);
    p.impulse(16.0, 16.0, 40.0, 5000.0, 1.0);
    for _ in 0..120 {
        p.step(&violent, &obs, DT, &p_params, &p_cfg);
    }
    for i in 0..64 {
        let heat = p.heat_of(i);
        assert!(
            (0.0..=1.0).contains(&heat),
            "particle {i} heat {heat} outside the colour ramp"
        );
    }
}
