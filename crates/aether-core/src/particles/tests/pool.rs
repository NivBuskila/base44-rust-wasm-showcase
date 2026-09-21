//! The pool as a population: the alive count, the respawn budget, staggered lifetimes, the spawn/impulse forces and bit-for-bit determinism.

use super::support::*;

#[test]
fn alive_count_is_stable_over_thousands_of_steps() {
    let (w, h) = (64, 36);
    let vel = const_field(w, h, 2.0, -1.0);
    let obs = Grid::new(w, h);
    let n = 2_000;
    // Steady-state death rate is n / mean_life = 2000/s; give the budget
    // some headroom and the population should sit at the pool size.
    let p_params = params(1.0, 1.0, 3_000.0);
    let p_cfg = cfg(2.2, 0.0, 0.6);

    let mut p = Particles::new(n, 9);
    p.set_active(n);
    p.seed_uniform(w, h, 1.0);

    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for s in 0..6_000 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        // Skip the first second: the seeded generation dies off on its own
        // schedule before the respawn loop reaches equilibrium.
        if s > 60 {
            lo = lo.min(p.alive());
            hi = hi.max(p.alive());
        }
    }
    assert!(lo > (n as f32 * 0.9) as usize, "pool died down to {lo}");
    assert!(hi <= n, "more alive than active: {hi}");
}

#[test]
fn spawn_rate_is_a_budget_not_a_suggestion() {
    let (w, h) = (32, 32);
    let vel = const_field(w, h, 0.0, 0.0);
    let obs = Grid::new(w, h);
    let n = 2_000;
    // 100 respawns/s against a 1 s lifetime can only sustain ~100 alive.
    let p_params = params(1.0, 1.0, 100.0);
    let p_cfg = cfg(0.0, 0.0, 0.6);

    let mut p = Particles::new(n, 10);
    p.set_active(n);
    p.seed_uniform(w, h, 1.0);
    for _ in 0..600 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
    }
    let alive = p.alive();
    assert!(
        alive > 20,
        "the budget starved the pool completely: {alive}"
    );
    assert!(
        alive < 400,
        "spawn_rate was ignored: {alive} alive on a 100/s budget"
    );

    // With no budget at all the pool must be allowed to empty.
    let mut dead = Particles::new(64, 11);
    dead.set_active(64);
    dead.seed_uniform(w, h, 0.5);
    let starved = params(1.0, 1.0, 0.0);
    for _ in 0..120 {
        dead.step(&vel, &obs, DT, &starved, &p_cfg);
    }
    assert_eq!(dead.alive(), 0);
}

#[test]
fn lifetimes_are_staggered_after_seeding_and_after_respawn() {
    let (w, h) = (64, 36);
    let vel = const_field(w, h, 0.0, 0.0);
    let obs = Grid::new(w, h);
    let n = 600;

    let spread = |p: &Particles| {
        let lives: Vec<f32> = (0..p.active()).map(|i| p.life_of(i)).collect();
        let mean = lives.iter().sum::<f32>() / lives.len() as f32;
        let var = lives.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / lives.len() as f32;
        (mean, var.sqrt())
    };

    let mut p = Particles::new(n, 12);
    p.set_active(n);
    p.seed_uniform(w, h, 2.0);
    let (mean, sd) = spread(&p);
    assert!(sd > 0.2 * mean, "seeded lifetimes are bunched: sd {sd}");

    // Now force the whole pool onto one clock and check the respawn path
    // breaks it up again.
    for i in 0..n {
        p.place(i, 10.0, 10.0, 0.0, 0.0, 0.5);
    }
    let p_params = params(1.0, 2.0, 100_000.0);
    let p_cfg = cfg(0.0, 0.0, 0.6);
    for _ in 0..40 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
    }
    let (mean2, sd2) = spread(&p);
    assert_eq!(p.alive(), n, "generous budget should refill the pool");
    assert!(
        sd2 > 0.1 * mean2,
        "respawn put the pool back in lockstep: sd {sd2}, mean {mean2}"
    );
    // Positions must be scattered too, not all sitting on (10, 10).
    let distinct = (0..n).filter(|&i| p.position(i).0 != 10.0).count();
    assert!(distinct > n - 10, "respawn did not scatter: {distinct}");
}

#[test]
fn spawn_burst_is_local_outward_and_capacity_safe() {
    let mut p = Particles::new(64, 13);
    p.set_active(64);
    let (cx, cy) = (30.0, 20.0);
    // Ask for more than the pool holds: must wrap, not panic.
    p.spawn_burst(cx, cy, 500, 40.0, 0.8, 1.5);

    for i in 0..p.active() {
        let (x, y) = p.position(i);
        let (vx, vy) = p.velocity(i);
        let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        assert!(d <= BURST_SCATTER + 1e-4, "particle {i} landed {d} away");
        // Placed along its own velocity, so the offset is strictly outward.
        let dot = (x - cx) * vx + (y - cy) * vy;
        assert!(dot >= 0.0, "particle {i} moves inward: {dot}");
        assert!(p.life_of(i) > 0.0);
        assert!((p.heat_of(i) - 0.8).abs() < 1e-6);
    }
    let speeds: Vec<f32> = (0..p.active())
        .map(|i| {
            let (vx, vy) = p.velocity(i);
            (vx * vx + vy * vy).sqrt()
        })
        .collect();
    assert!(speeds.iter().any(|&s| s > 1.0), "burst has no speed at all");
    assert!(
        speeds.iter().all(|&s| s <= 40.0 + 1e-3),
        "burst exceeded the requested speed"
    );

    // Degenerate input must be ignored, not propagated.
    p.spawn_burst(f32::NAN, cy, 10, 5.0, 0.5, 1.0);
    p.spawn_burst(cx, cy, 10, f32::INFINITY, 0.5, 1.0);
    for i in 0..p.active() {
        let (vx, vy) = p.velocity(i);
        assert!(vx.is_finite() && vy.is_finite());
        assert!(p.position(i).0.is_finite());
    }
}

#[test]
fn impulse_pushes_near_particles_out_and_leaves_far_ones_alone() {
    let mut p = Particles::new(4, 14);
    p.set_active(4);
    let (cx, cy) = (20.0, 20.0);
    p.place(0, 22.0, 20.0, 0.0, 0.0, 5.0); // right, close
    p.place(1, 20.0, 14.0, 0.0, 0.0, 5.0); // below, near the edge
    p.place(2, 40.0, 20.0, 0.0, 0.0, 5.0); // far outside
    p.place(3, 17.0, 17.0, 0.0, 0.0, 5.0); // diagonal, close
    p.impulse(cx, cy, 8.0, 50.0, 0.9);

    for i in [0usize, 1, 3] {
        let (x, y) = p.position(i);
        let (vx, vy) = p.velocity(i);
        let dot = (x - cx) * vx + (y - cy) * vy;
        assert!(dot > 0.0, "particle {i} was not pushed outward: {dot}");
        assert!(p.heat_of(i) > 0.0, "particle {i} got no heat");
    }
    assert_eq!(p.velocity(2), (0.0, 0.0), "distant particle was disturbed");
    assert_eq!(p.heat_of(2), 0.0);

    // Falloff must be monotonic in distance and reach zero at the edge.
    let near = p.velocity(0);
    let edge = p.velocity(1);
    let near_mag = (near.0 * near.0 + near.1 * near.1).sqrt();
    let edge_mag = (edge.0 * edge.0 + edge.1 * edge.1).sqrt();
    assert!(
        near_mag > edge_mag,
        "falloff is not monotonic: {near_mag} at r=2 vs {edge_mag} at r=6"
    );

    let mut boundary = Particles::new(1, 15);
    boundary.set_active(1);
    boundary.place(0, cx + 8.0, cy, 0.0, 0.0, 5.0);
    boundary.impulse(cx, cy, 8.0, 50.0, 0.9);
    assert_eq!(
        boundary.velocity(0),
        (0.0, 0.0),
        "a hard cutoff at the edge leaves a visible ring"
    );

    // Degenerate radius / non-finite centre must be no-ops.
    let before = p.velocity(0);
    p.impulse(cx, cy, 0.0, 50.0, 0.5);
    p.impulse(f32::NAN, cy, 8.0, 50.0, 0.5);
    p.impulse(cx, cy, 8.0, f32::NAN, 0.5);
    assert_eq!(p.velocity(0), before);
}

#[test]
fn identical_seeds_and_inputs_produce_identical_buffers() {
    let (w, h) = (48, 32);
    let vel = const_field(w, h, 7.0, -3.0);
    let mut obs = Grid::new(w, h);
    for y in 12..20 {
        for x in 12..20 {
            obs.set(x, y, 1.0);
        }
    }
    let p_params = params(1.0, 0.8, 40_000.0);
    let p_cfg = cfg(2.2, 3.0, 0.6);

    let run = || {
        let mut p = Particles::new(300, 0xDEAD_BEEF);
        p.set_active(300);
        p.seed_uniform(w, h, 0.8);
        for s in 0..200 {
            if s == 50 {
                p.spawn_burst(24.0, 16.0, 40, 30.0, 1.0, 0.6);
            }
            if s == 90 {
                p.impulse(24.0, 16.0, 10.0, 25.0, 0.7);
            }
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }
        p.build_render_buffer(w, h);
        p.render_buffer().to_vec()
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "same seed and inputs diverged");
    assert_eq!(a.len(), 300 * PARTICLE_STRIDE);
}

#[test]
fn empty_pools_and_absurd_timesteps_are_survivable() {
    let (w, h) = (16, 16);
    let vel = const_field(w, h, 3.0, 3.0);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 1.0, 1_000.0);
    let p_cfg = cfg(2.2, 10.0, 0.6);

    // Zero-capacity pool: every entry point must be a no-op.
    let mut empty = Particles::new(0, 16);
    empty.set_active(100);
    assert_eq!(empty.active(), 0);
    empty.seed_uniform(w, h, 1.0);
    empty.spawn_burst(4.0, 4.0, 10, 5.0, 1.0, 1.0);
    empty.impulse(4.0, 4.0, 5.0, 5.0, 1.0);
    empty.step(&vel, &obs, DT, &p_params, &p_cfg);
    empty.build_render_buffer(w, h);
    assert_eq!(empty.alive(), 0);
    assert!(empty.render_buffer().is_empty());

    // Deactivated pool: nothing moves, nothing panics.
    let mut idle = Particles::new(8, 17);
    idle.set_active(8);
    idle.seed_uniform(w, h, 1.0);
    let before = idle.position(0);
    idle.set_active(0);
    idle.step(&vel, &obs, DT, &p_params, &p_cfg);
    assert_eq!(idle.alive(), 0);
    assert_eq!(idle.position(0), before);
    assert!(idle.render_buffer().is_empty());

    // Absurd timesteps.
    let mut p = Particles::new(64, 18);
    p.set_active(64);
    p.seed_uniform(w, h, 1.0);
    for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e9, 1.0 / 480.0] {
        p.step(&vel, &obs, dt, &p_params, &p_cfg);
        for i in 0..p.active() {
            let (x, y) = p.position(i);
            assert!(
                (0.0..=(w - 1) as f32).contains(&x) && (0.0..=(h - 1) as f32).contains(&y),
                "dt {dt} moved particle {i} to ({x}, {y})"
            );
            assert!(p.life_of(i).is_finite(), "dt {dt} broke life");
        }
    }
    // A NaN dt must not be read as "everything died".
    let mut nan_dt = Particles::new(32, 19);
    nan_dt.set_active(32);
    nan_dt.seed_uniform(w, h, 5.0);
    nan_dt.step(&vel, &obs, f32::NAN, &p_params, &p_cfg);
    assert_eq!(nan_dt.alive(), 32, "a NaN dt wiped the pool");
}
