//! The invariants that must hold whatever the input: never inside an obstacle, never off the grid, never non-finite, and the obstacle test agreeing with the solver's own threshold at any resolution.

use super::support::*;

#[test]
fn live_particles_never_end_a_step_inside_an_obstacle() {
    let (w, h) = (64, 64);
    let vel = const_field(w, h, 20.0, 6.0);
    let mut obs = Grid::new(w, h);
    for y in 20..44 {
        for x in 20..44 {
            obs.set(x, y, 1.0);
        }
    }
    let p_params = params(1.0, 3.0, 50_000.0);
    let p_cfg = cfg(2.2, 0.0, 0.6);

    let mut p = Particles::new(2_000, 6);
    p.set_active(2_000);
    p.seed_uniform(w, h, 3.0);

    for _ in 0..400 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        for i in 0..p.active() {
            if p.life_of(i) <= 0.0 {
                continue;
            }
            let (x, y) = p.position(i);
            let xi = (x + 0.5) as usize;
            let yi = (y + 0.5) as usize;
            assert!(
                obs.get(xi, yi) < 0.5,
                "live particle {i} inside the obstacle at ({x}, {y})"
            );
        }
    }
    assert!(p.alive() > 1_000, "the pool collapsed: {}", p.alive());
}

#[test]
fn no_particle_ever_ends_a_step_outside_the_grid() {
    let (w, h) = (48, 24);
    // Strong outward flow plus gravity: everything is pushed at the walls.
    let mut vel = VecField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            vel.u.set(x, y, (x as f32 - 24.0) * 8.0);
            vel.v.set(x, y, (y as f32 - 12.0) * 8.0);
        }
    }
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 2.0, 30_000.0);
    let p_cfg = cfg(4.0, 60.0, 0.6);

    let mut p = Particles::new(1_500, 7);
    p.set_active(1_500);
    p.seed_uniform(w, h, 2.0);

    for _ in 0..300 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        for i in 0..p.active() {
            let (x, y) = p.position(i);
            assert!(
                (0.0..=(w - 1) as f32).contains(&x) && (0.0..=(h - 1) as f32).contains(&y),
                "particle {i} escaped to ({x}, {y})"
            );
        }
    }
    // And the normalised buffer stays in [0, 1], which is what the shader
    // assumes.
    p.build_render_buffer(w, h);
    for q in p.render_buffer().chunks(PARTICLE_STRIDE) {
        assert!((0.0..=1.0).contains(&q[0]) && (0.0..=1.0).contains(&q[1]));
        assert!((0.0..=1.0).contains(&q[3]));
    }
}

#[test]
fn everything_stays_finite_under_a_violent_field() {
    let (w, h) = (32, 32);
    let mut vel = VecField::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let s = if (x + y) % 2 == 0 { 9_000.0 } else { -9_000.0 };
            vel.u.set(x, y, s);
            vel.v.set(x, y, -s);
        }
    }
    let mut obs = Grid::new(w, h);
    for y in 10..22 {
        for x in 10..22 {
            obs.set(x, y, 1.0);
        }
    }
    let p_params = params(4.0, 1.0, 200_000.0);
    let p_cfg = cfg(20.0, 5_000.0, 0.6);

    let mut p = Particles::new(512, 8);
    p.set_active(512);
    p.seed_uniform(w, h, 1.0);

    for _ in 0..1_000 {
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
    }
    for i in 0..p.active() {
        let (x, y) = p.position(i);
        let (vx, vy) = p.velocity(i);
        assert!(
            x.is_finite() && y.is_finite() && vx.is_finite() && vy.is_finite(),
            "particle {i} went non-finite: ({x}, {y}) ({vx}, {vy})"
        );
        assert!(p.life_of(i).is_finite() && p.heat_of(i).is_finite());
    }
    p.build_render_buffer(w, h);
    assert!(p.render_buffer().iter().all(|v| v.is_finite()));
}

#[test]
fn a_particle_placed_at_nan_is_culled_not_propagated() {
    let (w, h) = (16, 16);
    let vel = const_field(w, h, 1.0, 1.0);
    let obs = Grid::new(w, h);
    let p_params = params(1.0, 1.0, 10_000.0);
    let p_cfg = cfg(2.2, 0.0, 0.6);

    let mut p = Particles::new(2, 20);
    p.set_active(2);
    p.place(0, f32::NAN, 8.0, 0.0, 0.0, 5.0);
    p.place(1, 8.0, 8.0, f32::INFINITY, 0.0, 5.0);
    p.step(&vel, &obs, DT, &p_params, &p_cfg);
    for i in 0..2 {
        let (x, y) = p.position(i);
        let (vx, vy) = p.velocity(i);
        assert!(x.is_finite() && y.is_finite(), "({x}, {y})");
        assert!(vx.is_finite() && vy.is_finite(), "({vx}, {vy})");
        assert!((0.0..=(w - 1) as f32).contains(&x));
    }
}

#[test]
fn garbage_fields_and_parameters_keep_every_invariant() {
    // The fluid sanitises itself once every few frames, so `step` can and
    // does get handed a field with a NaN or an infinity in it. None of the
    // per-particle state may inherit it, and `alive()` must keep agreeing
    // with the number of particles that actually have life left.
    let mut r = Rng::new(0xF00D);
    let (w, h) = (40, 24);
    for round in 0..120 {
        let mut vel = VecField::new(w, h);
        for i in 0..w * h {
            let pick = r.next_f32();
            vel.u.data[i] = match pick {
                p if p < 0.02 => f32::NAN,
                p if p < 0.04 => f32::INFINITY,
                p if p < 0.06 => f32::NEG_INFINITY,
                _ => r.range(-400.0, 400.0),
            };
            vel.v.data[i] = r.range(-400.0, 400.0);
        }
        let mut obs = Grid::new(w, h);
        for i in 0..w * h {
            obs.data[i] = if r.next_f32() < 0.3 { 1.0 } else { 0.0 };
        }
        let p_params = Params {
            particle_drag: r.range(-1.0, 6.0),
            particle_life: r.range(-1.0, 40.0),
            // Every third round starves the respawn budget, which is the
            // only way to exercise the "park it in bounds" branch.
            spawn_rate: if round % 3 == 0 {
                0.0
            } else {
                r.range(-100.0, 90_000.0)
            },
            ..Params::default()
        };
        let p_cfg = ParticleConfig {
            curl_influence: r.range(-20.0, 20.0),
            gravity: r.range(-300.0, 300.0),
            damping: r.range(-2.0, 25.0),
        };
        let mut p = Particles::new(120, round as u64);
        p.set_active(120);
        p.seed_uniform(w, h, 2.0);

        for s in 0..12 {
            let dt = match s % 7 {
                0 => f32::NAN,
                1 => 1e9,
                2 => -0.01,
                3 => 0.0,
                _ => r.range(1.0 / 480.0, 1.0 / 20.0),
            };
            if s % 5 == 0 {
                p.spawn_burst(
                    r.range(-5.0, 45.0),
                    r.range(-5.0, 30.0),
                    20,
                    r.range(-80.0, 80.0),
                    r.range(-0.5, 1.5),
                    r.range(-1.0, 3.0),
                );
            }
            if s % 4 == 0 {
                p.impulse(
                    r.range(-5.0, 45.0),
                    r.range(-5.0, 30.0),
                    r.range(-2.0, 30.0),
                    r.range(-200.0, 200.0),
                    r.range(-0.5, 1.5),
                );
            }
            p.step(&vel, &obs, dt, &p_params, &p_cfg);

            let counted = (0..p.active()).filter(|&i| p.life_of(i) > 0.0).count();
            assert_eq!(counted, p.alive(), "round {round} step {s}");
            for i in 0..p.active() {
                let (x, y) = p.position(i);
                let (vx, vy) = p.velocity(i);
                assert!(
                    x.is_finite() && y.is_finite() && vx.is_finite() && vy.is_finite(),
                    "round {round} step {s} particle {i}: ({x}, {y}) ({vx}, {vy})"
                );
                assert!(
                    (0.0..=(w - 1) as f32).contains(&x) && (0.0..=(h - 1) as f32).contains(&y),
                    "round {round} step {s} particle {i} escaped to ({x}, {y})"
                );
                let heat = p.heat_of(i);
                assert!((0.0..=1.0).contains(&heat), "heat {heat} out of range");
                if p.life_of(i) > 0.0 {
                    assert!(
                        obs.get((x + 0.5) as usize, (y + 0.5) as usize) < 0.5,
                        "live particle inside the obstacle at ({x}, {y})"
                    );
                }
            }
            p.build_render_buffer(w, h);
            for q in p.render_buffer().chunks(PARTICLE_STRIDE) {
                assert!(
                    q.iter().all(|v| (0.0..=1.0).contains(v)),
                    "render buffer out of the shader's range: {q:?}"
                );
            }
        }
    }
}

#[test]
fn obstacle_test_matches_the_solver_threshold() {
    // The fluid treats `>= 0.5` as solid; the particle cull must agree, or
    // particles pile up in the half-cell the two disagree about.
    let vel = const_field(8, 8, 0.0, 0.0);
    let mut obs = Grid::new(8, 8);
    obs.set(4, 4, 0.49);
    assert!(!obstacle_frame(&vel, &obs).is_solid(4.0, 4.0));
    obs.set(4, 4, 0.5);
    let frame = obstacle_frame(&vel, &obs);
    assert!(frame.is_solid(4.0, 4.0));
    assert!(frame.is_solid(4.49, 4.49), "nearest cell is still (4, 4)");
    assert!(!frame.is_solid(3.49, 4.0));
    // Out of range and non-finite coordinates must clamp, not index wild.
    assert!(!frame.is_solid(f32::NAN, f32::NAN));
    assert!(!frame.is_solid(1e9, 1e9));
    assert!(!frame.is_solid(-1e9, -1e9));
}

#[test]
fn a_lower_resolution_obstacle_still_culls_correctly() {
    // The engine hands `step` an obstacle at the fluid's resolution, so the
    // `osx`/`osy` rescale is dead code there — and dead code is where a
    // transposed or off-by-one factor hides until someone changes a constant.
    let (w, h) = (64, 36);
    let vel = const_field(w, h, 15.0, 0.0);
    let (ow, oh) = (32, 18);
    let mut obs = Grid::new(ow, oh);
    for y in 0..oh {
        for x in ow / 2..ow {
            obs.set(x, y, 1.0);
        }
    }
    let p_params = params(1.0, 5.0, 60_000.0);
    let mut p = Particles::new(400, 5);
    p.set_active(400);
    p.seed_uniform(w, h, 5.0);

    let mut live_seen = 0;
    let mut deepest = 0.0f32;
    for _ in 0..200 {
        p.step(&vel, &obs, DT, &p_params, &ParticleConfig::default());
        for i in 0..p.active() {
            if p.life_of(i) <= 0.0 {
                continue;
            }
            live_seen += 1;
            let (x, y) = p.position(i);
            let ox = x * (ow - 1) as f32 / (w - 1) as f32;
            let oy = y * (oh - 1) as f32 / (h - 1) as f32;
            assert!(
                obs.get((ox + 0.5) as usize, (oy + 0.5) as usize) < 0.5,
                "live particle at ({x}, {y}) is inside the half-res obstacle"
            );
            deepest = deepest.max(x);
        }
    }
    assert!(live_seen > 10_000, "nothing survived to test: {live_seen}");
    // The wall in grid space sits where `round(x * (ow - 1) / (w - 1))`
    // first reaches `ow / 2`, i.e. x = 31.5. Asserting only "no live
    // particle is inside" would pass just as happily with the scale off by
    // a cell in the *conservative* direction, so pin both sides: the flow
    // pushes particles at the wall every frame, so the deepest live one has
    // to end up in the last free half-cell.
    let wall = 15.5 * (w - 1) as f32 / (ow - 1) as f32;
    assert!(
        deepest > wall - 0.5 && deepest < wall,
        "the obstacle rescale is off: deepest live x {deepest}, wall at {wall}"
    );
}
