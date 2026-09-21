//! One hand, one spell: that each cast moves the field the way its name claims, and that the two one-shot casts (shatter, the release ring) fire exactly once per hold.

use super::support::*;

#[test]
fn an_idle_present_hand_injects_energy_and_an_absent_one_does_not() {
    let tracker = sweeping(&IDLE_HAND);
    assert_eq!(tracker.hands()[0].spell, Spell::Idle, "expected no spell");
    assert!(
        tracker.hands()[0].velocity[0] > 0.2,
        "hand should be moving"
    );

    let mut rig = Rig::new();
    let report = rig.run(&tracker, DT);
    assert_eq!(report.spells[0], Spell::Idle);
    assert!(
        report.injected > 0.0,
        "an idle hand must still drive the fluid"
    );
    assert!(rig.fluid.energy() > 0.0);
    // It must push the way the hand is going.
    let palm = palm_of(&tracker);
    assert!(rig.velocity_at(palm[0], palm[1]).0 > 0.0);

    let mut rig = Rig::new();
    let report = rig.run(&empty_tracker(), DT);
    assert_eq!(report.injected, 0.0, "no hand, no energy");
    assert_eq!(rig.fluid.energy(), 0.0);
    assert_eq!(report.spells, [Spell::Idle, Spell::Idle]);
}

#[test]
fn shatter_fires_once_per_hold_and_rearms_on_release() {
    let hand = Hand::at(0.5, 0.5).gesture(synth::THUMB_UP);
    let mut tracker = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, &hand);
    let mut rig = Rig::new();

    let mut bursts = 0;
    for _ in 0..30 {
        tracker.update_hands(&buf, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(tracker.hands()[0].spell, Spell::Shatter);
    assert_eq!(bursts, 1, "holding the gesture re-fired the burst");

    // Drop the hand for long enough to be released, then show it again.
    let empty = synth::buffer();
    for _ in 0..12 {
        tracker.update_hands(&empty, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(bursts, 1, "a burst fired while no hand was present");
    for _ in 0..30 {
        tracker.update_hands(&buf, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(bursts, 2, "the burst did not re-arm after release");

    // And the burst actually did something to the pool.
    assert!(
        (0..rig.particles.active()).any(|i| rig.particles.heat_of(i) > 0.9),
        "no particle came out of the burst hot"
    );
}

#[test]
fn ignite_paints_the_whole_stroke_not_just_the_endpoints() {
    let mut rig = Rig::new();
    let left = holding(&Hand::at(0.15, 0.5).gesture(synth::POINTING_UP));
    assert_eq!(left.hands()[0].spell, Spell::Ignite);
    rig.run(&left, DT);
    let a = to_grid(
        left.hands()[0].index_tip,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );

    // The finger is now far away; one frame of dye must cover the gap.
    let right = holding(&Hand::at(0.85, 0.5).gesture(synth::POINTING_UP));
    rig.run(&right, DT);
    let b = to_grid(
        right.hands()[0].index_tip,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    let span = length(b[0] - a[0], b[1] - a[1]);
    assert!(span > 100.0, "the test needs a long stroke, got {span}");

    let red = &rig.fluid.dye()[0];
    for t in [0.25, 0.5, 0.75] {
        let x = lerp(a[0], b[0], t);
        let y = lerp(a[1], b[1], t);
        assert!(
            red.sample(x, y) > 1e-4,
            "no dye at {t} along the stroke ({x}, {y})"
        );
    }
    // Mid-stroke brightness must be the same order as the endpoints, not a
    // faint smear from the splat tails.
    let mid = red.sample(lerp(a[0], b[0], 0.5), lerp(a[1], b[1], 0.5));
    let end = red.sample(b[0], b[1]);
    assert!(mid * 4.0 > end, "stroke is dotted: mid {mid} vs end {end}");
}

#[test]
fn vortex_winds_up_and_the_flow_is_tangential() {
    let hand = Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched();

    // Freshly latched versus held past the windup.
    let mut young = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, &hand);
    for _ in 0..GestureConfig::default().commit_frames {
        young.update_hands(&buf, HAND_DT);
        young.update_two_hand(HAND_DT);
    }
    assert_eq!(young.hands()[0].spell, Spell::Vortex);
    let mut fresh_rig = Rig::new();
    fresh_rig.run(&young, DT);
    let fresh = fresh_rig.fluid.energy();

    let old = holding(&hand);
    assert!(old.hands()[0].spell_age > VORTEX_WINDUP);
    let mut wound_rig = Rig::new();
    wound_rig.run(&old, DT);
    let wound = wound_rig.fluid.energy();
    assert!(
        wound > fresh * 4.0,
        "vortex did not wind up: {fresh} -> {wound}"
    );

    // Sample to the right of the pinch point: the flow there must be
    // perpendicular to the radius.
    let tip = old.hands()[0].index_tip;
    let thumb = old.hands()[0].thumb_tip;
    let center = to_grid(
        [(tip[0] + thumb[0]) * 0.5, (tip[1] + thumb[1]) * 0.5],
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    let radius = old.hands()[0].scale * (FLUID_W - 1) as f32 * PALM_RADIUS_SCALE;
    let probe = radius * VORTEX_RADIUS_SCALE * 0.4;
    let (u, v) = wound_rig.velocity_at(center[0] + probe, center[1]);
    assert!(v.abs() > 1.0, "no swirl at the probe: {v}");
    assert!(
        u.abs() < 0.35 * v.abs(),
        "flow is radial, not tangential: u {u} v {v}"
    );
    // Opposite sides of the core circulate opposite ways.
    let (_, v_left) = wound_rig.velocity_at(center[0] - probe, center[1]);
    assert!(v * v_left < 0.0, "not a circulation: {v} vs {v_left}");
}

#[test]
fn repel_pushes_out_and_attract_pulls_in_more_gently() {
    let mut out = Rig::new();
    let repel = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let palm = palm_of(&repel);
    // A particle just to the right of the palm, to check the particle path.
    out.particles
        .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
    out.run(&repel, DT);
    let (ru, _) = out.velocity_at(palm[0] + 10.0, palm[1]);
    assert!(ru > 0.0, "repel pushed inward: {ru}");
    assert!(
        out.particles.velocity(0).0 > 0.0,
        "repel did not push particles out"
    );

    let mut inward = Rig::new();
    let attract = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    inward
        .particles
        .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
    inward.run(&attract, DT);
    let (au, _) = inward.velocity_at(palm[0] + 10.0, palm[1]);
    assert!(au < 0.0, "attract pushed outward: {au}");
    assert!(
        inward.particles.velocity(0).0 < 0.0,
        "attract did not pull particles in"
    );
    assert!(
        au.abs() < ru.abs() * 0.8,
        "attract must be the weaker well: {au} vs {ru}"
    );

    // Both should be symmetric about the palm.
    let (lu, _) = out.velocity_at(palm[0] - 10.0, palm[1]);
    assert!((lu + ru).abs() < 0.2 * ru.abs(), "asymmetric push");
}

#[test]
fn freeze_damps_the_fluid_locally_and_leaves_the_rest_alone() {
    let mut rig = Rig::new();
    rig.fluid.velocity_mut().u.data.fill(50.0);
    let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::VICTORY));
    assert_eq!(tracker.hands()[0].spell, Spell::Freeze);
    let palm = palm_of(&tracker);
    for _ in 0..10 {
        rig.run(&tracker, DT);
    }
    let (near, _) = rig.velocity_at(palm[0], palm[1]);
    // Not a "did it go down" test: the always-on palm drag relaxes toward a
    // still hand's zero velocity all by itself, so a loose threshold here
    // passes for `Idle` too and proves nothing about the spell. Measure the
    // spell alone with the drag switched off, against its own rate.
    assert!(near < 40.0, "freeze did not damp the local flow: {near}");
    let (far, _) = rig.velocity_at(5.0, 5.0);
    assert!((far - 50.0).abs() < 1e-3, "freeze damped the whole grid");
    // And it lays down cold dye: hue 0.55 has no red in it.
    let dye = rig.fluid.dye();
    assert!(dye[2].sample(palm[0], palm[1]) > dye[0].sample(palm[0], palm[1]));

    // The falloff weight is ~1 over the middle of the disc, so the centre
    // must decay by exactly `exp(-FREEZE_RATE * t)` — and must do so at any
    // frame rate, since the weight sits inside the decay rate.
    for (hz, frames) in [(60.0f32, 30u32), (240.0, 120)] {
        let mut rig = Rig::new();
        rig.params.hand_force = 0.0;
        rig.fluid.velocity_mut().u.data.fill(50.0);
        for _ in 0..frames {
            rig.run(&tracker, 1.0 / hz);
        }
        let elapsed = frames as f32 / hz;
        let want = 50.0 * decay(FREEZE_RATE, elapsed);
        let got = rig.velocity_at(palm[0], palm[1]).0;
        assert!(
            (got - want).abs() < 0.02 * want,
            "{hz} Hz: freeze centre {got}, analytic {want}"
        );
    }

    // Nothing but the drag: an idle hand must leave the field alone once
    // the drag is off, which is what makes the comparison above meaningful.
    let mut idle = Rig::new();
    idle.params.hand_force = 0.0;
    idle.fluid.velocity_mut().u.data.fill(50.0);
    let idle_tracker = holding(&IDLE_HAND);
    assert_eq!(idle_tracker.hands()[0].spell, Spell::Idle);
    for _ in 0..30 {
        idle.run(&idle_tracker, 1.0 / 60.0);
    }
    let palm = palm_of(&idle_tracker);
    assert!(
        (idle.velocity_at(palm[0], palm[1]).0 - 50.0).abs() < 1e-3,
        "an idle hand damped the field"
    );
}

#[test]
fn a_sweeping_open_palm_still_repels() {
    let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(synth::OPEN_PALM));
    assert_eq!(tracker.hands()[0].spell, Spell::Repel);
}

#[test]
fn a_closed_fist_gathers_particles_instead_of_slinging_them_past() {
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    let hand = &tracker.hands()[0];
    let (sx, sy) = ((FLUID_W - 1) as f32, (FLUID_H - 1) as f32);
    let palm = to_grid(hand.palm, sx, sy);

    let mut rig = Rig::new();
    rig.particles
        .place(0, palm[0] + 10.0, palm[1], 0.0, 0.0, 30.0);
    let obstacle = Grid::new(FLUID_W, FLUID_H);
    let cfg = ParticleConfig::default();
    for _ in 0..60 {
        rig.run(&tracker, DT);
        rig.particles
            .step(rig.fluid.velocity(), &obstacle, DT, &rig.params, &cfg);
    }
    let (px, py) = rig.particles.position(0);
    let dist = length(px - palm[0], py - palm[1]);
    let (vx, vy) = rig.particles.velocity(0);
    assert!(dist < 10.0, "particle was not gathered: {dist} cells out");
    // Held, not orbiting: the grip has taken the speed off it.
    assert!(length(vx, vy) < 60.0, "particle still slinging: {vx} {vy}");
}

#[test]
fn one_pointing_hand_is_still_an_ignite() {
    let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::POINTING_UP));
    let mut rig = Rig::new();
    let report = rig.run(&tracker, DT);
    assert_eq!(report.spells[0], Spell::Ignite);
}

#[test]
fn opening_a_held_fist_fires_a_ring_once() {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    let mut rig = Rig::new();
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    for _ in 0..45 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        rig.run(&t, HAND_DT);
    }
    assert_eq!(t.hands()[0].spell, Spell::Attract);
    let palm = palm_of(&t);

    // Still the pool first so the only fast particles afterwards are the
    // ring's own, not two seconds of attract pull.
    rig.particles
        .seed_uniform(FLUID_W, FLUID_H, rig.params.particle_life);
    // Open the hand: the tracker takes `commit_frames` to relatch.
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut bursts = 0;
    let mut saw_release = false;
    for _ in 0..12 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        let report = rig.run(&t, HAND_DT);
        bursts += report.bursts;
        saw_release |= report.spells[0] == Spell::Release;
    }
    assert_eq!(bursts, 1, "the ring must fire exactly once per open");
    assert!(saw_release, "the release should be reported to the HUD");
    // Particles were re-seeded on a ring and fly outward from the palm.
    let mut outward = 0;
    let mut total = 0;
    for i in 0..rig.particles.active() {
        let (px, py) = rig.particles.position(i);
        let (vx, vy) = rig.particles.velocity(i);
        let speed = length(vx, vy);
        if speed < 50.0 {
            continue;
        }
        total += 1;
        if (px - palm[0]) * vx + (py - palm[1]) * vy > 0.0 {
            outward += 1;
        }
    }
    assert!(
        total > 100,
        "expected a burst of fast particles, got {total}"
    );
    assert!(
        outward * 10 > total * 9,
        "burst is not outward: {outward}/{total}"
    );
}

#[test]
fn a_fist_barely_closed_does_not_fire_a_ring() {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    let mut rig = Rig::new();
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    for _ in 0..4 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        rig.run(&t, HAND_DT);
    }
    assert_eq!(t.hands()[0].spell, Spell::Attract);
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut bursts = 0;
    for _ in 0..12 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        bursts += rig.run(&t, HAND_DT).bursts;
    }
    assert_eq!(bursts, 0, "an uncharged fist must not fire");
}
