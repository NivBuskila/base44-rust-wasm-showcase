//! The ceilings: no cast, however absurd its parameters, may leave the fluid non-finite or above the impulse ceiling, and garbage input may not panic.

use super::support::*;

#[test]
fn every_spell_keeps_the_fluid_finite_and_under_the_ceiling() {
    let cases = [
        (Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), Spell::Repel),
        (
            Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST),
            Spell::Attract,
        ),
        (
            Hand::at(0.5, 0.5).gesture(synth::POINTING_UP),
            Spell::Ignite,
        ),
        (Hand::at(0.5, 0.5).gesture(synth::VICTORY), Spell::Freeze),
        (Hand::at(0.5, 0.5).gesture(synth::THUMB_UP), Spell::Shatter),
        (
            Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched(),
            Spell::Vortex,
        ),
        (IDLE_HAND, Spell::Idle),
    ];
    for (hand, want) in cases {
        let tracker = sweeping(&hand);
        assert_eq!(tracker.hands()[0].spell, want, "wrong spell latched");
        let mut rig = Rig::new();
        for _ in 0..4 {
            let report = rig.run(&tracker, DT);
            assert!(report.injected.is_finite(), "{want:?} injected NaN");
            assert!(report.time_scale.is_finite());
        }
        let vel = rig.fluid.velocity();
        assert!(
            vel.u.max_abs() <= MAX_IMPULSE && vel.v.max_abs() <= MAX_IMPULSE,
            "{want:?} exceeded MAX_IMPULSE: {} / {}",
            vel.u.max_abs(),
            vel.v.max_abs()
        );
        assert_eq!(rig.fluid.sanitize(), 0, "{want:?} wrote a non-finite cell");
        for channel in rig.fluid.dye() {
            assert!(
                channel.data.iter().all(|v| v.is_finite()),
                "{want:?} wrote non-finite dye"
            );
        }
    }
}

#[test]
fn absurd_parameters_are_clamped_to_the_impulse_ceiling() {
    // hand_force saturates at 8 and dt at 0.25, which together ask for a
    // 3000 cells/s push — the clamp is the only thing between that and a
    // velocity field the projection cannot resolve.
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut rig = Rig::new();
    rig.params.hand_force = 8.0;
    rig.run(&tracker, 1e6);
    let vel = rig.fluid.velocity();
    assert!(
        (vel.u.max_abs() - MAX_IMPULSE).abs() < 1e-3,
        "clamp did not bite: {}",
        vel.u.max_abs()
    );
    assert!(vel.v.max_abs() <= MAX_IMPULSE);
    assert_eq!(rig.fluid.sanitize(), 0);
}

#[test]
fn an_ignite_stroke_respects_the_impulse_ceiling() {
    // The trail is a line of deliberately overlapping splats: they are
    // spaced `IGNITE_STEP` apart and each is `IGNITE_RADIUS` wide, so
    // clamping them one at a time is not enough — the sum is what lands in
    // the velocity field. Worst case is the module's own `MAX_DT` and the
    // largest `hand_force` `Params::set` allows.
    let mut worst = 0.0f32;
    for cells in [0.0f32, 1.2, 2.2, 3.4, 4.8, 9.0, 40.0, 200.0] {
        let mut rig = Rig::new();
        rig.params.hand_force = 8.0;
        let a = holding(&Hand::at(0.3, 0.5).gesture(synth::POINTING_UP));
        rig.run(&a, MAX_DT);
        let b =
            holding(&Hand::at(0.3 + cells / (FLUID_W - 1) as f32, 0.5).gesture(synth::POINTING_UP));
        rig.run(&b, MAX_DT);
        let vel = rig.fluid.velocity();
        worst = worst.max(vel.u.max_abs()).max(vel.v.max_abs());
    }
    assert!(
        worst <= MAX_IMPULSE,
        "ignite drove the field to {worst} cells/s, past the {MAX_IMPULSE} ceiling"
    );
}

#[test]
fn garbage_input_never_panics() {
    let tracker = sweeping(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));

    // Absurd timesteps.
    let mut rig = Rig::new();
    for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e9] {
        let report = rig.run(&tracker, dt);
        assert!(report.time_scale.is_finite());
        assert!(report.injected.is_finite());
    }
    assert_eq!(rig.fluid.sanitize(), 0);

    // A non-finite flow field must not reach the velocity grid.
    let mut rig = Rig::new();
    rig.params.flow_force = 1.0;
    rig.params.body_push = 1.0;
    rig.flow.u.data.fill(f32::NAN);
    rig.body.v.data.fill(f32::INFINITY);
    rig.run(&tracker, DT);
    assert_eq!(rig.fluid.sanitize(), 0, "NaN flow leaked into the fluid");

    // A landmark-free "present" hand, and hands off the edge of the frame.
    let mut rig = Rig::new();
    for hand in [
        Hand::at(-4.0, 9.0).gesture(synth::OPEN_PALM),
        Hand::at(0.5, 0.5).gesture(synth::POINTING_UP).scaled(0.9),
        Hand::at(0.5, 0.5).gesture(synth::THUMB_UP).scaled(0.006),
    ] {
        let tracker = holding(&hand);
        rig.run(&tracker, DT);
    }
    assert_eq!(rig.fluid.sanitize(), 0);

    // The smallest grid a Fluid can exist on, and an empty particle pool.
    let mut tiny = Fluid::new(2, 2);
    let mut none = Particles::new(0, 1);
    let mut state = SpellState::new();
    let report = apply(
        &tracker,
        &VecField::new(2, 2),
        &VecField::new(2, 2),
        &mut tiny,
        &mut none,
        &mut state,
        &Params::default(),
        [None; 2],
        [0.0; 2],
        DuetFrame::default(),
        DT,
    );
    assert!(report.injected.is_finite());
    assert_eq!(tiny.sanitize(), 0);
}
