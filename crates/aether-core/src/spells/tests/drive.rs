//! The always-on drive and the framerate independence of every force.

use super::support::*;

#[test]
fn the_whole_field_drives_push_the_fluid() {
    // Optical flow: model-free, so it must work with no tracker at all.
    let mut rig = Rig::new();
    rig.params.flow_force = 0.45;
    rig.flow.u.data.fill(4.0);
    let report = rig.run(&empty_tracker(), DT);
    assert!(report.injected > 0.0);
    let (u, _) = rig.velocity_at(100.0, 70.0);
    assert!(u > 0.0, "flow drive pushed nothing: {u}");

    // Body edge velocity, injected only where the silhouette moves.
    let mut rig = Rig::new();
    rig.params.body_push = 1.0;
    for y in 60..80 {
        for x in 100..120 {
            rig.body.u.set(x, y, -6.0);
        }
    }
    rig.run(&empty_tracker(), DT);
    let (inside, _) = rig.velocity_at(110.0, 70.0);
    let (outside, _) = rig.velocity_at(10.0, 10.0);
    assert!(inside < 0.0, "body edge did not push: {inside}");
    assert_eq!(outside, 0.0, "body edge leaked across the grid");

    // A zero body field must cost nothing and change nothing.
    let mut rig = Rig::new();
    rig.params.body_push = 1.0;
    assert_eq!(rig.run(&empty_tracker(), DT).injected, 0.0);
}

#[test]
fn forces_are_framerate_independent() {
    // The same simulated interval, split six ways. Both paths must deposit
    // the same energy; a `1 - rate * dt` decay or an unscaled impulse fails
    // this by a factor of six.
    let tracker = sweeping(&IDLE_HAND);
    let mut coarse = Rig::new();
    coarse.run(&tracker, DT);
    let mut fine = Rig::new();
    for _ in 0..6 {
        fine.run(&tracker, DT / 6.0);
    }
    let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
    assert!(a > 0.0);
    assert!(
        (a - b).abs() < 0.005 * a,
        "frame rate changed the outcome: {a} vs {b}"
    );

    // Repel is a pure dt-scaled impulse. Probed outside the palm drag's
    // radius but inside the push's, where nothing else has touched the
    // field, the two paths must agree to floating-point noise.
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let palm = palm_of(&tracker);
    let mut coarse = Rig::new();
    coarse.run(&tracker, DT);
    let mut fine = Rig::new();
    for _ in 0..6 {
        fine.run(&tracker, DT / 6.0);
    }
    let probe = palm[0] + 50.0;
    let (a, b) = (
        coarse.velocity_at(probe, palm[1]).0,
        fine.velocity_at(probe, palm[1]).0,
    );
    assert!(a > 1.0, "the probe should see the push: {a}");
    assert!(
        (a - b).abs() < 1e-3 * a,
        "impulse not dt-scaled: {a} vs {b}"
    );

    // Over the whole field the two differ by a few percent, and that is
    // operator splitting rather than a dt bug: the fine path interleaves
    // six drag passes with six impulses, so the drag sees — and damps — a
    // field the coarse path only produces after its single drag pass.
    let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
    assert!(
        (a - b).abs() < 0.04 * a,
        "splitting error too large: {a} vs {b}"
    );
}

#[test]
fn a_held_spell_survives_a_long_stepped_run() {
    // The real failure mode this guards: a force that is individually
    // clamped but accumulates every frame until the solver diverges.
    for gesture in [
        synth::OPEN_PALM,
        synth::CLOSED_FIST,
        synth::POINTING_UP,
        synth::VICTORY,
        synth::THUMB_UP,
    ] {
        let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(gesture));
        // A sixteenth of the real grid: this test is about what accumulates
        // over 240 steps, which is resolution independent, and at full size
        // five of these runs take a minute and a half in a debug build.
        let mut rig = Rig::sized(64, 36);
        for _ in 0..240 {
            rig.run(&tracker, DT);
            rig.fluid.step(DT, &rig.params);
        }
        assert_eq!(
            rig.fluid.sanitize(),
            0,
            "gesture {gesture} produced non-finite cells"
        );
        let speed = rig.fluid.max_speed();
        assert!(
            speed.is_finite() && speed < MAX_IMPULSE * 4.0,
            "gesture {gesture} diverged: {speed} cells/s"
        );
    }
}
