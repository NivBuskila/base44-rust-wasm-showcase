//! The combo payloads and the two-hand time warp.

use super::support::*;

#[test]
fn every_combo_payload_moves_the_field_and_stays_finite() {
    for effect in [
        ComboEffect::Nova,
        ComboEffect::Tempest,
        ComboEffect::Supernova,
    ] {
        let mut rig = Rig::new();
        let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        let hit = ComboHit {
            combo: 0,
            effect,
            slot: 0,
        };
        let report = rig.run_with(&tracker, Some(hit), DT);
        assert_eq!(report.bursts, 1, "{effect:?} did not report its burst");
        assert!(
            report.injected > 0.0 && report.injected.is_finite(),
            "{effect:?} injected {}",
            report.injected
        );
        // A payload is an event; it must not leave the solver unstable.
        for _ in 0..30 {
            rig.run(&tracker, DT);
        }
        assert_eq!(rig.fluid.sanitize(), 0, "{effect:?} destabilised the fluid");
    }
}

#[test]
fn time_scale_follows_the_two_hand_rotation_and_stays_clamped() {
    let mut rig = Rig::new();
    let forward = rotating(2.5, 90);
    assert!(forward.two_hand().both_present);
    let mut ts = 1.0;
    for _ in 0..60 {
        ts = rig.run(&forward, DT).time_scale;
    }
    assert!(ts > 1.1, "rotation did not warp time: {ts}");
    assert!(ts <= MAX_TIME_SCALE);

    let backward = rotating(-2.5, 90);
    for _ in 0..60 {
        ts = rig.run(&backward, DT).time_scale;
    }
    assert!(ts < 0.9, "reverse rotation should slow time: {ts}");
    assert!(ts >= MIN_TIME_SCALE);

    // An absurd spin must saturate, not run away.
    let wild = rotating(60.0, 120);
    for _ in 0..240 {
        ts = rig.run(&wild, DT).time_scale;
    }
    assert!(
        (MIN_TIME_SCALE..=MAX_TIME_SCALE).contains(&ts),
        "time scale escaped its bounds: {ts}"
    );

    // And it eases back to normal once the hands are gone.
    let none = empty_tracker();
    for _ in 0..120 {
        ts = rig.run(&none, DT).time_scale;
    }
    assert!((ts - 1.0).abs() < 0.05, "time scale did not recover: {ts}");
}
