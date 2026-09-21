//! Garbage in, no panic and no poisoned state; and that reset really forgets.

use super::support::*;

#[test]
fn garbage_input_never_panics_or_poisons_the_state() {
    let mut t = tracker();
    // Establish a good hand first, so there is state to poison.
    feed(&mut t, &Hand::at(0.4, 0.6).gesture(synth::OPEN_PALM), 8, DT);
    let good_palm = t.hands()[0].palm;

    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, &Hand::at(0.4, 0.6));
    for v in buf.iter_mut().skip(4).take(10) {
        *v = f32::NAN;
    }
    t.update_hands(&buf, DT);
    assert_eq!(
        t.hands()[0].palm,
        good_palm,
        "NaN landmarks entered the filter"
    );

    // Present, but every landmark collapsed to the origin.
    let mut zeroed = synth::buffer();
    zeroed[0] = 1.0;
    zeroed[3] = 0.99;
    t.update_hands(&zeroed, DT);
    assert!(t.hands()[0].palm.iter().all(|v| v.is_finite()));

    // Short buffers, absurd timesteps, infinities.
    t.update_hands(&[], f32::NAN);
    t.update_hands(&[1.0, 0.0], 1e9);
    t.update_hands(&buf, -1.0);
    t.update_two_hand(f32::NAN);
    t.update_pose(&[], 0.0);
    t.update_pose(&[1.0], DT);
    let mut pose = pose_buffer(&[(PL_LEFT_SHOULDER, f32::NAN, f32::INFINITY, 0.9)]);
    pose[1 + PL_RIGHT_HIP * 4 + 3] = f32::NAN;
    t.update_pose(&pose, 1e-9);
    t.update_two_hand(1e9);

    for hand in t.hands() {
        assert!(hand.palm.iter().all(|v| v.is_finite()));
        assert!(hand.velocity.iter().all(|v| v.is_finite()));
        assert!(hand.pinch.is_finite() && hand.openness.is_finite());
        assert!(hand.scale.is_finite() && hand.scale > 0.0);
        assert!(hand.spell_age.is_finite());
    }
    assert!(t.pose().center.iter().all(|v| v.is_finite()));
    assert!(t.pose().span.is_finite());
    assert!(t.two_hand().angular_velocity.is_finite());
    assert!(t.two_hand().distance.is_finite());
}

#[test]
fn reset_forgets_everything() {
    let mut t = tracker();
    feed(
        &mut t,
        &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM),
        10,
        DT,
    );
    assert_eq!(t.hands()[0].spell, Spell::Repel);
    t.reset();
    assert!(!t.hands()[0].present);
    assert_eq!(t.hands()[0].spell, Spell::Idle);
    assert!(!t.pose().present);
    assert!(!t.two_hand().both_present);
}
