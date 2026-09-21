//! Latching: a spell needs commit frames, survives a one-frame flicker, and a brief dropout keeps the hand present.

use super::support::*;

#[test]
fn a_spell_needs_commit_frames_to_latch() {
    let cfg = GestureConfig::default();
    let mut t = GestureTracker::new(cfg);
    let hand = Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM);
    for frame in 1..cfg.commit_frames {
        feed(&mut t, &hand, 1, DT);
        assert_eq!(
            t.hands()[0].spell,
            Spell::Idle,
            "latched after only {frame} frames"
        );
    }
    feed(&mut t, &hand, 1, DT);
    assert_eq!(t.hands()[0].spell, Spell::Repel);
    assert!(t.hands()[0].spell_age < 1e-6, "age should restart on latch");
    feed(&mut t, &hand, 3, DT);
    assert!(t.hands()[0].spell_age > 2.0 * DT, "age should accumulate");
}

#[test]
fn a_one_frame_flicker_does_not_dislodge_the_latched_spell() {
    let mut t = tracker();
    let fist = Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST);
    feed(&mut t, &fist, 10, DT);
    assert_eq!(t.hands()[0].spell, Spell::Attract);

    // MediaPipe really does emit one frame of the wrong label mid-hold.
    feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 1, DT);
    assert_eq!(
        t.hands()[0].spell,
        Spell::Attract,
        "one frame of Open_Palm dislodged the latch"
    );
    feed(&mut t, &fist, 1, DT);
    assert_eq!(t.hands()[0].spell, Spell::Attract);

    // Held, though, it must win.
    feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 4, DT);
    assert_eq!(t.hands()[0].spell, Spell::Repel);
}

#[test]
fn a_brief_dropout_keeps_the_hand_present() {
    let cfg = GestureConfig::default();
    let mut t = GestureTracker::new(cfg);
    feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 8, DT);
    assert!(t.hands()[0].present);

    let empty = synth::buffer();
    for frame in 1..cfg.release_frames {
        t.update_hands(&empty, DT);
        assert!(
            t.hands()[0].present,
            "released after only {frame} missing frames"
        );
        assert_eq!(t.hands()[0].spell, Spell::Repel, "spell dropped too early");
    }
    t.update_hands(&empty, DT);
    assert!(!t.hands()[0].present, "hand should be gone by now");
    assert_eq!(t.hands()[0].spell, Spell::Idle);
    assert_eq!(t.hands()[0].velocity, [0.0, 0.0]);
}
