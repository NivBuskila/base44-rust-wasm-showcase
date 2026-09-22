//! Reading one hand's shape: pinch and openness, the canned labels, and which of the two wins when they disagree.

use super::support::*;

#[test]
fn pinch_is_scale_invariant() {
    // The same pose near the camera and far from it. If pinch used raw
    // normalised distance this would differ by the scale ratio, and the
    // gesture would only work with the hand right up against the lens.
    let mut near = tracker();
    let mut far = tracker();
    feed(
        &mut near,
        &Hand::at(0.5, 0.5).pinched().scaled(0.16),
        20,
        DT,
    );
    feed(
        &mut far,
        &Hand::at(0.5, 0.5).pinched().scaled(0.045),
        20,
        DT,
    );
    let (a, b) = (near.hands()[0].pinch, far.hands()[0].pinch);
    assert!(
        (a - b).abs() < 0.02,
        "pinch not scale invariant: {a} vs {b}"
    );
    assert!(a > 0.9, "a touching pinch should read ~1, got {a}");

    // Openness too, and with the same hand at both distances.
    let mut near = tracker();
    let mut far = tracker();
    feed(
        &mut near,
        &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM).scaled(0.16),
        20,
        DT,
    );
    feed(
        &mut far,
        &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM).scaled(0.045),
        20,
        DT,
    );
    let (a, b) = (near.hands()[0].openness, far.hands()[0].openness);
    assert!((a - b).abs() < 0.02, "openness not scale invariant");
}

#[test]
fn pinch_and_openness_separate_the_poses() {
    let mut t = tracker();
    feed(
        &mut t,
        &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM),
        20,
        DT,
    );
    assert!(t.hands()[0].pinch < 0.05, "open hand read as pinched");
    assert!(t.hands()[0].openness > 0.95, "open hand read as closed");

    let mut t = tracker();
    feed(
        &mut t,
        &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST),
        20,
        DT,
    );
    assert!(t.hands()[0].openness < 0.05, "fist read as open");

    let mut t = tracker();
    feed(&mut t, &Hand::at(0.5, 0.5).pinched(), 20, DT);
    assert!(t.hands()[0].pinch > 0.9, "touching fingers read as open");
}

#[test]
fn canned_labels_map_to_their_spells() {
    for (gesture, want) in [
        (synth::OPEN_PALM, Spell::Repel),
        (synth::CLOSED_FIST, Spell::Attract),
        (synth::POINTING_UP, Spell::Ignite),
        (synth::VICTORY, Spell::Freeze),
        (synth::THUMB_UP, Spell::Shatter),
    ] {
        let mut t = tracker();
        feed(&mut t, &Hand::at(0.5, 0.5).gesture(gesture), 8, DT);
        assert_eq!(t.hands()[0].spell, want, "gesture {gesture} mismapped");
    }
}

#[test]
fn a_pinch_beats_a_weak_label_but_not_a_confident_one() {
    let mut t = tracker();
    feed(
        &mut t,
        &Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched(),
        8,
        DT,
    );
    assert_eq!(t.hands()[0].spell, Spell::Vortex);

    // A fist has the thumb and index tips close together too, so a pure
    // distance test would read it as a pinch; a confident label must win.
    let mut t = tracker();
    feed(
        &mut t,
        &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(0.95),
        8,
        DT,
    );
    assert_eq!(t.hands()[0].spell, Spell::Attract);
}

#[test]
fn an_unlabelled_fist_is_a_gravity_well_not_a_vortex() {
    // A fist's thumb and index tips end up ~0.16 hand scales apart, i.e. a
    // perfect pinch by thumb-to-index distance alone. Whenever the canned
    // label is unconfident — which is exactly what the geometric fallback
    // exists for, and what happens for a few frames every time a hand
    // closes — a distance-only test spins a vortex out of a closed fist.
    for score in [0.49, 0.0] {
        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(score),
            10,
            DT,
        );
        let hand = &t.hands()[0];
        assert!(hand.pinch > 0.9, "the fist geometry should look pinched");
        assert_eq!(
            hand.spell,
            Spell::Attract,
            "an unlabelled fist (score {score}) should be a gravity well"
        );
    }

    // The pinch path must still win when the fingers are actually held out
    // in front of the knuckles, label or no label.
    let mut t = tracker();
    feed(&mut t, &Hand::at(0.5, 0.5).pinched().score(0.0), 10, DT);
    assert_eq!(t.hands()[0].spell, Spell::Vortex);

    // The two poses are separated by where the pinch sits, not how tight
    // it is: both read pinch ~1, and only the reach differs.
    let mut fist = tracker();
    feed(
        &mut fist,
        &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(0.0),
        10,
        DT,
    );
    let a = pinch_reach(&fist.hands()[0]);
    let b = pinch_reach(&t.hands()[0]);
    assert!(a < PINCH_REACH, "fist reach {a} should be short");
    assert!(b > PINCH_REACH, "pinch reach {b} should be long");
}

#[test]
fn a_closing_hand_does_not_flash_a_vortex_on_the_way_to_a_fist() {
    // MediaPipe's score collapses while the fingers are mid-curl. The
    // commit window is only 3 frames, so a spurious candidate that holds
    // for a third of a second really does latch.
    let mut t = tracker();
    let mut seen = Vec::new();
    for frame in 0..20 {
        let (g, score) = if frame < 5 {
            (synth::OPEN_PALM, 0.9)
        } else if frame < 14 {
            (synth::CLOSED_FIST, 0.2)
        } else {
            (synth::CLOSED_FIST, 0.9)
        };
        feed(&mut t, &Hand::at(0.5, 0.5).gesture(g).score(score), 1, DT);
        seen.push(t.hands()[0].spell);
    }
    assert!(
        !seen.contains(&Spell::Vortex),
        "a vortex flashed while the hand was closing: {seen:?}"
    );
    assert_eq!(seen[19], Spell::Attract);
}
