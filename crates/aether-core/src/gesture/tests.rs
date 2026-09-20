use super::synth::{self, Hand};
use super::*;
use crate::config::POSE_STRIDE;

const DT: f32 = 1.0 / 30.0;

fn tracker() -> GestureTracker {
    GestureTracker::new(GestureConfig::default())
}

/// Feeds one hand for `frames` frames and returns the tracker.
fn feed(t: &mut GestureTracker, hand: &Hand, frames: usize, dt: f32) {
    let mut buf = synth::buffer();
    for _ in 0..frames {
        synth::write(&mut buf, 0, hand);
        t.update_hands(&buf, dt);
        t.update_two_hand(dt);
    }
}

/// A pose buffer with `present` set and only the listed joints visible.
fn pose_buffer(joints: &[(usize, f32, f32, f32)]) -> Vec<f32> {
    let mut buf = vec![0.0f32; POSE_STRIDE];
    buf[0] = 1.0;
    for &(j, x, y, v) in joints {
        let o = 1 + j * 4;
        buf[o] = x;
        buf[o + 1] = y;
        buf[o + 2] = 0.0;
        buf[o + 3] = v;
    }
    buf
}

#[test]
fn ema_alpha_composes_across_frame_rates() {
    // The claim the whole filter rests on: the surviving fraction of an
    // error after 100 ms must not depend on how many steps it took to get
    // there. `1 - rate * dt` fails this; a half-life does not.
    let one = 1.0 - ema_alpha(0.045, 0.1);
    let mut many = 1.0f32;
    for _ in 0..10 {
        many *= 1.0 - ema_alpha(0.045, 0.01);
    }
    assert!((one - many).abs() < 1e-6, "{one} vs {many}");
    // And a half-life means what it says.
    assert!((ema_alpha(0.05, 0.05) - 0.5).abs() < 1e-6);
}

#[test]
fn ema_alpha_and_dt_survive_degenerate_input() {
    assert_eq!(ema_alpha(0.0, 0.016), 1.0, "zero half-life must snap");
    assert_eq!(ema_alpha(f32::NAN, 0.016), 1.0);
    assert!(ema_alpha(0.05, f32::INFINITY).is_finite());
    assert_eq!(clamp_dt(f32::NAN), 1.0 / 60.0);
    assert_eq!(clamp_dt(-5.0), MIN_DT);
    assert_eq!(clamp_dt(1e9), MAX_DT);
    assert!(adaptive_alpha(0.045, f32::NAN, DT).is_finite());
}

#[test]
fn landmark_filter_is_framerate_independent() {
    // Two trackers see the same step input, one at 100 Hz and one at 10 Hz.
    // After the same 100 ms of simulated time they must agree.
    let cfg = GestureConfig::default();
    let (mut fast, mut slow) = (GestureTracker::new(cfg), GestureTracker::new(cfg));
    let start = Hand::at(0.40, 0.5);
    let moved = Hand::at(0.42, 0.5);
    feed(&mut fast, &start, 1, 0.01);
    feed(&mut slow, &start, 1, 0.1);
    feed(&mut fast, &moved, 10, 0.01);
    feed(&mut slow, &moved, 1, 0.1);

    let (a, b) = (fast.hands()[0].palm[0], slow.hands()[0].palm[0]);
    let step = 0.02;
    // 2% of the step. Not zero: the one-euro cutoff reads the step as a
    // ten-times-higher speed at the shorter dt, so the two paths converge
    // at slightly different rates on the first frame after a discontinuity.
    assert!(
        (a - b).abs() < 0.02 * step,
        "100 Hz and 10 Hz disagree: {a} vs {b}"
    );
    // And prove filtering actually happened rather than both snapping.
    let target = slow.hands()[0].palm[0];
    assert!(target > 0.40, "filter did not move toward the target");
}

#[test]
fn landmark_filter_rejects_jitter() {
    // Alternating noise is the signal MediaPipe actually produces when a
    // hand is held still; the filter exists to remove it.
    let mut t = tracker();
    let mut buf = synth::buffer();
    let mut raw_span = 0.0f32;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for frame in 0..60 {
        let jitter = if frame % 2 == 0 { 0.004 } else { -0.004 };
        let hand = Hand::at(0.5 + jitter, 0.5);
        synth::write(&mut buf, 0, &hand);
        t.update_hands(&buf, DT);
        if frame >= 40 {
            raw_span = 0.008;
            lo = lo.min(t.hands()[0].palm[0]);
            hi = hi.max(t.hands()[0].palm[0]);
        }
    }
    let filtered_span = hi - lo;
    assert!(
        filtered_span * 1.8 < raw_span,
        "jitter barely attenuated: {filtered_span} vs {raw_span}"
    );
}

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

#[test]
fn velocity_tracks_a_known_sweep() {
    // 0.012 normalised units per frame at 30 Hz = 0.36 per second.
    let mut t = tracker();
    let mut buf = synth::buffer();
    let mut x = 0.2;
    for _ in 0..40 {
        synth::write(&mut buf, 0, &Hand::at(x, 0.5));
        t.update_hands(&buf, DT);
        x += 0.012;
    }
    let v = t.hands()[0].velocity;
    assert!(
        (v[0] - 0.36).abs() < 0.05,
        "expected ~0.36 units/s, got {}",
        v[0]
    );
    assert!(v[1].abs() < 0.02, "no vertical motion was applied");

    // A still hand must settle back to zero, or it would push the fluid
    // forever after one swipe.
    feed(&mut t, &Hand::at(x, 0.5), 40, DT);
    assert!(t.hands()[0].velocity[0].abs() < 0.02);
}

#[test]
fn angular_velocity_is_continuous_through_the_pi_seam() {
    // Two palms rotating about the frame centre at a constant rate. The
    // pair's angle crosses +/-pi once per revolution; differentiating the
    // wrapped angle would spike by 2*pi/dt exactly there, and that spike
    // drives the time warp.
    let mut t = tracker();
    let mut buf = synth::buffer();
    let omega = 3.0f32;
    let radius = 0.12;
    let mut worst = 0.0f32;
    let frames = 160;
    for frame in 0..frames {
        let theta = omega * frame as f32 * DT;
        let (c, s) = (theta.cos(), theta.sin());
        synth::write(&mut buf, 0, &Hand::at(0.5 + radius * c, 0.5 + radius * s));
        synth::write(&mut buf, 1, &Hand::at(0.5 - radius * c, 0.5 - radius * s));
        t.update_hands(&buf, DT);
        t.update_two_hand(DT);
        if frame > 30 {
            let av = t.two_hand().angular_velocity;
            assert!(av.is_finite(), "non-finite angular velocity");
            assert!(
                (av - omega).abs() < 0.5 * omega,
                "spike at frame {frame}: {av} rad/s vs {omega}"
            );
            worst = worst.max((av - omega).abs());
        }
    }
    assert!(t.two_hand().both_present);
    assert!(worst < 0.5 * omega, "worst deviation {worst}");
    // A rigid rotation changes no distance.
    assert!(t.two_hand().distance_velocity.abs() < 0.2);
    assert!(t.two_hand().angle.abs() <= core::f32::consts::PI + 1e-5);
}

#[test]
fn two_hand_distance_and_midpoint_track_the_palms() {
    let mut t = tracker();
    let mut buf = synth::buffer();
    // Hands separating by 0.01 each per frame, i.e. 0.6 units/s of spread.
    let mut half = 0.1;
    for _ in 0..40 {
        synth::write(&mut buf, 0, &Hand::at(0.5 - half, 0.5));
        synth::write(&mut buf, 1, &Hand::at(0.5 + half, 0.5));
        t.update_hands(&buf, DT);
        t.update_two_hand(DT);
        half += 0.01;
    }
    let two = *t.two_hand();
    assert!(two.both_present);
    assert!((two.distance - 2.0 * half).abs() < 0.05, "{}", two.distance);
    assert!(
        (two.distance_velocity - 0.6).abs() < 0.12,
        "separation rate {}",
        two.distance_velocity
    );
    assert!((two.midpoint[0] - 0.5).abs() < 0.02);

    // Losing one hand must zero the rates rather than freeze them.
    synth::clear(&mut buf, 1);
    for _ in 0..GestureConfig::default().release_frames {
        t.update_hands(&buf, DT);
        t.update_two_hand(DT);
    }
    assert!(!t.two_hand().both_present);
    assert_eq!(t.two_hand().angular_velocity, 0.0);
    assert_eq!(t.two_hand().distance_velocity, 0.0);
}

#[test]
fn low_visibility_pose_landmarks_do_not_move_the_center() {
    let mut t = tracker();
    let torso = |hip: (f32, f32, f32)| {
        pose_buffer(&[
            (PL_LEFT_SHOULDER, 0.45, 0.35, 0.95),
            (PL_RIGHT_SHOULDER, 0.55, 0.35, 0.95),
            (PL_LEFT_HIP, 0.45, 0.70, 0.95),
            (PL_RIGHT_HIP, hip.0, hip.1, hip.2),
        ])
    };
    // The right hip is never credible, so its position must never count.
    let ghost = torso((0.55, 0.70, 0.05));
    for _ in 0..30 {
        t.update_pose(&ghost, DT);
    }
    let before = t.pose().center;
    assert!(t.pose().present);
    assert!((t.pose().span - 0.1).abs() < 0.01, "span {}", t.pose().span);

    // Now hallucinate it into the corner of the frame.
    let yanked = torso((0.98, 0.02, 0.05));
    for _ in 0..30 {
        t.update_pose(&yanked, DT);
    }
    let after = t.pose().center;
    assert!(
        (after[0] - before[0]).abs() < 1e-4 && (after[1] - before[1]).abs() < 1e-4,
        "an invisible joint moved the centre: {before:?} -> {after:?}"
    );

    // The same joint *is* credible: now it must count, or the gate would be
    // ignoring visibility rather than respecting it.
    let seen = torso((0.98, 0.02, 0.95));
    for _ in 0..40 {
        t.update_pose(&seen, DT);
    }
    assert!(
        (t.pose().center[0] - before[0]).abs() > 0.05,
        "a visible joint should move the centre"
    );
}

#[test]
fn pose_center_and_velocity_track_the_torso() {
    let mut t = tracker();
    let mut x = 0.3;
    for _ in 0..60 {
        let buf = pose_buffer(&[
            (PL_LEFT_SHOULDER, x - 0.05, 0.35, 0.9),
            (PL_RIGHT_SHOULDER, x + 0.05, 0.35, 0.9),
            (PL_LEFT_HIP, x - 0.05, 0.65, 0.9),
            (PL_RIGHT_HIP, x + 0.05, 0.65, 0.9),
        ]);
        t.update_pose(&buf, DT);
        x += 0.004;
    }
    let pose = t.pose();
    assert!(
        (pose.center[0] - (x - 0.004)).abs() < 0.03,
        "{:?}",
        pose.center
    );
    assert!((pose.center[1] - 0.5).abs() < 0.02);
    // 0.004 per frame at 30 Hz.
    assert!(
        (pose.velocity[0] - 0.12).abs() < 0.03,
        "velocity {:?}",
        pose.velocity
    );
    assert!((pose.span - 0.1).abs() < 0.005);
}

#[test]
fn pose_absence_releases_only_after_the_hysteresis_window() {
    let cfg = GestureConfig::default();
    let mut t = GestureTracker::new(cfg);
    let buf = pose_buffer(&[
        (PL_LEFT_SHOULDER, 0.45, 0.35, 0.9),
        (PL_RIGHT_SHOULDER, 0.55, 0.35, 0.9),
    ]);
    for _ in 0..5 {
        t.update_pose(&buf, DT);
    }
    assert!(t.pose().present);
    for _ in 1..cfg.release_frames {
        t.update_pose(&[], DT);
        assert!(t.pose().present, "pose released on a single dropped frame");
    }
    t.update_pose(&[], DT);
    assert!(!t.pose().present);
}

#[test]
fn the_torso_centre_is_live_on_the_first_pose_frame() {
    // The centroid gate reads filtered visibility, so a filter that eases
    // in from zero leaves the centre at its [0.5, 0.5] default for the
    // first few frames — and 0.5 is not where this torso is.
    let mut t = tracker();
    t.update_pose(
        &pose_buffer(&[
            (PL_LEFT_SHOULDER, 0.25, 0.30, 1.0),
            (PL_RIGHT_SHOULDER, 0.35, 0.30, 1.0),
            (PL_LEFT_HIP, 0.25, 0.70, 1.0),
            (PL_RIGHT_HIP, 0.35, 0.70, 1.0),
        ]),
        DT,
    );
    let c = t.pose().center;
    assert!(
        (c[0] - 0.3).abs() < 1e-4 && (c[1] - 0.5).abs() < 1e-4,
        "first-frame centre is still the default: {c:?}"
    );
    assert_eq!(t.pose().velocity, [0.0, 0.0]);
}

#[test]
fn a_released_pose_reappears_without_a_velocity_spike() {
    let mut t = tracker();
    let torso = |x: f32| {
        pose_buffer(&[
            (PL_LEFT_SHOULDER, x - 0.05, 0.35, 0.95),
            (PL_RIGHT_SHOULDER, x + 0.05, 0.35, 0.95),
            (PL_LEFT_HIP, x - 0.05, 0.70, 0.95),
            (PL_RIGHT_HIP, x + 0.05, 0.70, 0.95),
        ])
    };
    for _ in 0..40 {
        t.update_pose(&torso(0.3), DT);
    }
    for _ in 0..60 {
        t.update_pose(&[], DT);
    }
    assert!(!t.pose().present);

    // The body is standing still somewhere else when tracking resumes.
    // Carrying `prev_center` across the gap would differentiate the jump.
    t.update_pose(&torso(0.7), DT);
    let c = t.pose().center;
    assert!(
        (c[0] - 0.7).abs() < 1e-3,
        "re-acquired centre slid in from the old position: {c:?}"
    );
    assert_eq!(
        t.pose().velocity,
        [0.0, 0.0],
        "a stationary body reported motion on re-acquisition"
    );
    for _ in 0..10 {
        t.update_pose(&torso(0.7), DT);
        assert!(
            t.pose().velocity[0].abs() < 0.02,
            "velocity spike after re-acquisition: {:?}",
            t.pose().velocity
        );
    }
}

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

#[test]
fn wrap_pi_folds_into_the_principal_branch() {
    use core::f32::consts::{PI, TAU};
    assert!((wrap_pi(0.25) - 0.25).abs() < 1e-6);
    assert!((wrap_pi(0.25 + TAU) - 0.25).abs() < 1e-5);
    assert!((wrap_pi(-0.25 - TAU) + 0.25).abs() < 1e-5);
    // The seam itself: a step from just under +pi to just over must read as
    // a small positive step, not as -2*pi.
    let step = wrap_pi((-PI + 0.05) - (PI - 0.05));
    assert!((step - 0.1).abs() < 1e-4, "seam step {step}");
    assert_eq!(wrap_pi(f32::NAN), 0.0);
    for k in -20..20 {
        assert!(wrap_pi(k as f32 * 1.7).abs() <= PI + 1e-5);
    }
}
