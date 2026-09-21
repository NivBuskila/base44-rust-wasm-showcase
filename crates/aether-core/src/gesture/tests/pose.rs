//! The body: the torso centre, its velocity, low-visibility joints and the release hysteresis.

use super::support::*;

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
