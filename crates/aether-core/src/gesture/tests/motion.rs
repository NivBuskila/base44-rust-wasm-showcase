//! Velocities: linear, angular through the pi seam, and the two-hand geometry.

use super::support::*;

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
