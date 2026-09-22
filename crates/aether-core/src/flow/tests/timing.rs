//! Per-second, not per-frame: the reported speed may not depend on the frame rate, and an extreme dt stays clamped.

use super::support::*;

#[test]
fn flow_is_per_second_not_per_frame() {
    let t = Texture::new(11);
    let (a, b) = (t.still(W, H), t.translated(W, H, 2.0, 0.0));
    let slow = flow_of(&a, &b, 1.0 / 30.0);
    let fast = flow_of(&a, &b, 1.0 / 60.0);
    let (su, _) = mean_interior(slow.flow(), 16);
    let (fu, _) = mean_interior(fast.flow(), 16);
    assert!(su > 1.0, "baseline flow too small to compare: {su}");
    let ratio = fu / su;
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "halving dt should double the flow, got {ratio}x"
    );
}

#[test]
fn the_reported_speed_does_not_depend_on_the_frame_rate() {
    // The same physical 60 cells/second, sampled at four frame rates. Any
    // bias that is a *per-frame* constant — a subtracted noise floor, say —
    // becomes `constant / dt` in the published cells/second and shows up
    // here as a spread that tracks the frame rate.
    const SPEED: f32 = 60.0;
    let t = Texture::new(40);
    let still = t.still(W, H);
    for fps in [15.0f32, 30.0, 60.0, 144.0] {
        let dt = 1.0 / fps;
        let f = flow_of(&still, &t.translated(W, H, SPEED * dt, 0.0), dt);
        let (u, v) = mean_interior(f.flow(), 16);
        assert!(
            (u - SPEED).abs() < 0.06 * SPEED,
            "{fps} fps reported {u} cells/s for a true {SPEED}"
        );
        assert!(v.abs() < 0.1 * SPEED, "{fps} fps invented v {v}");
    }
}

#[test]
fn extreme_dt_stays_finite_and_clamped() {
    let t = Texture::new(14);
    let (a, b) = (t.still(W, H), t.translated(W, H, 3.0, 1.0));
    for dt in [f32::NAN, f32::INFINITY, 0.0, -1.0, 1e-9, 1e9] {
        let f = flow_of(&a, &b, dt);
        assert!(
            f.flow().max_speed() <= MAX_FLOW + 1e-3,
            "dt {dt} produced {}",
            f.flow().max_speed()
        );
        assert!(f.flow().u.data.iter().all(|v| v.is_finite()), "dt {dt}");
        assert!(f.motion_energy().is_finite(), "dt {dt}");
        // A negative or zero dt must never invert the direction of time.
        assert!(f.dominant_motion().0 >= 0.0, "dt {dt} flipped the sign");
    }
}
