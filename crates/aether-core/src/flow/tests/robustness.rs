//! Degenerate input, reconfiguration mid-stream and the level cap.

use super::support::*;

#[test]
fn flat_image_produces_no_nan() {
    // The degenerate case: Ix = Iy = 0 everywhere, so the Horn-Schunck
    // denominator is alpha^2 alone and a zero alpha divides by zero.
    for alpha in [0.0, 12.0, f32::NAN] {
        let cfg = FlowConfig {
            alpha,
            ..Default::default()
        };
        let mut f = OpticalFlow::new(W, H, cfg);
        f.update(&vec![128; W * H], DT);
        f.update(&vec![130; W * H], DT);
        assert!(
            f.flow().u.data.iter().all(|v| v.is_finite()),
            "non-finite flow at alpha {alpha}"
        );
        assert!(f.flow().v.data.iter().all(|v| v.is_finite()));
        assert!(f.motion_energy().is_finite());
        assert!(f.flow().max_speed() < 1e-3, "flat image invented motion");
    }
}

#[test]
fn uniform_noise_is_finite_and_bounded() {
    let mut rng = Rng::new(13);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    for _ in 0..12 {
        let frame: Vec<u8> = (0..W * H).map(|_| (rng.next_f32() * 255.0) as u8).collect();
        f.update(&frame, DT);
        assert!(f.flow().u.data.iter().all(|v| v.is_finite()));
        assert!(f.flow().v.data.iter().all(|v| v.is_finite()));
        assert!(
            f.flow().max_speed() <= MAX_FLOW + 1e-3,
            "unbounded flow: {}",
            f.flow().max_speed()
        );
        assert!(f.motion_energy().is_finite());
    }
}

#[test]
fn a_long_run_of_hostile_input_stays_bounded() {
    // Everything the browser can throw at once: garbage dt, config churn
    // mid-stream, resets between frames, moving and static content. The
    // state machine around the reused pyramid is the thing under test —
    // a stale cached level would show up as flow on a static frame.
    let t = Texture::new(42);
    let mut rng = Rng::new(99);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    for step in 0..150 {
        let dt = match step % 5 {
            0 => 1.0 / 144.0,
            1 => 1.0 / 30.0,
            2 => 0.0,
            3 => -0.5,
            _ => f32::NAN,
        };
        if step % 37 == 0 {
            f.reset();
        }
        if step % 23 == 0 {
            f.set_config(FlowConfig {
                levels: (step / 23) % 7,
                iters: step % 40,
                alpha: rng.next_f32() * 30.0,
                noise_floor: rng.next_f32() * 0.2,
            });
        }
        let frame = t.translated(W, H, rng.signed() * 3.0, rng.signed() * 3.0);
        f.update(&frame, dt);
        assert!(
            f.flow().u.data.iter().all(|v| v.is_finite())
                && f.flow().v.data.iter().all(|v| v.is_finite()),
            "non-finite flow at step {step}"
        );
        assert!(
            f.flow().max_speed() <= MAX_FLOW + 1e-3,
            "step {step} produced {}",
            f.flow().max_speed()
        );
        assert!(f.motion_energy().is_finite() && f.motion_energy() >= 0.0);
        assert!(f.levels() >= 1 && f.levels() <= MAX_LEVELS);
    }
    // Coming to rest must actually mean rest, not a decaying tail.
    let still = t.still(W, H);
    f.set_config(FlowConfig::default());
    f.update(&still, DT);
    f.update(&still, DT);
    assert!(
        f.flow().max_speed() < 1e-5,
        "did not settle: {}",
        f.flow().max_speed()
    );
}

#[test]
fn reset_clears_history_and_flow() {
    let t = Texture::new(15);
    let mut f = flow_of(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), DT);
    assert!(f.flow().max_speed() > 0.0);
    f.reset();
    assert!(!f.has_history());
    assert_eq!(f.flow().max_speed(), 0.0);
    assert_eq!(f.motion_energy(), 0.0);
    assert_eq!(f.dominant_motion(), (0.0, 0.0));
    // The first frame after a reset is history again, not motion.
    f.update(&t.still(W, H), DT);
    assert!(f.has_history());
    assert_eq!(f.flow().max_speed(), 0.0);
}

#[test]
fn level_count_is_capped_by_resolution() {
    assert_eq!(effective_levels(128, 72, 3), 3);
    // 72 >> 5 is 2 samples tall, so the 6th level is dropped.
    assert_eq!(effective_levels(128, 72, 99), 5);
    assert_eq!(effective_levels(1024, 1024, 99), MAX_LEVELS);
    assert_eq!(effective_levels(8, 8, 4), 2);
    assert_eq!(effective_levels(4, 4, 5), 1);
    assert_eq!(effective_levels(2, 2, 3), 1);
    assert_eq!(effective_levels(128, 72, 0), 1);
}

#[test]
fn reconfiguring_mid_stream_stays_correct() {
    // Changing the level count invalidates the cached pyramid; if that
    // cache is not dropped the next frame differences mismatched images.
    let t = Texture::new(17);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    f.update(&t.still(W, H), DT);
    f.update(&t.translated(W, H, 2.0, 0.0), DT);
    f.set_config(FlowConfig {
        levels: 2,
        ..Default::default()
    });
    assert_eq!(f.levels(), 2);
    f.update(&t.translated(W, H, 4.0, 0.0), DT);
    let (u, v) = mean_interior(f.flow(), 16);
    assert!(u > 0.5 * 2.0 / DT, "lost motion after reconfigure: {u}");
    assert!(v.abs() < 0.35 * 2.0 / DT, "spurious v: {v}");
}

#[test]
fn degenerate_tiny_grid_does_not_panic() {
    // Not reachable from the browser, but the solver must not assume it
    // has room for a pyramid.
    let mut f = OpticalFlow::new(0, 0, FlowConfig::default());
    assert_eq!((f.width(), f.height()), (2, 2));
    assert_eq!(f.levels(), 1);
    f.update(&[0, 0, 0, 0], DT);
    f.update(&[255, 0, 0, 255], DT);
    assert!(f.flow().max_speed().is_finite());
    assert!(f.motion_energy().is_finite());
}
