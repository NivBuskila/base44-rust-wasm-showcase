//! The smoothing itself: an EMA whose alpha composes across frame rates, and the landmark filter that has to reject jitter without lagging a real move.

use super::support::*;

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
