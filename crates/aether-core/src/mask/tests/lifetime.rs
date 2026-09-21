//! The mask's life over time: framerate-independent smoothing, the fade of a lost body and the 30 Hz mask against a 60 Hz loop.

use super::support::*;

#[test]
fn temporal_smoothing_is_framerate_independent() {
    let mask = rect_mask(80, 176, 30, 120);
    let empty = vec![0.0f32; CELLS];
    let mut coarse = body(MaskConfig::default());
    let mut fine = body(MaskConfig::default());
    coarse.update_f32(&mask, W, H, false, DT30);
    fine.update_f32(&mask, W, H, false, DT30);

    // One 100 ms decay against ten 10 ms decays of the same input.
    coarse.update_f32(&empty, W, H, false, 0.1);
    for _ in 0..10 {
        fine.update_f32(&empty, W, H, false, 0.01);
    }

    let worst = coarse
        .confidence()
        .data
        .iter()
        .zip(&fine.confidence().data)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-4, "EMA differs by {worst} between frame rates");
    // 2^(-0.1 / 0.08) = 0.42; if it had not decayed the test proves nothing.
    let peak = coarse.confidence().max_abs();
    assert!((0.40..0.44).contains(&peak), "peak confidence {peak}");
}

#[test]
fn a_lost_body_fades_out_completely_and_clears_present() {
    let cfg = MaskConfig::default();
    let mut m = body(cfg);
    m.update_f32(&body_rect(), W, H, false, DT30);
    m.update_f32(&body_rect_shifted(2), W, H, false, DT30);
    let full = m.obstacle().sum();
    assert!(full > 0.0);
    assert!(m.edge_velocity().max_speed() > 0.0);

    // Halfway through the fade the wall is still there, weaker, and still
    // reported as present so the engine keeps re-uploading it.
    let half_steps = (0.5 * cfg.fade_seconds / (1.0 / 60.0)) as usize;
    for _ in 0..half_steps {
        m.decay(1.0 / 60.0);
    }
    assert!(m.present(), "present dropped before the field was empty");
    let mid = m.obstacle().sum();
    assert!(mid < full && mid > 0.0, "obstacle {mid} vs {full}");
    assert!(m.coverage() > 0.0);

    while m.staleness() < cfg.fade_seconds {
        m.decay(1.0 / 60.0);
    }
    assert_eq!(m.obstacle().sum(), 0.0, "a wall outlived the fade");
    assert_eq!(m.edge_velocity().max_speed(), 0.0);
    assert_eq!(m.coverage(), 0.0);
    assert!(!m.present());
    // And it stays gone.
    for _ in 0..600 {
        m.decay(1.0 / 60.0);
    }
    assert!(all_finite(&m) && m.staleness().is_finite());
}

#[test]
fn the_fade_is_framerate_independent() {
    let mask = body_rect();
    let mut coarse = body(MaskConfig::default());
    let mut fine = body(MaskConfig::default());
    coarse.update_f32(&mask, W, H, false, DT30);
    fine.update_f32(&mask, W, H, false, DT30);
    for _ in 0..12 {
        coarse.decay(1.0 / 60.0);
    }
    for _ in 0..48 {
        fine.decay(1.0 / 240.0);
    }
    let (a, b) = (coarse.obstacle().sum(), fine.obstacle().sum());
    assert!(a > 0.0);
    assert!((a - b).abs() < 1e-3 * a, "fade diverged: {a} vs {b}");
}

#[test]
fn a_30hz_mask_against_a_60hz_loop_barely_dips() {
    // Two render frames of staleness is the normal case, not a lost body:
    // a linear fade would pulse the obstacle by ~10% every mask period.
    let mask = body_rect();
    let mut m = body(MaskConfig::default());
    m.update_f32(&mask, W, H, false, DT30);
    let full = m.obstacle().sum();
    m.decay(1.0 / 60.0);
    m.decay(1.0 / 60.0);
    assert!(
        m.obstacle().sum() > 0.97 * full,
        "obstacle dipped to {} of {full} between mask frames",
        m.obstacle().sum()
    );
    m.update_f32(&mask, W, H, false, DT30);
    assert!((m.obstacle().sum() - full).abs() < 1e-6, "not restored");
}

#[test]
fn fade_envelope_spans_exactly_fade_seconds() {
    assert_eq!(fade_envelope(0.0, 0.35), 1.0);
    assert_eq!(fade_envelope(0.35, 0.35), 0.0);
    assert_eq!(fade_envelope(10.0, 0.35), 0.0);
    assert_eq!(fade_envelope(f32::MAX, 0.35), 0.0);
    assert_eq!(fade_envelope(f32::NAN, 0.35), 0.0);
    // Zero and NaN fade windows retire the body at once rather than never.
    assert_eq!(fade_envelope(1e-6, 0.0), 0.0);
    assert_eq!(fade_envelope(1e-6, f32::NAN), 0.0);
    let mut prev = 1.0;
    for i in 0..=35 {
        let v = fade_envelope(i as f32 * 0.01, 0.35);
        assert!(
            v <= prev && (0.0..=1.0).contains(&v),
            "not monotonic at {i}"
        );
        prev = v;
    }
}

#[test]
fn reset_forgets_everything() {
    let mut m = body(MaskConfig::default());
    m.update_f32(&body_rect(), W, H, false, DT30);
    m.update_f32(&body_rect_shifted(4), W, H, false, DT30);
    assert!(m.present() && m.obstacle().sum() > 0.0);
    m.reset();
    assert!(!m.present());
    assert_eq!(m.obstacle().sum(), 0.0);
    assert_eq!(m.coverage(), 0.0);
    assert_eq!(m.edge_velocity().max_speed(), 0.0);
    assert_eq!(m.confidence().sum(), 0.0);
    // A fade started before the reset must not resurrect anything.
    m.decay(1.0 / 60.0);
    assert_eq!(m.obstacle().sum(), 0.0);
    // And the next body still engages on its first frame.
    m.update_f32(&body_rect(), W, H, false, DT30);
    assert!(m.present());
}

#[test]
fn ema_keep_composes_and_degrades_safely() {
    let one = ema_keep(0.08, 0.1);
    let mut many = 1.0;
    for _ in 0..10 {
        many *= ema_keep(0.08, 0.01);
    }
    assert!((one - many).abs() < 1e-6, "{one} vs {many}");
    // Exactly one half-life keeps half.
    assert!((ema_keep(0.08, 0.08) - 0.5).abs() < 1e-6);
    // Garbage means "no smoothing", never "never update".
    assert_eq!(ema_keep(0.0, 0.016), 0.0);
    assert_eq!(ema_keep(-1.0, 0.016), 0.0);
    assert_eq!(ema_keep(f32::NAN, 0.016), 0.0);
    assert_eq!(ema_keep(f32::INFINITY, 0.016), 0.0);
}

#[test]
fn sane_dt_is_finite_and_positive() {
    assert_eq!(sane_dt(f32::NAN), NOMINAL_DT);
    assert_eq!(sane_dt(f32::INFINITY), NOMINAL_DT);
    assert_eq!(sane_dt(0.0), MIN_DT);
    assert_eq!(sane_dt(-1.0), MIN_DT);
    assert_eq!(sane_dt(1e9), MAX_DT);
    assert_eq!(sane_dt(DT30), DT30);
}
