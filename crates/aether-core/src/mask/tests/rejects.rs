//! What is not a body: an almost-solid frame, a speck, and degenerate input.

use super::support::*;

#[test]
fn an_almost_solid_mask_is_rejected() {
    let solid = vec![1.0f32; CELLS];
    let mut fresh = body(sharp());
    fresh.update_f32(&solid, W, H, false, DT30);
    assert_eq!(
        fresh.obstacle().sum(),
        0.0,
        "segmentation walled off the screen"
    );
    assert_eq!(fresh.coverage(), 0.0);
    assert!(!fresh.present());

    // And a bogus frame must not destroy a good silhouette.
    let mut m = body(sharp());
    m.update_f32(&body_rect(), W, H, false, DT30);
    let (good, cov) = (m.obstacle().sum(), m.coverage());
    m.update_f32(&solid, W, H, false, DT30);
    assert!(
        (m.obstacle().sum() - good).abs() < 1e-6,
        "bogus mask overwrote a good one"
    );
    assert!((m.coverage() - cov).abs() < 1e-6);

    // Just under the limit is a legitimate close-up body.
    let mut big = body(sharp());
    let rows = fy(0.8);
    big.update_f32(&rect_mask(0, W, 0, rows), W, H, false, DT30);
    assert!(big.present());
    let expected = rows as f32 / H as f32;
    assert!(
        (big.coverage() - expected).abs() < 1e-3,
        "coverage {} expected {expected}",
        big.coverage()
    );
}

#[test]
fn a_tiny_blob_is_noise_rather_than_a_body() {
    let mut m = body(sharp());
    m.update_f32(&rect_mask(100, 110, 40, 50), W, H, false, DT30);
    // 100 cells of 36864 is 0.27%, below the noise floor: the field has to
    // be empty, not merely "not present" (see the module invariant).
    assert!(!m.present());
    assert_eq!(m.obstacle().sum(), 0.0);
    assert_eq!(m.coverage(), 0.0);
    assert_eq!(m.edge_velocity().max_speed(), 0.0);
}

#[test]
fn degenerate_input_never_panics_and_never_writes_nan() {
    let mut m = body(MaskConfig::default());
    // Established body first, so there is state to corrupt.
    m.update_f32(&body_rect(), W, H, false, DT30);
    let good = m.obstacle().sum();

    m.update_f32(&[], 0, 0, false, DT30);
    m.update_f32(&[], 256, 144, false, DT30);
    m.update_f32(&[1.0], 1, 1, true, 0.0);
    m.update_f32(&[1.0, 0.0], 8, 8, false, f32::NAN);
    m.update_f32(&[0.5; 16], 4, 4, false, -5.0);
    m.update_f32(&rect_mask(0, 4, 0, 4), usize::MAX, usize::MAX, false, DT30);
    assert!(
        (m.obstacle().sum() - good).abs() < 1e-6,
        "an unusable mask was accepted"
    );

    // Garbage confidences, mixed with a real block.
    let mut junk = body_rect();
    for (i, v) in junk.iter_mut().enumerate() {
        match i % 5 {
            0 if *v == 0.0 => *v = f32::NAN,
            1 if *v == 0.0 => *v = f32::INFINITY,
            2 if *v == 0.0 => *v = -1e9,
            _ => {}
        }
    }
    m.update_f32(&junk, W, H, false, 1e9);
    assert!(all_finite(&m), "NaN reached the obstacle or velocity field");
    m.decay(f32::NAN);
    m.decay(-1.0);
    m.decay(1e9);
    assert!(all_finite(&m));
    assert!(m.staleness().is_finite());

    // A garbage config must not produce NaN either.
    m.set_config(MaskConfig {
        threshold: f32::NAN,
        half_life: f32::NAN,
        blur_passes: usize::MAX,
        push_gain: f32::INFINITY,
        fade_seconds: f32::NAN,
    });
    m.update_f32(&body_rect(), W, H, false, DT30);
    m.decay(1.0 / 60.0);
    assert!(all_finite(&m));
}
