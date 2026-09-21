//! The two hand-rolled field reads, checked against the general-purpose ones they replace.

use super::support::*;

#[test]
fn sample_uv_agrees_with_grid_sample() {
    let mut vel = VecField::new(9, 5);
    for y in 0..5 {
        for x in 0..9 {
            vel.u.set(x, y, (x * 3 + y) as f32);
            vel.v.set(x, y, (x as f32) * -0.5 + y as f32);
        }
    }
    for &(x, y) in &[
        (0.0, 0.0),
        (3.25, 1.75),
        (8.0, 4.0),
        (-5.0, 2.0),
        (100.0, 100.0),
        (f32::NAN, 1.0),
    ] {
        let fused = sample_uv(&vel, x, y);
        let reference = vel.sample(x, y);
        assert_eq!(fused, reference, "mismatch at ({x}, {y})");
    }
}

#[test]
fn curl_matches_an_analytic_rigid_rotation() {
    let vel = rot_field(32, 32, 16.0, 16.0, 2.5);
    // omega = dv/dx - du/dy = 2 * 2.5 in the interior.
    assert!((curl_at(&vel, 16.0, 16.0) - 5.0).abs() < 1e-4);
    assert!((curl_at(&vel, 4.2, 27.8) - 5.0).abs() < 1e-4);
    // A pure translation has no curl.
    let flat = const_field(16, 16, 9.0, -4.0);
    assert_eq!(curl_at(&flat, 8.0, 8.0), 0.0);
    // Border lookups must clamp instead of panicking.
    assert!(curl_at(&vel, 0.0, 0.0).is_finite());
    assert!(curl_at(&vel, 31.0, 31.0).is_finite());
    assert!(curl_at(&vel, f32::NAN, 1e9).is_finite());
}
