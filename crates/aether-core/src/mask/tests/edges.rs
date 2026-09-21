//! The edge velocity the body pushes the fluid with: direction, rest, per-second scaling and its ceiling.

use super::support::*;

#[test]
fn a_silhouette_moving_right_pushes_right_at_both_edges() {
    let cfg = MaskConfig {
        half_life: 0.0,
        push_gain: 1.0,
        ..MaskConfig::default()
    };
    let mut m = body(cfg);
    m.update_f32(&body_rect(), W, H, false, DT30);
    m.update_f32(&body_rect_shifted(2), W, H, false, DT30);

    let e = m.edge_velocity();
    let row = 70;
    // Leading edge: one mask unit of advance in dt seconds, gain 1.
    assert!(
        (e.u.get(141, row) - 30.0).abs() < 1e-3,
        "leading edge {} (expected +30 cells/s)",
        e.u.get(141, row)
    );
    assert!(
        e.u.get(101, row) > 0.0,
        "trailing edge pushed {} (expected right)",
        e.u.get(101, row)
    );
    // A horizontal edge has no vertical component at mid height.
    assert!(e.v.get(141, row).abs() < 1e-6);
    // Zero deep inside the body and out in empty space.
    assert_eq!(e.u.get(120, row), 0.0, "pushed deep inside the body");
    assert_eq!(e.v.get(120, row), 0.0);
    assert_eq!(e.u.get(20, row), 0.0, "pushed far from the silhouette");
    assert_eq!(e.u.get(220, row), 0.0);
    // And the band is narrow: nothing three cells past the new edge.
    assert_eq!(e.u.get(145, row), 0.0);
    assert!(
        e.u.data.iter().sum::<f32>() > 0.0,
        "net push is not rightward"
    );
}

#[test]
fn a_stationary_silhouette_produces_no_edge_velocity() {
    for cfg in [sharp(), MaskConfig::default()] {
        let mut m = body(cfg);
        let mask = body_rect();
        for _ in 0..8 {
            m.update_f32(&mask, W, H, false, DT30);
        }
        assert!(
            m.edge_velocity().max_speed() < 1e-3,
            "still body pushed at {} cells/s",
            m.edge_velocity().max_speed()
        );
        assert!(m.present());
    }
}

#[test]
fn edge_velocity_is_per_second_not_per_frame() {
    let cfg = MaskConfig {
        half_life: 0.0,
        push_gain: 1.0,
        ..MaskConfig::default()
    };
    let mut slow = body(cfg);
    let mut fast = body(cfg);
    for (m, dt) in [(&mut slow, DT30), (&mut fast, DT30 * 0.5)] {
        m.update_f32(&body_rect(), W, H, false, dt);
        m.update_f32(&body_rect_shifted(2), W, H, false, dt);
    }
    let (a, b) = (
        slow.edge_velocity().u.get(141, 70),
        fast.edge_velocity().u.get(141, 70),
    );
    assert!(a > 0.0);
    assert!(
        (b - 2.0 * a).abs() < 1e-3,
        "same displacement in half the time gave {b}, expected 2 * {a}"
    );
}

#[test]
fn edge_velocity_is_bounded_even_for_an_absurd_config() {
    let mut m = body(MaskConfig {
        half_life: 0.0,
        push_gain: 1e6,
        ..MaskConfig::default()
    });
    m.update_f32(&body_rect(), W, H, false, 1e-9);
    m.update_f32(&body_rect_shifted(20), W, H, false, 1e-9);
    assert!(all_finite(&m));
    assert!(
        m.edge_velocity().max_speed() <= MAX_EDGE_SPEED + 1e-3,
        "unclamped boundary speed {}",
        m.edge_velocity().max_speed()
    );
}
