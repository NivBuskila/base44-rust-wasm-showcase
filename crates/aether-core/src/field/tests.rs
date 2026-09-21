use super::{Grid, VecField};

#[test]
fn sample_reproduces_lattice_values() {
    let mut g = Grid::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            g.set(x, y, (x * 10 + y) as f32);
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let s = g.sample(x as f32, y as f32);
            assert!(
                (s - (x * 10 + y) as f32).abs() < 1e-4,
                "lattice mismatch at {x},{y}: {s}"
            );
        }
    }
}

#[test]
fn sample_interpolates_linearly() {
    let mut g = Grid::new(4, 4);
    g.set(0, 0, 0.0);
    g.set(1, 0, 10.0);
    assert!((g.sample(0.5, 0.0) - 5.0).abs() < 1e-5);
    assert!((g.sample(0.25, 0.0) - 2.5).abs() < 1e-5);
}

#[test]
fn sample_clamps_outside_the_grid() {
    let mut g = Grid::new(4, 4);
    g.set(0, 0, 3.0);
    g.set(3, 3, 7.0);
    assert!((g.sample(-100.0, -100.0) - 3.0).abs() < 1e-6);
    assert!((g.sample(1e6, 1e6) - 7.0).abs() < 1e-6);
}

#[test]
fn sample_survives_nan() {
    let mut g = Grid::new(4, 4);
    g.set(0, 0, 1.0);
    // Must not panic, must not return NaN.
    assert!(g.sample(f32::NAN, f32::NAN).is_finite());
    assert!(g.sample(f32::INFINITY, 0.0).is_finite());
}

#[test]
fn splat_is_centred_and_local() {
    let mut g = Grid::new(32, 32);
    g.splat(16.0, 16.0, 4.0, 1.0);
    // Peak at the centre.
    assert!(g.get(16, 16) > g.get(18, 16));
    assert!(g.get(18, 16) > g.get(20, 16));
    // Nothing leaked far away.
    assert_eq!(g.get(0, 0), 0.0);
    assert_eq!(g.get(31, 31), 0.0);
    assert!(g.sum() > 0.0);
}

#[test]
fn splat_at_the_border_does_not_panic() {
    let mut g = Grid::new(16, 16);
    g.splat(0.0, 0.0, 6.0, 1.0);
    g.splat(15.0, 15.0, 6.0, 1.0);
    g.splat(-50.0, 100.0, 6.0, 1.0);
    assert!(g.get(0, 0) > 0.0);
}

#[test]
fn splat_ignores_non_finite_input() {
    let mut g = Grid::new(8, 8);
    g.splat(f32::NAN, 4.0, 2.0, 1.0);
    g.splat(4.0, 4.0, 2.0, f32::INFINITY);
    assert_eq!(g.sanitize(), 0, "no NaN should have been written");
    assert_eq!(g.sum(), 0.0);
}

#[test]
fn resample_preserves_a_constant_field() {
    let src = Grid::filled(16, 9, 0.75);
    let mut dst = Grid::new(64, 36);
    dst.resample_from(&src);
    for v in &dst.data {
        assert!((v - 0.75).abs() < 1e-5);
    }
}

#[test]
fn resample_preserves_a_linear_ramp() {
    let mut src = Grid::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            src.set(x, y, x as f32 / 15.0);
        }
    }
    let mut dst = Grid::new(31, 31);
    dst.resample_from(&src);
    // A bilinear resample must reproduce a linear ramp exactly.
    for y in 0..31 {
        for x in 0..31 {
            let expect = x as f32 / 30.0;
            assert!(
                (dst.get(x, y) - expect).abs() < 1e-4,
                "ramp broken at {x},{y}: {} vs {expect}",
                dst.get(x, y)
            );
        }
    }
}

#[test]
fn resample_same_size_is_a_copy() {
    let mut src = Grid::new(8, 8);
    src.set(3, 4, 9.0);
    let mut dst = Grid::new(8, 8);
    dst.resample_from(&src);
    assert_eq!(src, dst);
}

#[test]
fn blur_spreads_and_conserves_interior_mass() {
    let mut g = Grid::new(32, 32);
    g.set(16, 16, 1.0);
    let before = g.sum();
    let mut scratch = Grid::new(32, 32);
    g.blur(&mut scratch, 3);
    assert!((g.sum() - before).abs() < 1e-4, "mass changed");
    assert!(g.get(16, 16) < 1.0, "peak should have flattened");
    assert!(g.get(17, 16) > 0.0, "should have spread to the neighbour");
}

#[test]
fn blur_leaves_a_constant_field_alone() {
    let mut g = Grid::filled(16, 16, 0.4);
    let mut scratch = Grid::new(16, 16);
    g.blur(&mut scratch, 5);
    for v in &g.data {
        assert!((v - 0.4).abs() < 1e-6, "constant field changed: {v}");
    }
}

#[test]
fn gradient_matches_an_analytic_ramp() {
    let mut g = Grid::new(16, 16);
    for y in 0..16 {
        for x in 0..16 {
            g.set(x, y, 2.0 * x as f32 + 3.0 * y as f32);
        }
    }
    let (dx, dy) = g.gradient(8, 8);
    assert!((dx - 2.0).abs() < 1e-4, "dx {dx}");
    assert!((dy - 3.0).abs() < 1e-4, "dy {dy}");
}

#[test]
fn sanitize_scrubs_non_finite_cells() {
    let mut g = Grid::new(4, 4);
    g.data[0] = f32::NAN;
    g.data[1] = f32::INFINITY;
    g.data[2] = 1.0;
    assert_eq!(g.sanitize(), 2);
    assert_eq!(g.data[0], 0.0);
    assert_eq!(g.data[1], 0.0);
    assert_eq!(g.data[2], 1.0);
}

#[test]
fn fill_from_u8_normalizes() {
    let mut g = Grid::new(2, 2);
    g.fill_from_u8(&[0, 255, 128, 64]);
    assert_eq!(g.data[0], 0.0);
    assert_eq!(g.data[1], 1.0);
    assert!((g.data[2] - 0.5019608).abs() < 1e-6);
}

#[test]
fn vecfield_energy_is_zero_at_rest() {
    let f = VecField::new(8, 8);
    assert_eq!(f.energy(), 0.0);
    assert_eq!(f.max_speed(), 0.0);
}

#[test]
fn vecfield_energy_matches_hand_calculation() {
    let mut f = VecField::new(2, 2);
    f.u.data.fill(3.0);
    f.v.data.fill(4.0);
    // 0.5 * (9 + 16) = 12.5 per cell.
    assert!((f.energy() - 12.5).abs() < 1e-5);
    assert!((f.max_speed() - 5.0).abs() < 1e-5);
}
