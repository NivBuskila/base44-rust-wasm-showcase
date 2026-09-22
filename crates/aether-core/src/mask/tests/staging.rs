//! Ingest: a staged mask becomes the cells it describes, at any resolution, mirrored exactly once, with speckle closed out.

use super::support::*;

#[test]
fn a_rectangle_becomes_the_same_rectangle_of_solid_cells() {
    let mut m = body(sharp());
    m.update_f32(&body_rect(), W, H, false, DT30);

    for y in 0..H {
        for x in 0..W {
            let inside = (100..140).contains(&x) && (40..100).contains(&y);
            let want = if inside { 1.0 } else { 0.0 };
            assert_eq!(m.obstacle().get(x, y), want, "cell {x},{y}");
        }
    }
    // The close must not erode the shape, corners included.
    let expect = (40 * 60) as f32 / CELLS as f32;
    assert!(
        (m.coverage() - expect).abs() < 1e-6,
        "coverage {} vs {expect}",
        m.coverage()
    );
    assert!(m.present());
}

#[test]
fn a_smaller_mask_is_stretched_across_the_whole_grid() {
    // The mask covers the camera frame, and the frame is the screen, so a
    // block over the middle half of a square mask must land over the middle
    // half of a 16:9 grid in *both* axes.
    let (mw, mh) = (64usize, 64usize);
    let mut src = vec![0.0f32; mw * mh];
    for y in 16..48 {
        for x in 16..48 {
            src[y * mw + x] = 1.0;
        }
    }
    let mut m = body(sharp());
    m.update_f32(&src, mw, mh, false, DT30);

    // The 0.5 crossing sits at mask coordinate 15.5 and 47.5, i.e. grid
    // x = 15.5 * 255/63 = 62.7 .. 192.2 and y = 35.2 .. 107.8.
    assert_eq!(m.obstacle().get(70, 70), 1.0);
    assert_eq!(m.obstacle().get(185, 70), 1.0);
    assert_eq!(m.obstacle().get(56, 70), 0.0);
    assert_eq!(m.obstacle().get(198, 70), 0.0);
    assert_eq!(m.obstacle().get(128, 30), 0.0, "block leaked upwards");
    assert_eq!(m.obstacle().get(128, 114), 0.0, "block leaked downwards");
    // ~130 x ~72 cells of 256 x 144.
    assert!(
        (0.23..0.28).contains(&m.coverage()),
        "coverage {}",
        m.coverage()
    );
}

#[test]
fn mirror_flips_the_silhouette_horizontally() {
    let (mw, mh) = (64usize, 64usize);
    let mut src = vec![0.0f32; mw * mh];
    for y in 0..mh {
        for x in 0..16 {
            src[y * mw + x] = 1.0;
        }
    }
    let mut plain = body(sharp());
    plain.update_f32(&src, mw, mh, false, DT30);
    let mut flipped = body(sharp());
    flipped.update_f32(&src, mw, mh, true, DT30);

    let mid = H / 2;
    let right = W - 11;
    assert_eq!(plain.obstacle().get(10, mid), 1.0);
    assert_eq!(plain.obstacle().get(right, mid), 0.0);
    assert_eq!(
        flipped.obstacle().get(10, mid),
        0.0,
        "mirrored mask stayed on the left"
    );
    assert_eq!(
        flipped.obstacle().get(right, mid),
        1.0,
        "mirrored mask never reached the right"
    );
    assert!((plain.coverage() - flipped.coverage()).abs() < 1e-6);
}

#[test]
fn single_cell_speckle_does_not_survive_the_close() {
    let speckle = [(10usize, 10usize), (200, 20), (30, 120), (240, 130)];
    let mut src = body_rect();
    for &(x, y) in &speckle {
        src[y * W + x] = 1.0;
    }
    let mut m = body(sharp());
    m.update_f32(&src, W, H, false, DT30);

    for &(x, y) in &speckle {
        assert_eq!(
            m.obstacle().get(x, y),
            0.0,
            "speckle at {x},{y} reached the solver"
        );
    }
    let (bx, by) = body_interior();
    assert_eq!(m.obstacle().get(bx, by), 1.0, "the body was eroded away");
    let expect = (40 * 60) as f32 / CELLS as f32;
    assert!((m.coverage() - expect).abs() < 1e-6);
}

#[test]
fn the_u8_path_matches_the_float_path() {
    let float = body_rect();
    let bytes: Vec<u8> = float.iter().map(|&v| (v * 255.0) as u8).collect();
    let mut a = body(sharp());
    let mut b = body(sharp());
    a.update_f32(&float, W, H, false, DT30);
    b.update_u8(&bytes, W, H, false, DT30);
    assert_eq!(a.obstacle().data, b.obstacle().data);
    assert_eq!(a.coverage(), b.coverage());
}

#[test]
fn a_mask_resolution_change_is_handled() {
    let mut m = body(sharp());
    for &(mw, mh) in &[(64usize, 64usize), (32, 48), (256, 144), (17, 9)] {
        let mut src = vec![0.0f32; mw * mh];
        for y in (mh / 4)..(3 * mh / 4) {
            for x in (mw / 4)..(3 * mw / 4) {
                src[y * mw + x] = 1.0;
            }
        }
        m.update_f32(&src, mw, mh, false, DT30);
        assert!(m.present(), "{mw}x{mh} mask produced no body");
        assert!(
            (0.15..0.35).contains(&m.coverage()),
            "{mw}x{mh} coverage {}",
            m.coverage()
        );
        assert!(all_finite(&m));
    }
}
