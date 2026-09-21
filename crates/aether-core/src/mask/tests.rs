//! Mask pipeline tests.

use super::*;

/// Grid the tests run `BodyMask` on, deliberately **not**
/// `config::W/H`.
///
/// `BodyMask::new` takes its dimensions explicitly, so a unit test for it
/// has no business depending on the app's current display resolution — and
/// these tests pick their cells for a reason (clear of the walls, straddling
/// a specific edge). Tying them to a tunable meant that changing the fluid
/// grid for the frame budget turned every carefully chosen coordinate into
/// either an out-of-bounds index or a different part of the frame.
const W: usize = 256;
const H: usize = 144;

const CELLS: usize = W * H;
const DT30: f32 = 1.0 / 30.0;

/// Config with the temporal EMA disabled, so one update lands the raw mask
/// exactly and a test's arithmetic stays analytic.
fn sharp() -> MaskConfig {
    MaskConfig {
        half_life: 0.0,
        ..MaskConfig::default()
    }
}

fn body(cfg: MaskConfig) -> BodyMask {
    BodyMask::new(W, H, cfg)
}

/// A mask at grid resolution with a filled rectangle. Same resolution means
/// the resample is a copy, so cell indices are directly comparable.
///
/// Coordinates are clamped: these tests are written against grid fractions
/// (see [`fx`] / [`fy`]) rather than absolute cells, and a clamp here means
/// a future resolution change degrades a rectangle instead of panicking
/// with an out-of-bounds index.
fn rect_mask(x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; CELLS];
    for y in y0..y1.min(H) {
        for x in x0..x1.min(W) {
            m[y * W + x] = 1.0;
        }
    }
    m
}

/// Vertical grid coordinate from a fraction of the frame height, for the
/// tests whose geometry is naturally proportional (a mask that covers most
/// of the frame) rather than a specific cell.
fn fy(frac: f32) -> usize {
    ((H as f32 * frac) as usize).min(H - 1)
}

/// The standard test silhouette: a solid block left of centre, well clear
/// of every domain wall so border effects cannot be mistaken for the
/// behaviour under test.
fn body_rect() -> Vec<f32> {
    rect_mask(100, 140, 40, 100)
}

/// [`body_rect`] translated `cells` to the right, for boundary-motion tests.
fn body_rect_shifted(cells: usize) -> Vec<f32> {
    rect_mask(100 + cells, 140 + cells, 40, 100)
}

/// A point inside [`body_rect`], safely away from its edges.
fn body_interior() -> (usize, usize) {
    (120, 70)
}

fn all_finite(m: &BodyMask) -> bool {
    m.obstacle()
        .data
        .iter()
        .chain(m.confidence().data.iter())
        .chain(m.edge_velocity().u.data.iter())
        .chain(m.edge_velocity().v.data.iter())
        .all(|v| v.is_finite())
}

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
fn sane_dt_is_finite_and_positive() {
    assert_eq!(sane_dt(f32::NAN), NOMINAL_DT);
    assert_eq!(sane_dt(f32::INFINITY), NOMINAL_DT);
    assert_eq!(sane_dt(0.0), MIN_DT);
    assert_eq!(sane_dt(-1.0), MIN_DT);
    assert_eq!(sane_dt(1e9), MAX_DT);
    assert_eq!(sane_dt(DT30), DT30);
}
