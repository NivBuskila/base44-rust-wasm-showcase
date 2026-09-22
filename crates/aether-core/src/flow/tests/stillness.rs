//! The other half of being right: a still camera must read as still, including under sensor noise, or the fluid is driven by nothing.

use super::support::*;

#[test]
fn duplicate_frame_is_perfectly_still() {
    let t = Texture::new(8);
    let frame = t.still(W, H);
    let f = flow_of(&frame, &frame, DT);
    assert!(f.has_history());
    assert!(
        f.flow().max_speed() < 1e-6,
        "duplicate frame moved: {}",
        f.flow().max_speed()
    );
    assert!(f.motion_energy() < 1e-6);
    assert_eq!(f.dominant_motion().0, 0.0);
}

#[test]
fn first_frame_has_no_history_and_no_flow() {
    let t = Texture::new(9);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    assert!(!f.has_history());
    f.update(&t.still(W, H), DT);
    assert!(
        f.has_history(),
        "second frame should have something to diff"
    );
    assert_eq!(f.flow().max_speed(), 0.0);
    assert_eq!(f.motion_energy(), 0.0);
}

#[test]
fn a_static_scene_never_accumulates_drift() {
    let t = Texture::new(10);
    let frame = t.still(W, H);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    for _ in 0..200 {
        f.update(&frame, DT);
        assert!(
            f.flow().max_speed() < 1e-5,
            "drifted to {}",
            f.flow().max_speed()
        );
    }
    assert_eq!(f.dominant_motion(), (0.0, 0.0));
}

#[test]
fn noise_floor_gates_below_and_passes_above_unchanged() {
    let t = Texture::new(12);
    let (a, b) = (t.still(W, H), t.translated(W, H, 2.0, 0.0));
    let open = {
        let cfg = FlowConfig {
            noise_floor: 0.0,
            ..Default::default()
        };
        let mut f = OpticalFlow::new(W, H, cfg);
        f.update(&a, DT);
        f.update(&b, DT);
        f
    };
    let floor = 1.0; // cells/frame, i.e. half the real motion
    let gated = {
        let cfg = FlowConfig {
            noise_floor: floor,
            ..Default::default()
        };
        let mut f = OpticalFlow::new(W, H, cfg);
        f.update(&a, DT);
        f.update(&b, DT);
        f
    };
    // The three regimes of the deadband, as behaviour rather than formula:
    // fully rejected at or below the floor, untouched once clear of it, and
    // monotonically attenuated in between. "Untouched above" is the part
    // that matters: a floor that is *subtracted* instead biases every
    // magnitude by `floor`, and since the floor is per-frame that bias is
    // `floor / dt` in the published cells/second — frame-rate dependent.
    let mut below = 0;
    let mut above = 0;
    for i in 0..W * H {
        let (u0, v0) = (open.flow().u.data[i], open.flow().v.data[i]);
        let raw = (u0 * u0 + v0 * v0).sqrt() * DT;
        if raw >= MAX_STEP - 1e-3 {
            continue; // Clamped, so the deadband is not the only effect.
        }
        let (u1, v1) = (gated.flow().u.data[i], gated.flow().v.data[i]);
        let got = (u1 * u1 + v1 * v1).sqrt() * DT;
        assert!(
            got <= raw + 1e-3,
            "cell {i}: the floor amplified {raw} to {got}"
        );
        if raw <= floor {
            assert!(got == 0.0, "cell {i}: {raw} <= floor {floor} gave {got}");
            below += 1;
        } else if raw >= 2.0 * floor {
            assert!(
                (got - raw).abs() < 1e-3,
                "cell {i}: {raw} is clear of floor {floor} but was cut to {got}"
            );
            above += 1;
        }
    }
    assert!(
        below > 0 && above > 0,
        "only {below}/{above} cells; vacuous"
    );
}

#[test]
fn sensor_noise_on_a_still_camera_does_not_drive_the_fluid() {
    // The reason the noise floor exists: a static scene through a real
    // sensor jitters by a couple of LSB every frame, and that must not
    // read as motion.
    let t = Texture::new(19);
    let base = t.still(W, H);
    let mut rng = Rng::new(20);
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    let moving = flow_of(&base, &t.translated(W, H, 2.0, 0.0), DT);
    for _ in 0..20 {
        let frame: Vec<u8> = base
            .iter()
            .map(|&b| (b as f32 + rng.signed() * 2.5).clamp(0.0, 255.0) as u8)
            .collect();
        f.update(&frame, DT);
        assert!(
            f.motion_energy() < 1.0,
            "sensor noise read as motion: {} (real motion is {})",
            f.motion_energy(),
            moving.motion_energy()
        );
        assert!(f.flow().max_speed() < 8.0, "{}", f.flow().max_speed());
    }
}
