//! Solver tests: ground-truth recovery, robustness, and the config surface.

use super::*;
use crate::rng::Rng;
use core::f32::consts::TAU;

/// Deterministic band-limited test texture.
///
/// Sampled analytically so a translated or zoomed copy is exact instead of
/// resampled, and built from one wave per octave so that *every* pyramid
/// level still has real gradients: a single high-frequency octave is gone
/// by the second decimation and the coarse solve would have nothing to
/// lock onto, which makes a "flow is nonzero" assertion vacuous.
struct Texture {
    waves: [(f32, f32, f32, f32); 3],
    norm: f32,
}

impl Texture {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut waves = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 3];
        // Wavelengths in full-resolution cells; the longest survives three
        // decimations, the shortest gives the finest level something to do.
        let spec = [(30.0f32, 0.45f32), (14.0, 0.30), (8.0, 0.25)];
        // The seed picks the overall orientation, but the three waves are
        // then spread evenly: a texture whose gradients all point the same
        // way leaves the flow component across them unobservable (the
        // aperture problem), and the ground truth would be unrecoverable
        // for reasons that have nothing to do with the solver.
        let base = rng.next_f32() * TAU;
        for (i, (slot, &(wavelength, amp))) in waves.iter_mut().zip(spec.iter()).enumerate() {
            let theta = base + i as f32 * TAU / 3.0;
            let k = TAU / wavelength;
            *slot = (k * theta.cos(), k * theta.sin(), rng.next_f32() * TAU, amp);
        }
        let norm = waves.iter().map(|w| w.3.abs()).sum::<f32>().max(1e-6);
        Self { waves, norm }
    }

    fn at(&self, x: f32, y: f32) -> f32 {
        let s: f32 = self
            .waves
            .iter()
            .map(|&(kx, ky, phase, amp)| amp * (kx * x + ky * y + phase).sin())
            .sum();
        0.5 + 0.5 * s / self.norm
    }

    fn render(&self, w: usize, h: usize, map: impl Fn(f32, f32) -> (f32, f32)) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = map(x as f32, y as f32);
                out[y * w + x] = (self.at(sx, sy).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        out
    }

    fn still(&self, w: usize, h: usize) -> Vec<u8> {
        self.render(w, h, |x, y| (x, y))
    }

    /// Content shifted by `(dx, dy)` cells, so the true flow is `(dx, dy)`.
    fn translated(&self, w: usize, h: usize, dx: f32, dy: f32) -> Vec<u8> {
        self.render(w, h, |x, y| (x - dx, y - dy))
    }

    /// Content magnified by `s` about the frame centre: outward flow.
    fn zoomed(&self, w: usize, h: usize, s: f32) -> Vec<u8> {
        let cx = (w - 1) as f32 * 0.5;
        let cy = (h - 1) as f32 * 0.5;
        self.render(w, h, |x, y| (cx + (x - cx) / s, cy + (y - cy) / s))
    }
}

const W: usize = 128;
const H: usize = 72;
const DT: f32 = 1.0 / 30.0;

fn flow_of(prev: &[u8], cur: &[u8], dt: f32) -> OpticalFlow {
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    f.update(prev, dt);
    f.update(cur, dt);
    f
}

/// Mean flow over the interior, away from the border where the clamped
/// warp invents data that was never in frame.
fn mean_interior(f: &VecField, margin: usize) -> (f32, f32) {
    let mut su = 0.0f64;
    let mut sv = 0.0f64;
    let mut n = 0u32;
    for y in margin..f.h() - margin {
        for x in margin..f.w() - margin {
            let (u, v) = f.get(x, y);
            su += u as f64;
            sv += v as f64;
            n += 1;
        }
    }
    let inv = 1.0 / n.max(1) as f64;
    ((su * inv) as f32, (sv * inv) as f32)
}

fn assert_recovers(prev: &[u8], cur: &[u8], want: (f32, f32)) {
    let f = flow_of(prev, cur, DT);
    let (u, v) = mean_interior(f.flow(), 16);
    let (wu, wv) = (want.0 / DT, want.1 / DT);
    let scale = (wu * wu + wv * wv).sqrt();
    assert!(
        (u - wu).abs() < 0.35 * scale,
        "u {u} should be near {wu} (v {v})"
    );
    assert!(
        (v - wv).abs() < 0.35 * scale,
        "v {v} should be near {wv} (u {u})"
    );
    // Sign is the part a magnitude-only test would never catch.
    if wu.abs() > 1.0 {
        assert!(u.signum() == wu.signum(), "u sign flipped: {u} vs {wu}");
    }
    if wv.abs() > 1.0 {
        assert!(v.signum() == wv.signum(), "v sign flipped: {v} vs {wv}");
    }
}

#[test]
fn recovers_translation_plus_x() {
    let t = Texture::new(1);
    assert_recovers(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), (2.0, 0.0));
}

#[test]
fn recovers_translation_minus_x() {
    let t = Texture::new(2);
    assert_recovers(&t.still(W, H), &t.translated(W, H, -2.0, 0.0), (-2.0, 0.0));
}

#[test]
fn recovers_translation_plus_y() {
    let t = Texture::new(3);
    assert_recovers(&t.still(W, H), &t.translated(W, H, 0.0, 2.0), (0.0, 2.0));
}

#[test]
fn recovers_translation_minus_y() {
    let t = Texture::new(4);
    assert_recovers(&t.still(W, H), &t.translated(W, H, 0.0, -2.0), (0.0, -2.0));
}

#[test]
fn recovers_diagonal_translation() {
    let t = Texture::new(5);
    assert_recovers(&t.still(W, H), &t.translated(W, H, 3.0, -2.0), (3.0, -2.0));
}

#[test]
fn dominant_motion_and_energy_track_a_translation() {
    let t = Texture::new(6);
    let f = flow_of(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), DT);
    let (dx, dy) = f.dominant_motion();
    assert!(dx > 0.5 * 2.0 / DT, "dominant dx too small: {dx}");
    assert!(dy.abs() < 0.35 * 2.0 / DT, "spurious dominant dy: {dy}");
    // `motion_energy >= dx^2` would be vacuous: mean(u^2 + v^2) is never
    // below mean(u)^2 for *any* field, so that inequality holds even if the
    // solver returns garbage. Pin it to the analytic truth instead. It lands
    // slightly under (2 / DT)^2 because the border fade zeroes the frame
    // edge, and the field is near-uniform so it cannot land far over.
    let truth = (2.0 / DT) * (2.0 / DT);
    assert!(
        f.motion_energy() > 0.6 * truth && f.motion_energy() < 1.3 * truth,
        "energy {} should be near {truth}",
        f.motion_energy()
    );
}

#[test]
fn zoom_produces_outward_flow() {
    let t = Texture::new(7);
    let f = flow_of(&t.still(W, H), &t.zoomed(W, H, 1.12), DT);
    let flow = f.flow();
    let (cx, cy) = ((W - 1) as f32 * 0.5, (H - 1) as f32 * 0.5);
    let mut radial = 0.0f64;
    let mut divergence = 0.0f64;
    let mut n = 0u32;
    for y in 12..H - 12 {
        for x in 12..W - 12 {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt();
            if r < 8.0 {
                continue; // Sub-pixel motion near the centre of dilation.
            }
            let (u, v) = flow.get(x, y);
            radial += ((u * dx + v * dy) / r) as f64;
            let (dudx, _) = flow.u.gradient(x, y);
            let (_, dvdy) = flow.v.gradient(x, y);
            divergence += (dudx + dvdy) as f64;
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    assert!(radial / n > 1.0, "flow is not outward: {}", radial / n);
    assert!(divergence / n > 0.0, "divergence {}", divergence / n);
}

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
fn flow_is_per_second_not_per_frame() {
    let t = Texture::new(11);
    let (a, b) = (t.still(W, H), t.translated(W, H, 2.0, 0.0));
    let slow = flow_of(&a, &b, 1.0 / 30.0);
    let fast = flow_of(&a, &b, 1.0 / 60.0);
    let (su, _) = mean_interior(slow.flow(), 16);
    let (fu, _) = mean_interior(fast.flow(), 16);
    assert!(su > 1.0, "baseline flow too small to compare: {su}");
    let ratio = fu / su;
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "halving dt should double the flow, got {ratio}x"
    );
}

#[test]
fn the_reported_speed_does_not_depend_on_the_frame_rate() {
    // The same physical 60 cells/second, sampled at four frame rates. Any
    // bias that is a *per-frame* constant — a subtracted noise floor, say —
    // becomes `constant / dt` in the published cells/second and shows up
    // here as a spread that tracks the frame rate.
    const SPEED: f32 = 60.0;
    let t = Texture::new(40);
    let still = t.still(W, H);
    for fps in [15.0f32, 30.0, 60.0, 144.0] {
        let dt = 1.0 / fps;
        let f = flow_of(&still, &t.translated(W, H, SPEED * dt, 0.0), dt);
        let (u, v) = mean_interior(f.flow(), 16);
        assert!(
            (u - SPEED).abs() < 0.06 * SPEED,
            "{fps} fps reported {u} cells/s for a true {SPEED}"
        );
        assert!(v.abs() < 0.1 * SPEED, "{fps} fps invented v {v}");
    }
}

#[test]
fn small_motion_is_not_systematically_understated() {
    // Half a cell to two cells per frame is the common case for a hand held
    // in front of the camera, and it is where a magnitude bias hides: the
    // 35% band `assert_recovers` allows would not notice a 10% cut.
    let t = Texture::new(41);
    let still = t.still(W, H);
    for d in [0.5f32, 1.0, 2.0] {
        let f = flow_of(&still, &t.translated(W, H, d, 0.0), DT);
        let (u, _) = mean_interior(f.flow(), 16);
        let truth = d / DT;
        assert!(
            (u - truth).abs() < 0.08 * truth,
            "{d} cells/frame reported {u} cells/s, truth {truth}"
        );
    }
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
fn extreme_dt_stays_finite_and_clamped() {
    let t = Texture::new(14);
    let (a, b) = (t.still(W, H), t.translated(W, H, 3.0, 1.0));
    for dt in [f32::NAN, f32::INFINITY, 0.0, -1.0, 1e-9, 1e9] {
        let f = flow_of(&a, &b, dt);
        assert!(
            f.flow().max_speed() <= MAX_FLOW + 1e-3,
            "dt {dt} produced {}",
            f.flow().max_speed()
        );
        assert!(f.flow().u.data.iter().all(|v| v.is_finite()), "dt {dt}");
        assert!(f.motion_energy().is_finite(), "dt {dt}");
        // A negative or zero dt must never invert the direction of time.
        assert!(f.dominant_motion().0 >= 0.0, "dt {dt} flipped the sign");
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
fn single_level_still_recovers_small_motion() {
    // Without a pyramid, Horn-Schunck only sees motion inside its own
    // aperture, so this checks the flat (non-multi-scale) path works too.
    let t = Texture::new(16);
    let cfg = FlowConfig {
        levels: 1,
        iters: 24,
        ..Default::default()
    };
    let mut f = OpticalFlow::new(W, H, cfg);
    f.update(&t.still(W, H), DT);
    f.update(&t.translated(W, H, 1.0, 0.0), DT);
    let (u, v) = mean_interior(f.flow(), 16);
    assert_eq!(f.levels(), 1);
    assert!(u > 0.4 / DT, "single level lost the motion: {u}");
    assert!(v.abs() < 0.35 / DT, "spurious v: {v}");
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

#[test]
fn a_moving_patch_localises_the_flow() {
    // Global translation is the easy case; a real hand moves a small part
    // of the frame, and the flow must stay near it instead of smearing
    // over the whole field.
    let t = Texture::new(18);
    let still = t.still(W, H);
    let mut moved = still.clone();
    let shifted = t.translated(W, H, 3.0, 0.0);
    for y in 20..52 {
        for x in 80..120 {
            moved[y * W + x] = shifted[y * W + x];
        }
    }
    let f = flow_of(&still, &moved, DT);
    let flow = f.flow();
    let mut inside = 0.0f64;
    let mut outside = 0.0f64;
    for y in 0..H {
        for x in 0..W {
            let (u, v) = flow.get(x, y);
            let mag = (u * u + v * v).sqrt() as f64;
            if (24..48).contains(&y) && (88..112).contains(&x) {
                inside += mag;
            } else if x < 48 {
                outside += mag;
            }
        }
    }
    inside /= (24 * 24) as f64;
    outside /= (48 * H) as f64;
    assert!(
        inside > 4.0 * outside.max(1e-6),
        "flow smeared: {inside} inside vs {outside} outside"
    );
}
