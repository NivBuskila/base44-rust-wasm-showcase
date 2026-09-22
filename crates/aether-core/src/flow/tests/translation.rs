//! What the solver must actually recover: translation in every direction, zoom, a localised patch, and small motion without systematic understatement.

use super::support::*;

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
