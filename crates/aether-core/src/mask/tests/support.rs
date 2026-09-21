//! Fixtures: the sharp/soft rectangle builders, the body rects and the all-finite check.

pub(super) use crate::mask::*;

pub(super) const H: usize = 144;
pub(super) const CELLS: usize = W * H;
pub(super) const DT30: f32 = 1.0 / 30.0;

//! Mask pipeline tests.


/// Grid the tests run `BodyMask` on, deliberately **not**
/// `config::W/H`.
///
/// `BodyMask::new` takes its dimensions explicitly, so a unit test for it
/// has no business depending on the app's current display resolution — and
/// these tests pick their cells for a reason (clear of the walls, straddling
/// a specific edge). Tying them to a tunable meant that changing the fluid
/// grid for the frame budget turned every carefully chosen coordinate into
/// either an out-of-bounds index or a different part of the frame.
pub(super) const W: usize = 256;

/// Config with the temporal EMA disabled, so one update lands the raw mask
/// exactly and a test's arithmetic stays analytic.
pub(super) fn sharp() -> MaskConfig {
    MaskConfig {
        half_life: 0.0,
        ..MaskConfig::default()
    }
}

pub(super) fn body(cfg: MaskConfig) -> BodyMask {
    BodyMask::new(W, H, cfg)
}

/// A mask at grid resolution with a filled rectangle. Same resolution means
/// the resample is a copy, so cell indices are directly comparable.
///
/// Coordinates are clamped: these tests are written against grid fractions
/// (see [`fx`] / [`fy`]) rather than absolute cells, and a clamp here means
/// a future resolution change degrades a rectangle instead of panicking
/// with an out-of-bounds index.
pub(super) fn rect_mask(x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<f32> {
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
pub(super) fn fy(frac: f32) -> usize {
    ((H as f32 * frac) as usize).min(H - 1)
}

/// The standard test silhouette: a solid block left of centre, well clear
/// of every domain wall so border effects cannot be mistaken for the
/// behaviour under test.
pub(super) fn body_rect() -> Vec<f32> {
    rect_mask(100, 140, 40, 100)
}

/// [`body_rect`] translated `cells` to the right, for boundary-motion tests.
pub(super) fn body_rect_shifted(cells: usize) -> Vec<f32> {
    rect_mask(100 + cells, 140 + cells, 40, 100)
}

/// A point inside [`body_rect`], safely away from its edges.
pub(super) fn body_interior() -> (usize, usize) {
    (120, 70)
}

pub(super) fn all_finite(m: &BodyMask) -> bool {
    m.obstacle()
        .data
        .iter()
        .chain(m.confidence().data.iter())
        .chain(m.edge_velocity().u.data.iter())
        .chain(m.edge_velocity().v.data.iter())
        .all(|v| v.is_finite())
}
