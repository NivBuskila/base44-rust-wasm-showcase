//! Grid-walking primitives and scalar helpers shared by every spell: disc and
//! capsule iteration, radial kicks, dye rings, clamps and coordinate mapping.

use super::*;

/// Walks the cells within `radius` of the segment `a`–`b`, handing each its
/// distance along the segment and a smooth falloff weight across it.
pub(super) fn for_capsule(
    w: usize,
    h: usize,
    a: [f32; 2],
    b: [f32; 2],
    radius: f32,
    mut f: impl FnMut(usize, usize, f32, f32),
) {
    if !a.iter().chain(b.iter()).all(|v| v.is_finite())
        || !radius.is_finite()
        || radius <= 0.0
        || w < 1
        || h < 1
    {
        return;
    }
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-6 {
        return;
    }
    let x0 = (a[0].min(b[0]) - radius).max(0.0) as usize;
    let y0 = (a[1].min(b[1]) - radius).max(0.0) as usize;
    let x1 = ((a[0].max(b[0]) + radius).max(0.0) as usize).min(w - 1);
    let y1 = ((a[1].max(b[1]) + radius).max(0.0) as usize).min(h - 1);
    if x0 > x1 || y0 > y1 {
        return;
    }
    let len = len2.sqrt();
    let r2 = radius * radius;
    for y in y0..=y1 {
        let py = y as f32 - a[1];
        for x in x0..=x1 {
            let px = x as f32 - a[0];
            let t = ((px * dx + py * dy) / len2).clamp(0.0, 1.0);
            let (cx, cy) = (px - dx * t, py - dy * t);
            let d2 = cx * cx + cy * cy;
            if d2 > r2 {
                continue;
            }
            f(x, y, t * len, smoothstep(radius, 0.0, d2.sqrt()));
        }
    }
}

/// An outward velocity step over a disc, falling off to zero at `reach`.
pub(super) fn radial_kick(
    fluid: &mut Fluid,
    gw: usize,
    gh: usize,
    center: [f32; 2],
    reach: f32,
    impulse: f32,
    report: &mut SpellReport,
) {
    let mut injected = 0.0;
    let vel = fluid.velocity_mut();
    for_disc(gw, gh, center, reach, |x, y, dx, dy, r, w| {
        let (nx, ny) = radial(dx, dy, r);
        let du = clamp_impulse(nx * impulse * w);
        let dv = clamp_impulse(ny * impulse * w);
        vel.add(x, y, du, dv);
        injected += du * du + dv * dv;
    });
    report.injected += injected;
}

/// Lays dye around a circle instead of on a disc, so the effect reads as a
/// shell expanding outward rather than a blob fading out.
pub(super) fn ring_dye(fluid: &mut Fluid, center: [f32; 2], ring: f32, splat: f32, dye: [f32; 3]) {
    let n = 28;
    for i in 0..n {
        let theta = i as f32 / n as f32 * core::f32::consts::TAU;
        fluid.add_dye(
            center[0] + theta.cos() * ring,
            center[1] + theta.sin() * ring,
            dye,
            splat,
        );
    }
}

// ---------------------------------------------------------------- internals

/// Walks the cells within `radius` of a grid-space point, handing each one its
/// offset `(dx, dy)`, its distance `r`, and a smooth falloff weight (1 at the
/// centre, 0 with zero slope at the rim).
///
/// A hard-edged disc leaves a ring of cells moving at full speed next to
/// stationary ones, and that ring is visible as a circle every single time.
///
/// `r` is passed even though the callback could compute it: every caller needs
/// it, either for a radial direction or for a profile, and the square root is
/// the most expensive thing in the loop.
pub(super) fn for_disc(
    w: usize,
    h: usize,
    center: [f32; 2],
    radius: f32,
    mut f: impl FnMut(usize, usize, f32, f32, f32, f32),
) {
    let (cx, cy) = (center[0], center[1]);
    if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 || w < 1 || h < 1
    {
        return;
    }
    let x0 = (cx - radius).max(0.0) as usize;
    let y0 = (cy - radius).max(0.0) as usize;
    let x1 = ((cx + radius).max(0.0) as usize).min(w - 1);
    let y1 = ((cy + radius).max(0.0) as usize).min(h - 1);
    if x0 > x1 || y0 > y1 {
        return;
    }
    let r2 = radius * radius;
    for y in y0..=y1 {
        let dy = y as f32 - cy;
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let d2 = dx * dx + dy * dy;
            if d2 > r2 {
                continue;
            }
            let r = d2.sqrt();
            f(x, y, dx, dy, r, smoothstep(radius, 0.0, r));
        }
    }
}

/// Normalised `[0, 1]` to grid space. Clamped, because a landmark just off the
/// edge of the frame is normal and must not splat onto the opposite border.
#[inline]
pub(super) fn to_grid(p: [f32; 2], sx: f32, sy: f32) -> [f32; 2] {
    [
        (fin(p[0]) * sx).clamp(0.0, sx),
        (fin(p[1]) * sy).clamp(0.0, sy),
    ]
}

/// Unit vector from the disc centre, given the distance the walker already
/// measured. `(0, 0)` at the exact centre, where there is no direction.
#[inline]
pub(super) fn radial(dx: f32, dy: f32, r: f32) -> (f32, f32) {
    if r > 1e-6 {
        (dx / r, dy / r)
    } else {
        (0.0, 0.0)
    }
}

#[inline]
pub(super) fn clamp_impulse(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-MAX_IMPULSE, MAX_IMPULSE)
    } else {
        0.0
    }
}

#[inline]
pub(super) fn fin(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
pub(super) fn clamp_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(0.0, MAX_DT)
}

/// A spell's hue at a given intensity, ready for [`Fluid::add_dye`].
#[inline]
pub(super) fn tint(spell: Spell, amount: f32) -> [f32; 3] {
    let a = fin(amount).max(0.0);
    let rgb = hue_to_rgb(spell.hue());
    [rgb[0] * a, rgb[1] * a, rgb[2] * a]
}

/// Grid-space to source-field scale, for sampling a field of a different
/// resolution than the fluid.
#[inline]
pub(super) fn resample_scale(field: &VecField, gw: usize, gh: usize) -> (f32, f32) {
    if field.w() < 2 || field.h() < 2 || gw < 2 || gh < 2 {
        return (0.0, 0.0);
    }
    (
        (field.w() - 1) as f32 / (gw - 1) as f32,
        (field.h() - 1) as f32 / (gh - 1) as f32,
    )
}

#[inline]
pub(super) fn is_zero(field: &VecField) -> bool {
    field.u.max_abs() == 0.0 && field.v.max_abs() == 0.0
}
