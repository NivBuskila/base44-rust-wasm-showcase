//! Drawing a thrown bolt into the field: the streak along its path and the
//! impact where it lands. Detection lives in the parent; this is only effect.

use super::*;
use crate::fluid::Fluid;
use crate::math::hue_to_rgb;
use crate::particles::Particles;

/// Draws one thrown bolt: a streak from the palm and an impact where it lands.
pub fn fire(
    fluid: &mut Fluid,
    particles: &mut Particles,
    particle_life: f32,
    throw: Throw,
    sx: f32,
    sy: f32,
) {
    let power = fin(throw.power).clamp(0.0, 1.0);
    let (nx, ny) = (fin(throw.dir[0]), fin(throw.dir[1]));
    let from = [
        (fin(throw.from[0]) * sx).clamp(0.0, sx),
        (fin(throw.from[1]) * sy).clamp(0.0, sy),
    ];
    if nx == 0.0 && ny == 0.0 {
        // At the camera: nothing to fly across, so it all happens at the
        // palm, bigger than a landing because it is coming at the viewer.
        impact(fluid, particles, particle_life, from, power, 1.6);
        return;
    }
    let reach = REACH * (0.55 + 0.45 * power);
    let to = [
        (from[0] + nx * reach * sx).clamp(0.0, sx),
        (from[1] + ny * reach * sy).clamp(0.0, sy),
    ];
    let rgb = hue_to_rgb(BOLT_HUE);
    let drive = TRAIL_SPEED * (0.5 + 0.5 * power);

    // The trail brightens towards the far end, so the eye follows the bolt away
    // from the hand instead of reading a uniform stripe.
    for i in 0..TRAIL_STEPS {
        let t = (i as f32 + 1.0) / TRAIL_STEPS as f32;
        let x = from[0] + (to[0] - from[0]) * t;
        let y = from[1] + (to[1] - from[1]) * t;
        let w = t * t;
        fluid.add_force(x, y, nx * drive * w, ny * drive * w, TRAIL_RADIUS);
        let a = TRAIL_DYE * w * (0.5 + 0.5 * power);
        fluid.add_dye(x, y, [rgb[0] * a, rgb[1] * a, rgb[2] * a], TRAIL_RADIUS);
    }

    impact(fluid, particles, particle_life, to, power, 1.0);
}

/// The detonation: an outward puff of particles, a dye flash and a ring of
/// force. `size` scales the radius; a bolt landing far away is 1.0.
fn impact(
    fluid: &mut Fluid,
    particles: &mut Particles,
    particle_life: f32,
    at: [f32; 2],
    power: f32,
    size: f32,
) {
    let rgb = hue_to_rgb(BOLT_HUE);
    let radius = IMPACT_RADIUS * size;
    let burst = IMPACT_SPEED * (0.5 + 0.5 * power) * size;
    let count =
        ((particles.active() / IMPACT_FRACTION) as f32 * (0.5 + 0.5 * power) * size) as usize;
    particles.spawn_burst(
        at[0],
        at[1],
        count.min(IMPACT_MAX),
        burst,
        1.0,
        particle_life,
    );
    particles.impulse(at[0], at[1], radius * 1.5, 0.0, 1.0);
    let flash = IMPACT_DYE * (0.5 + 0.5 * power);
    fluid.add_dye(
        at[0],
        at[1],
        [rgb[0] * flash, rgb[1] * flash, rgb[2] * flash],
        radius,
    );
    // An event, not a force: a one-off outward push, deliberately un-scaled by
    // dt so a throw hits as hard at 30 fps as at 144.
    let n = 20;
    for i in 0..n {
        let theta = i as f32 / n as f32 * core::f32::consts::TAU;
        let (cx, cy) = (theta.cos(), theta.sin());
        fluid.add_force(
            at[0] + cx * radius * 0.5,
            at[1] + cy * radius * 0.5,
            cx * burst,
            cy * burst,
            radius * 0.5,
        );
    }
}
