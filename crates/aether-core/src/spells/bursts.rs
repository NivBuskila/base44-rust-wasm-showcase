//! One-shot payloads: the charged-fist release ring and the combo rewards
//! (nova, tempest, supernova).

use super::*;

/// Ring radius as a multiple of the palm radius, and the shell speed at full
/// charge (half of it at zero charge).
const RELEASE_RING_SCALE: f32 = 1.3;
const RELEASE_SPEED: f32 = 260.0;
/// Velocity step kicked into the fluid on the ring; a shell, not a disc.
const RELEASE_IMPULSE: f32 = 420.0;
const RELEASE_REACH_SCALE: f32 = 4.0;
const RELEASE_DYE: f32 = 1.6;
const RELEASE_FRACTION: usize = 5;
const RELEASE_MAX: usize = 16_000;
/// Combo payloads. These are events, so like `Shatter` and `Release` their
/// impulses are velocity *steps* and are deliberately not scaled by `dt`: a
/// sequence the caster performed once must hit exactly as hard at 30 fps as at
/// 144, or the reward for landing it depends on the machine.
///
/// Nova: a fast bright shell, tuned to always read even in a busy field.
const NOVA_RING_SCALE: f32 = 1.6;
const NOVA_SPEED: f32 = 330.0;
const NOVA_IMPULSE: f32 = 520.0;
const NOVA_DYE: f32 = 2.4;
const NOVA_FRACTION: usize = 4;
/// Tempest: a wide, long-lived rotation rather than a flash.
const TEMPEST_RADIUS_SCALE: f32 = 7.0;
const TEMPEST_SPEED: f32 = 420.0;
const TEMPEST_DYE: f32 = 1.5;
/// Supernova: the charged combo. Reaches far, kicks hard, and recycles a big
/// slice of the pool into the shell so the screen visibly reorganises.
const SUPERNOVA_RING_SCALE: f32 = 2.4;
const SUPERNOVA_SPEED: f32 = 560.0;
const SUPERNOVA_IMPULSE: f32 = 780.0;
const SUPERNOVA_REACH_SCALE: f32 = 6.5;
const SUPERNOVA_DYE: f32 = 3.4;
const SUPERNOVA_FRACTION: usize = 2;
/// Hard cap on particles any one combo may recycle.
const COMBO_MAX: usize = 24_000;

/// Lets a charged fist go: a shell of particles, a ring-shaped kick into the
/// fluid and a ring of dye, all scaled by how long the fist was held.
pub(super) fn fire_release(
    fluid: &mut Fluid,
    particles: &mut Particles,
    params: &Params,
    palm: [f32; 2],
    radius: f32,
    charge: f32,
    report: &mut SpellReport,
) {
    let (gw, gh) = (fluid.width(), fluid.height());
    let charge = charge.clamp(0.0, 1.0);
    let ring = radius * RELEASE_RING_SCALE;
    let power = 0.5 + 0.5 * charge;
    report.bursts += 1;
    let count = ((particles.active() / RELEASE_FRACTION) as f32 * power) as usize;
    particles.spawn_ring(
        palm[0],
        palm[1],
        count.min(RELEASE_MAX),
        ring,
        RELEASE_SPEED * power,
        1.0,
        params.particle_life,
    );
    // An event, not a force: a velocity step, deliberately not scaled by dt.
    let reach = radius * RELEASE_REACH_SCALE;
    let shell = ring * 0.6;
    let mut injected = 0.0;
    let vel = fluid.velocity_mut();
    for_disc(gw, gh, palm, reach, |x, y, dx, dy, r, _| {
        // Peaked on the ring, zero at the centre and the reach.
        let band = ((r - ring) / shell).powi(2);
        let w = (-band).exp() * smoothstep(0.0, ring * 0.5, r);
        let (nx, ny) = radial(dx, dy, r);
        let du = clamp_impulse(nx * RELEASE_IMPULSE * power * w);
        let dv = clamp_impulse(ny * RELEASE_IMPULSE * power * w);
        vel.add(x, y, du, dv);
        injected += du * du + dv * dv;
    });
    report.injected += injected;
    let dye = tint(Spell::Release, RELEASE_DYE * power);
    let n = 24;
    for i in 0..n {
        let theta = i as f32 / n as f32 * core::f32::consts::TAU;
        fluid.add_dye(
            palm[0] + theta.cos() * ring,
            palm[1] + theta.sin() * ring,
            dye,
            radius * 0.45,
        );
    }
}

/// Applies a completed sequence's payload at `palm`.
///
/// Split by effect rather than parameterised into one blend, because these are
/// meant to be *recognisably different* rewards: a caster who lands the hard
/// sequence has to see something the easy one never produces.
pub(super) fn fire_combo(
    effect: ComboEffect,
    fluid: &mut Fluid,
    particles: &mut Particles,
    params: &Params,
    palm: [f32; 2],
    radius: f32,
    report: &mut SpellReport,
) {
    let (gw, gh) = (fluid.width(), fluid.height());
    report.bursts += 1;
    match effect {
        ComboEffect::Nova => {
            let ring = radius * NOVA_RING_SCALE;
            let count = (particles.active() / NOVA_FRACTION).min(COMBO_MAX);
            particles.spawn_ring(
                palm[0],
                palm[1],
                count,
                ring,
                NOVA_SPEED,
                1.0,
                params.particle_life,
            );
            radial_kick(fluid, gw, gh, palm, ring * 2.5, NOVA_IMPULSE, report);
            ring_dye(
                fluid,
                palm,
                ring,
                radius * 0.5,
                tint(Spell::Shatter, NOVA_DYE),
            );
        }
        ComboEffect::Tempest => {
            // Pure rotation, no radial term: the storm has to keep spinning
            // after the impulse lands, and an outward push would blow the
            // structure apart before the swirl becomes visible.
            let outer = radius * TEMPEST_RADIUS_SCALE;
            let mut injected = 0.0;
            let vel = fluid.velocity_mut();
            for_disc(gw, gh, palm, outer, |x, y, dx, dy, r, w| {
                let (nx, ny) = radial(dx, dy, r);
                let du = clamp_impulse(-ny * TEMPEST_SPEED * w);
                let dv = clamp_impulse(nx * TEMPEST_SPEED * w);
                vel.add(x, y, du, dv);
                injected += du * du + dv * dv;
            });
            report.injected += injected;
            ring_dye(
                fluid,
                palm,
                outer * 0.55,
                outer * 0.3,
                tint(Spell::Vortex, TEMPEST_DYE),
            );
        }
        ComboEffect::Supernova => {
            let ring = radius * SUPERNOVA_RING_SCALE;
            let count = (particles.active() / SUPERNOVA_FRACTION).min(COMBO_MAX);
            particles.spawn_ring(
                palm[0],
                palm[1],
                count,
                ring,
                SUPERNOVA_SPEED,
                1.0,
                params.particle_life,
            );
            radial_kick(
                fluid,
                gw,
                gh,
                palm,
                radius * SUPERNOVA_REACH_SCALE,
                SUPERNOVA_IMPULSE,
                report,
            );
            // Heat first, then dye: the flash is what sells the charge time.
            particles.impulse(palm[0], palm[1], ring * 2.0, 0.0, 1.0);
            ring_dye(
                fluid,
                palm,
                ring,
                radius * 0.8,
                tint(Spell::Ignite, SUPERNOVA_DYE),
            );
            fluid.add_dye(
                palm[0],
                palm[1],
                tint(Spell::Shatter, SUPERNOVA_DYE * 0.6),
                radius,
            );
        }
    }
}
