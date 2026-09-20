//! Two-hand duets: the tide current between palm and fist, the clap and the
//! big bang, applied between the hands.

use super::*;

/// Duet payloads, from `crate::duet`.
///
/// Tide is continuous, so like the vortex it is a *relaxation* toward a target
/// speed rather than an accumulating impulse: a current held for ten seconds
/// must not be ten times the current held for one. The tube is a capsule from
/// the palm to the fist, `TIDE_WIDTH` palm radii wide.
const TIDE_SPEED: f32 = 300.0;
const TIDE_RATE: f32 = 6.0;
const TIDE_WIDTH: f32 = 1.1;
const TIDE_DYE: f32 = 1.1;
/// Dye splats laid along the tube per unit of palm radius of length.
const TIDE_SPLAT_SPACING: f32 = 0.8;
const TIDE_MAX_SPLATS: usize = 24;
/// Clap and big bang are events: velocity steps, not scaled by `dt`, like the
/// combos above. The clap is a fast, tight shell from where the hands met; the
/// big bang is the biggest thing in the engine and is meant to read as such.
const CLAP_RING_SCALE: f32 = 1.4;
const CLAP_SPEED: f32 = 380.0;
const CLAP_IMPULSE: f32 = 600.0;
const CLAP_REACH_SCALE: f32 = 4.0;
const CLAP_DYE: f32 = 2.2;
const CLAP_FRACTION: usize = 4;
const BANG_RING_SCALE: f32 = 3.0;
const BANG_SPEED: f32 = 640.0;
const BANG_IMPULSE: f32 = 860.0;
const BANG_REACH_SCALE: f32 = 9.0;
const BANG_DYE: f32 = 3.8;
const BANG_FRACTION: usize = 2;
const DUET_MAX: usize = 32_000;

/// Applies this step's two-hand duets: the tide between palm and fist, and
/// whichever event fired, at the midpoint of the pair.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_duet(
    tracker: &GestureTracker,
    fluid: &mut Fluid,
    particles: &mut Particles,
    params: &Params,
    duet: DuetFrame,
    sx: f32,
    sy: f32,
    dt: f32,
    report: &mut SpellReport,
) {
    let two = tracker.two_hand();
    let hands = tracker.hands();
    if !two.both_present || hands.len() < 2 {
        return;
    }
    let (gw, gh) = (fluid.width(), fluid.height());
    let radius = |scale: f32| (scale * sx * PALM_RADIUS_SCALE).clamp(MIN_RADIUS, MAX_RADIUS);
    let mean_radius = (radius(hands[0].scale) + radius(hands[1].scale)) * 0.5;

    let tide = fin(duet.tide).clamp(0.0, 1.0);
    if tide > 1e-3 {
        // The duet recogniser only reports a tide for fist + palm, but it ramps
        // down after the pair breaks, so the roles are read from whichever hand
        // is (still) the fist; the other end is the source.
        let (from, to) = if hands[1].spell == Spell::Attract {
            (&hands[0], &hands[1])
        } else {
            (&hands[1], &hands[0])
        };
        fire_tide(
            fluid,
            particles,
            to_grid(from.palm, sx, sy),
            to_grid(to.palm, sx, sy),
            mean_radius,
            tide,
            dt,
            report,
        );
    }

    let Some(fire) = duet.fire else {
        return;
    };
    let center = to_grid(two.midpoint, sx, sy);
    let power = 0.5 + 0.5 * fin(fire.power).clamp(0.0, 1.0);
    report.bursts += 1;
    match fire.effect {
        // Tide is continuous and never an event; nothing to do here.
        DuetEffect::Tide => {}
        DuetEffect::Clap => {
            let ring = mean_radius * CLAP_RING_SCALE;
            let count = ((particles.active() / CLAP_FRACTION) as f32 * power) as usize;
            particles.spawn_ring(
                center[0],
                center[1],
                count.min(DUET_MAX),
                ring,
                CLAP_SPEED * power,
                0.8,
                params.particle_life,
            );
            radial_kick(
                fluid,
                gw,
                gh,
                center,
                mean_radius * CLAP_REACH_SCALE,
                CLAP_IMPULSE * power,
                report,
            );
            ring_dye(
                fluid,
                center,
                ring,
                mean_radius * 0.5,
                tint(Spell::Repel, CLAP_DYE * power),
            );
        }
        DuetEffect::BigBang => {
            let ring = mean_radius * BANG_RING_SCALE;
            let count = ((particles.active() / BANG_FRACTION) as f32 * power) as usize;
            particles.spawn_ring(
                center[0],
                center[1],
                count.min(DUET_MAX),
                ring,
                BANG_SPEED * power,
                1.0,
                params.particle_life,
            );
            radial_kick(
                fluid,
                gw,
                gh,
                center,
                mean_radius * BANG_REACH_SCALE * power,
                BANG_IMPULSE * power,
                report,
            );
            // Heat everything the shell will cross, then the shell itself and a
            // white-hot core: the flash is what pays for the squeeze.
            particles.impulse(center[0], center[1], ring * 2.5, 0.0, 1.0);
            ring_dye(
                fluid,
                center,
                ring,
                mean_radius,
                tint(Spell::Ignite, BANG_DYE * power),
            );
            fluid.add_dye(
                center[0],
                center[1],
                tint(Spell::Shatter, BANG_DYE * 0.7 * power),
                mean_radius * 1.5,
            );
        }
    }
}

/// A steady current from `from` to `to`: the fluid inside a capsule between the
/// two palms is relaxed toward a constant speed along the line, and a gradient
/// of dye is laid along it from the source hand's hue to the sink's.
#[allow(clippy::too_many_arguments)]
pub(super) fn fire_tide(
    fluid: &mut Fluid,
    particles: &mut Particles,
    from: [f32; 2],
    to: [f32; 2],
    radius: f32,
    strength: f32,
    dt: f32,
    report: &mut SpellReport,
) {
    let (gw, gh) = (fluid.width(), fluid.height());
    let (ax, ay) = (from[0], from[1]);
    let len = length(to[0] - ax, to[1] - ay);
    if len < 1.0 {
        return;
    }
    let (dx, dy) = normalize(to[0] - ax, to[1] - ay);
    let tube = radius * TIDE_WIDTH;
    let target = TIDE_SPEED * strength;
    let k = 1.0 - decay(TIDE_RATE, dt);
    let mut injected = 0.0;
    let vel = fluid.velocity_mut();
    for_capsule(gw, gh, from, to, tube, |x, y, along, w| {
        // Ease in at the source and out at the sink so the current joins the
        // hands' own radial fields instead of slamming into them.
        let ends = smoothstep(0.0, tube, along) * smoothstep(len, len - tube, along);
        let speed = target * w * ends;
        let (u, v) = vel.get(x, y);
        let du = clamp_impulse((dx * speed - u) * k);
        let dv = clamp_impulse((dy * speed - v) * k);
        vel.add(x, y, du, dv);
        injected += du * du + dv * dv;
    });
    report.injected += injected;

    let n = ((len / (radius * TIDE_SPLAT_SPACING)).ceil() as usize).clamp(2, TIDE_MAX_SPLATS);
    let amount = TIDE_DYE * strength * dt / n as f32 * 4.0;
    let source = tint(Spell::Repel, amount);
    let sink = tint(Spell::Attract, amount);
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        let dye = [
            lerp(source[0], sink[0], t),
            lerp(source[1], sink[1], t),
            lerp(source[2], sink[2], t),
        ];
        fluid.add_dye(lerp(ax, to[0], t), lerp(ay, to[1], t), dye, tube * 0.6);
    }
    // Warmth along the current so the carried particles glow in it.
    particles.impulse(
        lerp(ax, to[0], 0.5),
        lerp(ay, to[1], 0.5),
        len * 0.5,
        0.0,
        0.6 * strength,
    );
}
