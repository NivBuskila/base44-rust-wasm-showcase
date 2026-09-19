//! Maps gesture state onto forces, dye and particle bursts.
//!
//! This is the layer that makes the thing feel like magic rather than like a
//! physics demo, and it is deliberately the only place where "what a gesture
//! means" is decided — [`crate::gesture`] classifies, this module acts.
//!
//! ## Contract
//!
//! [`apply`] is called once per simulation step, before [`crate::fluid::Fluid::step`].
//! It must:
//!
//! - **Always** push the fluid with hand motion when a hand is present, so
//!   even `Spell::Idle` feels responsive.
//! - Drive the fluid from raw optical flow scaled by `params.flow_force`. This
//!   path is model-free, so it keeps working when perception is unavailable.
//! - Apply the per-spell effect for each hand:
//!   `Vortex` tangential swirl at the pinch point; `Repel` outward radial push;
//!   `Attract` inward pull; `Ignite` a hot dye trail at the index tip;
//!   `Freeze` local velocity damping plus a cold palette; `Shatter` a one-shot
//!   particle burst that must not re-fire until the spell is released.
//! - Inject the body silhouette's boundary velocity, scaled by
//!   `params.body_push`.
//! - Report the time scale implied by the two-hand rotation gesture, which the
//!   engine folds into the next frame's `dt`.
//!
//! All forces scale with `dt` so behaviour is frame-rate independent, and no
//! force may exceed `MAX_IMPULSE` grid units per second — an unclamped
//! impulse is how a fluid solver turns into NaN soup.

use crate::combo::{ComboEffect, ComboHit};
use crate::config::Params;
use crate::field::VecField;
use crate::fluid::Fluid;
use crate::gesture::{GestureTracker, Spell};
use crate::math::{decay, hue_to_rgb, length, lerp, normalize, smoothstep};
use crate::particles::Particles;

/// Hard ceiling on any single velocity impulse, in grid cells per second.
pub const MAX_IMPULSE: f32 = 900.0;

/// Bounds on the time warp the two-hand rotation may request.
///
/// Below the lower bound the app looks frozen and users assume it crashed;
/// above the upper bound the fluid solver's CFL condition starts to bite and
/// advection smears instead of flowing.
pub const MIN_TIME_SCALE: f32 = 0.25;
/// Upper bound on the requested time warp.
pub const MAX_TIME_SCALE: f32 = 2.5;

/// Longest timestep one call integrates. `apply` is public and the engine's own
/// clamp is not part of this module's contract.
const MAX_DT: f32 = 0.25;

/// Rate, per second, at which the fluid under a palm is pulled toward that
/// hand's velocity.
///
/// A *relaxation* rather than an additive impulse: an impulse proportional to
/// hand speed accumulates for as long as the hand hovers, and since the
/// velocity dissipation is a fraction of a percent per frame the field winds up
/// far past anything the hand is doing, which reads as the fluid exploding for
/// no reason. Relaxing toward the hand's velocity is bounded by construction
/// and is also the better physical model: a hand is a moving solid dragging the
/// fluid with it.
const HAND_DRAG_RATE: f32 = 9.0;

/// Palm blob radius as a multiple of the hand's on-screen size, and the bounds
/// it is clamped into. The bounds matter: a hand pressed against the lens
/// otherwise drives a blob the size of the screen.
const PALM_RADIUS_SCALE: f32 = 0.9;
const MIN_RADIUS: f32 = 4.0;
const MAX_RADIUS: f32 = 56.0;

/// Peak tangential speed a fully wound vortex holds, in grid cells per second.
const VORTEX_SPEED: f32 = 240.0;
/// How fast the swirl converges on that target, per second.
const VORTEX_RATE: f32 = 7.0;
/// Seconds of holding the pinch to reach full strength. Ramping rather than
/// snapping is most of what makes the vortex feel like it is being *charged*.
const VORTEX_WINDUP: f32 = 1.1;
const VORTEX_RADIUS_SCALE: f32 = 2.2;
/// Fraction of the vortex radius over which the solid-body core ramps in.
const VORTEX_CORE: f32 = 0.35;
const VORTEX_DYE: f32 = 0.9;

/// Radial acceleration of the repel push, in grid cells per second squared.
const REPEL_ACCEL: f32 = 1500.0;
/// Attract is deliberately weaker than repel. An inward well concentrates
/// everything it touches, so at equal strength it piles the whole field into a
/// few cells and the projection has to resolve a pressure spike every frame.
const ATTRACT_RATIO: f32 = 0.6;
/// How fast a closed fist bleeds the speed off the particles it gathered, per
/// second at the palm. High enough that they settle into a held clump within a
/// fraction of a second, low enough that they still visibly stream inward.
const GRIP_RATE: f32 = 25.0;
/// Radius of the positional hold, as a multiple of the palm radius. Kept close
/// to the palm so only what the fist actually swallowed travels with it.
const GRIP_RADIUS_SCALE: f32 = 2.2;
/// Rate, per second, at which a gripped particle's position closes on the palm.
/// Fast enough that the clump tracks a moving fist, not instant, so the pull
/// still reads as suction rather than teleporting.
const GRIP_PULL: f32 = 26.0;
const RADIAL_RADIUS_SCALE: f32 = 2.6;
/// Radial acceleration applied to particles, in grid cells per second squared.
const PARTICLE_ACCEL: f32 = 2200.0;
const RADIAL_DYE: f32 = 0.7;

/// Dye deposited per second along an ignite stroke.
const IGNITE_DYE: f32 = 3.2;
const IGNITE_RADIUS: f32 = 3.5;
/// Spacing of trail splats along the stroke, in grid cells. Below the splat
/// radius, so consecutive splats overlap into a continuous line.
const IGNITE_STEP: f32 = 1.5;
const IGNITE_MAX_STEPS: usize = 64;
/// Forward acceleration the stroke drags the fluid with.
const IGNITE_ACCEL: f32 = 700.0;

/// Local velocity damping rate of the freeze spell, per second.
const FREEZE_RATE: f32 = 6.0;
const FREEZE_DYE: f32 = 1.4;
const FREEZE_RADIUS_SCALE: f32 = 2.0;

/// Initial radial speed of burst particles, in grid cells per second.
const SHATTER_SPEED: f32 = 150.0;
/// Velocity step the burst kicks into the fluid. Not a rate: see the arm.
const SHATTER_IMPULSE: f32 = 260.0;
const SHATTER_DYE: f32 = 2.2;
/// Fraction of the pool recycled into one burst, and its hard cap.
const SHATTER_FRACTION: usize = 6;
const SHATTER_MAX: usize = 12_000;

/// Release: seconds of holding the fist that count as a full charge, the least
/// charge worth firing, and the burst it drives.
const RELEASE_CHARGE_TIME: f32 = 1.2;
const RELEASE_MIN_CHARGE: f32 = 0.12;
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
/// Seconds the HUD keeps showing "release" after the ring fires.
const RELEASE_SHOW: f32 = 0.45;

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

/// How much of a hand's own spell strength is withheld while it is mid-sequence.
///
/// Every combo is built out of gestures that are also spells on their own, so a
/// caster winding up a sequence was firing full-strength Attract/Repel/Freeze
/// effects into the same field the combo is about to reorganise, and the payoff
/// read as more of the same. Fading the intermediate steps out turns them into a
/// wind-up: the field quiets down as the sequence advances, then the combo lands.
const COMBO_CHARGE_DAMP: f32 = 0.8;

/// Seconds of rotation rate mapped into one unit of time scale.
const TIME_GAIN: f32 = 0.32;
/// Smoothing rate of the requested time scale, per second. Without it the time
/// warp inherits the tracker's residual jitter and the whole simulation
/// shimmers.
const TIME_SMOOTH_RATE: f32 = 4.0;

/// What [`apply`] did this step, surfaced to the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpellReport {
    /// Latched spell per hand slot.
    pub spells: [Spell; 2],
    /// Total kinetic energy injected this step.
    pub injected: f32,
    /// Particle bursts fired this step.
    pub bursts: u32,
    /// Time scale requested by the two-hand gesture, 1.0 = normal.
    pub time_scale: f32,
}

/// Cross-frame state the spell layer needs (one-shot latches, trail history).
#[derive(Clone, Debug)]
pub struct SpellState {
    /// Whether each hand's one-shot spell has already fired this hold.
    fired: [bool; 2],
    /// Previous index-tip position per hand, in **grid space**, for continuous
    /// trails.
    last_tip: [[f32; 2]; 2],
    has_tip: [bool; 2],
    /// Smoothed time scale, carried across frames so the two-hand warp eases
    /// in and out instead of stepping.
    time_scale: f32,
    /// Latched spell seen last step per hand, for edge-triggered spells.
    prev_spell: [Spell; 2],
    /// How charged the release is, `[0, 1]`, built up while the fist is held.
    charge: [f32; 2],
    /// Seconds left of reporting `Release` after a ring fired.
    release_show: [f32; 2],
}

impl Default for SpellState {
    fn default() -> Self {
        Self {
            fired: [false; 2],
            last_tip: [[0.0; 2]; 2],
            has_tip: [false; 2],
            time_scale: 1.0,
            prev_spell: [Spell::Idle; 2],
            charge: [0.0; 2],
            release_show: [0.0; 2],
        }
    }
}

impl SpellState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Applies every gesture-driven effect for one simulation step.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    tracker: &GestureTracker,
    flow: &VecField,
    body_edge: &VecField,
    fluid: &mut Fluid,
    particles: &mut Particles,
    state: &mut SpellState,
    params: &Params,
    // A gesture sequence that completed this step, recognised by
    // `crate::combo`. Its effect is applied at the casting hand's palm.
    combo: [Option<ComboHit>; 2],
    // Per-hand sequence depth, 0..1, from `ComboTracker::charging`. Fades a
    // hand's own spell out while it is winding a combo up.
    charging: [f32; 2],
    dt: f32,
) -> SpellReport {
    let dt = clamp_dt(dt);
    let mut report = SpellReport {
        time_scale: 1.0,
        ..Default::default()
    };
    let (gw, gh) = (fluid.width(), fluid.height());
    if gw < 2 || gh < 2 {
        return report;
    }
    // Grid space spans [0, w-1] x [0, h-1], so these are both the
    // normalised-to-grid position scale and the velocity scale.
    let (sx, sy) = ((gw - 1) as f32, (gh - 1) as f32);

    drive_fields(flow, body_edge, fluid, params, dt, &mut report);

    let base_force = params.hand_force.max(0.0);
    let slots = report.spells.len();
    for (slot, hand) in tracker.hands().iter().enumerate().take(slots) {
        if !hand.present || !hand.palm[0].is_finite() || !hand.palm[1].is_finite() {
            report.spells[slot] = Spell::Idle;
            state.fired[slot] = false;
            state.has_tip[slot] = false;
            state.prev_spell[slot] = Spell::Idle;
            state.charge[slot] = 0.0;
            state.release_show[slot] = 0.0;
            continue;
        }
        // How far this hand is into a sequence, and the matching fade applied to
        // its own spell effects while it winds that sequence up.
        let winding = fin(charging[slot]).clamp(0.0, 1.0);
        let wind_gain = 1.0 - COMBO_CHARGE_DAMP * winding;
        let force = base_force * wind_gain;
        report.spells[slot] = hand.spell;
        // Releasing the gesture is what re-arms the one-shot.
        if hand.spell != Spell::Shatter {
            state.fired[slot] = false;
        }
        let opened_fist = state.prev_spell[slot] == Spell::Attract && hand.spell != Spell::Attract;
        state.prev_spell[slot] = hand.spell;

        let palm = to_grid(hand.palm, sx, sy);
        let tip = to_grid(hand.index_tip, sx, sy);
        let pinch_point = to_grid(
            [
                (hand.index_tip[0] + hand.thumb_tip[0]) * 0.5,
                (hand.index_tip[1] + hand.thumb_tip[1]) * 0.5,
            ],
            sx,
            sy,
        );
        let radius = (hand.scale * sx * PALM_RADIUS_SCALE).clamp(MIN_RADIUS, MAX_RADIUS);
        // Per-axis scaling is correct here rather than anisotropic: landmarks
        // are normalised by frame width and height separately, so this maps
        // each component back to the cells it actually crossed.
        let hand_v = [fin(hand.velocity[0]) * sx, fin(hand.velocity[1]) * sy];

        // Always on, whatever the spell: a hand that does not move the fluid
        // reads as the app not having noticed it.
        if force > 0.0 {
            let rate = HAND_DRAG_RATE * force;
            let mut injected = 0.0;
            let vel = fluid.velocity_mut();
            for_disc(gw, gh, palm, radius, |x, y, _, _, _, w| {
                // The falloff scales the *rate*, not the step: `(T - u) * w * k`
                // composes as `(1 - w*k)^n`, which is not `1 - w*k_total`, so a
                // 360 Hz session would converge measurably faster than a 60 Hz
                // one. Putting the weight inside `decay` makes the coupling a
                // continuous-time rate and the outcome exactly frame-rate
                // independent, at the cost of an `exp` per cell.
                let k = 1.0 - decay(rate * w, dt);
                let (u, v) = vel.get(x, y);
                let du = clamp_impulse((hand_v[0] - u) * k);
                let dv = clamp_impulse((hand_v[1] - v) * k);
                vel.add(x, y, du, dv);
                injected += du * du + dv * dv;
            });
            report.injected += injected;
        }

        match hand.spell {
            // The last two are reported by this layer, never latched.
            Spell::Idle | Spell::Release => {}

            Spell::Vortex => {
                let ramp =
                    smoothstep(0.0, VORTEX_WINDUP, hand.spell_age) * hand.pinch.clamp(0.0, 1.0);
                if ramp > 1e-3 {
                    let outer = radius * VORTEX_RADIUS_SCALE;
                    // Opposite spin per hand, so a two-handed pinch reads as
                    // one symmetric pair instead of two identical eddies.
                    let spin = if hand.right { 1.0 } else { -1.0 };
                    let k = 1.0 - decay(VORTEX_RATE, dt);
                    let mut injected = 0.0;
                    let vel = fluid.velocity_mut();
                    for_disc(gw, gh, pinch_point, outer, |x, y, dx, dy, r, w| {
                        // Zero at the centre: a 1/r tangential field is
                        // singular there, and the projection turns that
                        // singularity into a pressure spike that punches a hole
                        // through the middle of the vortex.
                        let core = smoothstep(0.0, outer * VORTEX_CORE, r);
                        let (nx, ny) = radial(dx, dy, r);
                        let speed = VORTEX_SPEED * ramp * w * core * spin;
                        let (u, v) = vel.get(x, y);
                        let du = clamp_impulse((-ny * speed - u) * k);
                        let dv = clamp_impulse((nx * speed - v) * k);
                        vel.add(x, y, du, dv);
                        injected += du * du + dv * dv;
                    });
                    report.injected += injected;
                    let dye = tint(Spell::Vortex, VORTEX_DYE * ramp * dt);
                    fluid.add_dye(pinch_point[0], pinch_point[1], dye, radius * 0.7);
                }
            }

            Spell::Repel | Spell::Attract => {
                let outward = if hand.spell == Spell::Repel {
                    1.0
                } else {
                    -ATTRACT_RATIO
                };
                let outer = radius * RADIAL_RADIUS_SCALE;
                let accel = REPEL_ACCEL * force * outward * dt;
                let mut injected = 0.0;
                let vel = fluid.velocity_mut();
                for_disc(gw, gh, palm, outer, |x, y, dx, dy, r, w| {
                    let (nx, ny) = radial(dx, dy, r);
                    let du = clamp_impulse(nx * accel * w);
                    let dv = clamp_impulse(ny * accel * w);
                    vel.add(x, y, du, dv);
                    injected += du * du + dv * dv;
                });
                report.injected += injected;
                particles.impulse(
                    palm[0],
                    palm[1],
                    outer,
                    PARTICLE_ACCEL * outward * dt,
                    if outward > 0.0 { 0.3 } else { 0.55 },
                );
                // A fist has to *hold* what it pulled in, or the particles
                // shoot through the palm and orbit back out.
                if hand.spell == Spell::Attract {
                    particles.damp(palm[0], palm[1], outer, GRIP_RATE, dt);
                    // Positional hold: inside the palm the clump is carried with
                    // the fist instead of being swept off by the flow.
                    particles.grip(palm[0], palm[1], radius * GRIP_RADIUS_SCALE, GRIP_PULL, dt);
                }
                let dye = tint(hand.spell, RADIAL_DYE * force * dt);
                fluid.add_dye(palm[0], palm[1], dye, radius);
            }

            Spell::Ignite => {
                // The whole segment since the last frame, not a dot at the
                // current tip: at 30 Hz inference a finger crosses 60 cells
                // between frames and a per-frame dot leaves a dotted line.
                let from = if state.has_tip[slot] {
                    state.last_tip[slot]
                } else {
                    tip
                };
                let span = length(tip[0] - from[0], tip[1] - from[1]);
                let steps = ((span / IGNITE_STEP).ceil() as usize).clamp(1, IGNITE_MAX_STEPS);
                // Split per step so a fast stroke paints the same total dye as
                // a slow one, just spread over more ground.
                let inv = 1.0 / steps as f32;
                let dye = tint(Spell::Ignite, IGNITE_DYE * dt * inv);
                let (dirx, diry) = normalize(tip[0] - from[0], tip[1] - from[1]);
                // The ceiling applies to the *stroke*, not to each splat along
                // it. Consecutive splats are spaced below their own radius so
                // they overlap on purpose, and clamping them individually lets
                // the sum land well past `MAX_IMPULSE` — measured at 1218
                // cells/s for a two-step stroke at `hand_force` 8 and the
                // module's own `MAX_DT`.
                let push = clamp_impulse(IGNITE_ACCEL * force * dt) * inv;
                let (du, dv) = (clamp_impulse(dirx * push), clamp_impulse(diry * push));
                for step in 0..steps {
                    let t = (step + 1) as f32 * inv;
                    let x = lerp(from[0], tip[0], t);
                    let y = lerp(from[1], tip[1], t);
                    fluid.add_dye(x, y, dye, IGNITE_RADIUS);
                    fluid.add_force(x, y, du, dv, IGNITE_RADIUS);
                    report.injected += du * du + dv * dv;
                }
                // Heat only: the particles are already being carried by the
                // fluid, they just need to glow where the flame passed.
                particles.impulse(tip[0], tip[1], IGNITE_RADIUS * 3.0, 0.0, 1.0);
            }

            Spell::Freeze => {
                let outer = radius * FREEZE_RADIUS_SCALE;
                let mut injected = 0.0;
                let vel = fluid.velocity_mut();
                for_disc(gw, gh, palm, outer, |x, y, _, _, _, w| {
                    // Weight in the rate, as in the palm drag above, so the
                    // damping is the same after 100 ms at any frame rate.
                    let damp = 1.0 - decay(FREEZE_RATE * w, dt);
                    let (u, v) = vel.get(x, y);
                    let du = clamp_impulse(-u * damp);
                    let dv = clamp_impulse(-v * damp);
                    vel.add(x, y, du, dv);
                    injected += du * du + dv * dv;
                });
                report.injected += injected;
                let dye = tint(Spell::Freeze, FREEZE_DYE * dt);
                fluid.add_dye(palm[0], palm[1], dye, outer * 0.6);
            }

            Spell::Shatter => {
                if !state.fired[slot] {
                    state.fired[slot] = true;
                    report.bursts += 1;
                    let count = (particles.active() / SHATTER_FRACTION).min(SHATTER_MAX);
                    particles.spawn_burst(
                        palm[0],
                        palm[1],
                        count,
                        SHATTER_SPEED,
                        1.0,
                        params.particle_life,
                    );
                    // A burst is an event, not a force, so its impulse is a
                    // velocity step and is deliberately *not* scaled by dt:
                    // one gesture must hit exactly as hard at 30 fps as at 144.
                    let mut injected = 0.0;
                    let vel = fluid.velocity_mut();
                    for_disc(gw, gh, palm, radius * 2.4, |x, y, dx, dy, r, w| {
                        let (nx, ny) = radial(dx, dy, r);
                        let du = clamp_impulse(nx * SHATTER_IMPULSE * w);
                        let dv = clamp_impulse(ny * SHATTER_IMPULSE * w);
                        vel.add(x, y, du, dv);
                        injected += du * du + dv * dv;
                    });
                    report.injected += injected;
                    let dye = tint(Spell::Shatter, SHATTER_DYE);
                    fluid.add_dye(palm[0], palm[1], dye, radius);
                }
            }
        }

        // Release: a fist that has been held charges a ring, and opening it
        // lets the ring go. Edge-triggered on the latched spell, so it fires
        // once per open no matter how the hand ends up.
        if hand.spell == Spell::Attract {
            state.charge[slot] = smoothstep(0.0, RELEASE_CHARGE_TIME, hand.spell_age);
        }
        if opened_fist {
            // Mid-sequence the ring is a side effect of changing gesture, not a
            // cast: fading it with the wind-up keeps the combo's own flash the
            // only thing that reads as an event.
            let charge = state.charge[slot] * wind_gain;
            state.charge[slot] = 0.0;
            if charge >= RELEASE_MIN_CHARGE {
                fire_release(fluid, particles, params, palm, radius, charge, &mut report);
                state.release_show[slot] = RELEASE_SHOW;
            }
        }
        if state.release_show[slot] > 0.0 {
            state.release_show[slot] -= dt;
            report.spells[slot] = Spell::Release;
        }

        // A completed sequence fires on top of whatever the hand is currently
        // doing: the last gesture of the combo is still a spell, and cutting it
        // off would make a successful cast feel like a dropped frame.
        if let Some(hit) = combo.get(slot).copied().flatten() {
            fire_combo(hit.effect, fluid, particles, params, palm, radius, &mut report);
        }

        state.last_tip[slot] = tip;
        state.has_tip[slot] = true;
    }

    // Rotating both hands warps time. Smoothed, because this multiplies every
    // other subsystem's dt and a one-frame spike is visible everywhere at once.
    let two = tracker.two_hand();
    let target = if two.both_present {
        (1.0 + fin(two.angular_velocity) * TIME_GAIN).clamp(MIN_TIME_SCALE, MAX_TIME_SCALE)
    } else {
        1.0
    };
    let eased =
        state.time_scale + (target - state.time_scale) * (1.0 - decay(TIME_SMOOTH_RATE, dt));
    state.time_scale = if eased.is_finite() {
        eased.clamp(MIN_TIME_SCALE, MAX_TIME_SCALE)
    } else {
        1.0
    };
    report.time_scale = state.time_scale;
    if !report.injected.is_finite() {
        report.injected = 0.0;
    }

    report
}

/// Lets a charged fist go: a shell of particles, a ring-shaped kick into the
/// fluid and a ring of dye, all scaled by how long the fist was held.
fn fire_release(
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
fn fire_combo(
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
            ring_dye(fluid, palm, ring, radius * 0.5, tint(Spell::Shatter, NOVA_DYE));
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

/// An outward velocity step over a disc, falling off to zero at `reach`.
fn radial_kick(
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
fn ring_dye(fluid: &mut Fluid, center: [f32; 2], ring: f32, splat: f32, dye: [f32; 3]) {
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

/// The two whole-field drives — optical flow and the body silhouette's boundary
/// motion — fused into one pass.
///
/// Fused rather than sequential because each pass is a read-modify-write over
/// the whole velocity field; doing them separately doubles the traffic on the
/// largest array in the engine to no benefit.
fn drive_fields(
    flow: &VecField,
    body_edge: &VecField,
    fluid: &mut Fluid,
    params: &Params,
    dt: f32,
    report: &mut SpellReport,
) {
    // Both are injected every frame, so they are *rates*: the field settles at
    // roughly rate / dissipation. The `dt * 60` factor expresses them as "per
    // frame at 60 fps" while staying frame-rate independent.
    let flow_gain = if params.flow_force > 0.0 {
        params.flow_force * dt * 60.0
    } else {
        0.0
    };
    // The engine hands us a zero field whenever no body is tracked, which is
    // the common case, so a linear scan is worth it to skip a full
    // sample-and-write pass over the velocity field.
    let body_gain = if params.body_push > 0.0 && !is_zero(body_edge) {
        params.body_push * dt * 60.0
    } else {
        0.0
    };
    if flow_gain == 0.0 && body_gain == 0.0 {
        return;
    }

    let (gw, gh) = (fluid.width(), fluid.height());
    let (fsx, fsy) = resample_scale(flow, gw, gh);
    let (bsx, bsy) = resample_scale(body_edge, gw, gh);
    let mut injected = 0.0;
    let vel = fluid.velocity_mut();
    for y in 0..gh {
        let fy = y as f32 * fsy;
        let by = y as f32 * bsy;
        for x in 0..gw {
            let mut du = 0.0;
            let mut dv = 0.0;
            if flow_gain != 0.0 {
                let (u, v) = flow.sample(x as f32 * fsx, fy);
                du += u * flow_gain;
                dv += v * flow_gain;
            }
            if body_gain != 0.0 {
                let (u, v) = body_edge.sample(x as f32 * bsx, by);
                du += u * body_gain;
                dv += v * body_gain;
            }
            let du = clamp_impulse(du);
            let dv = clamp_impulse(dv);
            vel.add(x, y, du, dv);
            injected += du * du + dv * dv;
        }
    }
    report.injected += injected;
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
fn for_disc(
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
fn to_grid(p: [f32; 2], sx: f32, sy: f32) -> [f32; 2] {
    [
        (fin(p[0]) * sx).clamp(0.0, sx),
        (fin(p[1]) * sy).clamp(0.0, sy),
    ]
}

/// Unit vector from the disc centre, given the distance the walker already
/// measured. `(0, 0)` at the exact centre, where there is no direction.
#[inline]
fn radial(dx: f32, dy: f32, r: f32) -> (f32, f32) {
    if r > 1e-6 {
        (dx / r, dy / r)
    } else {
        (0.0, 0.0)
    }
}

#[inline]
fn clamp_impulse(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-MAX_IMPULSE, MAX_IMPULSE)
    } else {
        0.0
    }
}

#[inline]
fn fin(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
fn clamp_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(0.0, MAX_DT)
}

/// A spell's hue at a given intensity, ready for [`Fluid::add_dye`].
#[inline]
fn tint(spell: Spell, amount: f32) -> [f32; 3] {
    let a = fin(amount).max(0.0);
    let rgb = hue_to_rgb(spell.hue());
    [rgb[0] * a, rgb[1] * a, rgb[2] * a]
}

/// Grid-space to source-field scale, for sampling a field of a different
/// resolution than the fluid.
#[inline]
fn resample_scale(field: &VecField, gw: usize, gh: usize) -> (f32, f32) {
    if field.w() < 2 || field.h() < 2 || gw < 2 || gh < 2 {
        return (0.0, 0.0);
    }
    (
        (field.w() - 1) as f32 / (gw - 1) as f32,
        (field.h() - 1) as f32 / (gh - 1) as f32,
    )
}

#[inline]
fn is_zero(field: &VecField) -> bool {
    field.u.max_abs() == 0.0 && field.v.max_abs() == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLOW_H, FLOW_W, FLUID_H, FLUID_W};
    use crate::field::Grid;
    use crate::gesture::synth::{self, Hand};
    use crate::particles::ParticleConfig;
    use crate::gesture::{GestureConfig, GestureTracker};

    /// Render step.
    const DT: f32 = 1.0 / 60.0;
    /// Inference step, which is what the tracker is fed at.
    const HAND_DT: f32 = 1.0 / 30.0;

    struct Rig {
        fluid: Fluid,
        particles: Particles,
        flow: VecField,
        body: VecField,
        state: SpellState,
        params: Params,
    }

    impl Rig {
        /// The field drives are off by default so the gesture paths can be
        /// measured on their own; the two tests that care switch them on.
        fn new() -> Self {
            Self::sized(FLUID_W, FLUID_H)
        }

        /// A rig on a smaller grid, for the tests that need hundreds of solver
        /// steps rather than a realistic resolution.
        fn sized(w: usize, h: usize) -> Self {
            let params = Params {
                flow_force: 0.0,
                body_push: 0.0,
                ..Params::default()
            };
            let mut particles = Particles::new(4096, 0xA37E);
            particles.set_active(4096);
            particles.seed_uniform(w, h, params.particle_life);
            Self {
                fluid: Fluid::new(w, h),
                particles,
                flow: VecField::new(FLOW_W, FLOW_H),
                body: VecField::new(w, h),
                state: SpellState::new(),
                params,
            }
        }

        fn run(&mut self, tracker: &GestureTracker, dt: f32) -> SpellReport {
            self.run_with(tracker, None, dt)
        }

        /// One step with a completed sequence handed in, as the engine does it.
        fn run_with(
            &mut self,
            tracker: &GestureTracker,
            combo: Option<ComboHit>,
            dt: f32,
        ) -> SpellReport {
            apply(
                tracker,
                &self.flow,
                &self.body,
                &mut self.fluid,
                &mut self.particles,
                &mut self.state,
                &self.params,
                {
                    let mut per_hand = [None; 2];
                    if let Some(h) = combo {
                        per_hand[h.slot.min(1)] = Some(h);
                    }
                    per_hand
                },
                [0.0; 2],
                dt,
            )
        }

        fn velocity_at(&self, x: f32, y: f32) -> (f32, f32) {
            self.fluid.velocity().sample(x, y)
        }
    }

    fn empty_tracker() -> GestureTracker {
        let mut t = GestureTracker::new(GestureConfig::default());
        t.update_hands(&synth::buffer(), HAND_DT);
        t.update_two_hand(HAND_DT);
        t
    }

    /// A tracker holding one still hand, held long enough to latch and to age
    /// past the vortex windup.
    fn holding(hand: &Hand) -> GestureTracker {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        synth::write(&mut buf, 0, hand);
        for _ in 0..60 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
        }
        t
    }

    /// A still hand held just long enough to latch its spell.
    fn latched(hand: &Hand) -> GestureTracker {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        synth::write(&mut buf, 0, hand);
        for _ in 0..10 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
        }
        t
    }

    /// The same, but sweeping right at 0.36 normalised units per second.
    fn sweeping(hand: &Hand) -> GestureTracker {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        let mut moving = *hand;
        for _ in 0..60 {
            synth::write(&mut buf, 0, &moving);
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
            moving.x += 0.012;
        }
        t
    }

    /// Two palms rotating about the frame centre at `omega` radians per second.
    fn rotating(omega: f32, frames: usize) -> GestureTracker {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        for frame in 0..frames {
            let theta = omega * frame as f32 * HAND_DT;
            let (c, s) = (theta.cos(), theta.sin());
            synth::write(&mut buf, 0, &Hand::at(0.5 + 0.12 * c, 0.5 + 0.12 * s));
            synth::write(&mut buf, 1, &Hand::at(0.5 - 0.12 * c, 0.5 - 0.12 * s));
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
        }
        t
    }

    /// Grid-space palm of slot 0.
    fn palm_of(tracker: &GestureTracker) -> [f32; 2] {
        to_grid(
            tracker.hands()[0].palm,
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        )
    }

    const IDLE_HAND: Hand = Hand {
        x: 0.4,
        y: 0.5,
        scale: 0.13,
        gesture: synth::NONE,
        // Unconfident, so no label path fires and the geometry is ambiguous.
        score: 0.1,
        pinched: false,
        right: true,
    };

    #[test]
    fn an_idle_present_hand_injects_energy_and_an_absent_one_does_not() {
        let tracker = sweeping(&IDLE_HAND);
        assert_eq!(tracker.hands()[0].spell, Spell::Idle, "expected no spell");
        assert!(
            tracker.hands()[0].velocity[0] > 0.2,
            "hand should be moving"
        );

        let mut rig = Rig::new();
        let report = rig.run(&tracker, DT);
        assert_eq!(report.spells[0], Spell::Idle);
        assert!(
            report.injected > 0.0,
            "an idle hand must still drive the fluid"
        );
        assert!(rig.fluid.energy() > 0.0);
        // It must push the way the hand is going.
        let palm = palm_of(&tracker);
        assert!(rig.velocity_at(palm[0], palm[1]).0 > 0.0);

        let mut rig = Rig::new();
        let report = rig.run(&empty_tracker(), DT);
        assert_eq!(report.injected, 0.0, "no hand, no energy");
        assert_eq!(rig.fluid.energy(), 0.0);
        assert_eq!(report.spells, [Spell::Idle, Spell::Idle]);
    }

    #[test]
    fn every_spell_keeps_the_fluid_finite_and_under_the_ceiling() {
        let cases = [
            (Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), Spell::Repel),
            (
                Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST),
                Spell::Attract,
            ),
            (
                Hand::at(0.5, 0.5).gesture(synth::POINTING_UP),
                Spell::Ignite,
            ),
            (Hand::at(0.5, 0.5).gesture(synth::VICTORY), Spell::Freeze),
            (Hand::at(0.5, 0.5).gesture(synth::THUMB_UP), Spell::Shatter),
            (
                Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched(),
                Spell::Vortex,
            ),
            (IDLE_HAND, Spell::Idle),
        ];
        for (hand, want) in cases {
            let tracker = sweeping(&hand);
            assert_eq!(tracker.hands()[0].spell, want, "wrong spell latched");
            let mut rig = Rig::new();
            for _ in 0..4 {
                let report = rig.run(&tracker, DT);
                assert!(report.injected.is_finite(), "{want:?} injected NaN");
                assert!(report.time_scale.is_finite());
            }
            let vel = rig.fluid.velocity();
            assert!(
                vel.u.max_abs() <= MAX_IMPULSE && vel.v.max_abs() <= MAX_IMPULSE,
                "{want:?} exceeded MAX_IMPULSE: {} / {}",
                vel.u.max_abs(),
                vel.v.max_abs()
            );
            assert_eq!(rig.fluid.sanitize(), 0, "{want:?} wrote a non-finite cell");
            for channel in rig.fluid.dye() {
                assert!(
                    channel.data.iter().all(|v| v.is_finite()),
                    "{want:?} wrote non-finite dye"
                );
            }
        }
    }

    #[test]
    fn absurd_parameters_are_clamped_to_the_impulse_ceiling() {
        // hand_force saturates at 8 and dt at 0.25, which together ask for a
        // 3000 cells/s push — the clamp is the only thing between that and a
        // velocity field the projection cannot resolve.
        let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
        let mut rig = Rig::new();
        rig.params.hand_force = 8.0;
        rig.run(&tracker, 1e6);
        let vel = rig.fluid.velocity();
        assert!(
            (vel.u.max_abs() - MAX_IMPULSE).abs() < 1e-3,
            "clamp did not bite: {}",
            vel.u.max_abs()
        );
        assert!(vel.v.max_abs() <= MAX_IMPULSE);
        assert_eq!(rig.fluid.sanitize(), 0);
    }

    #[test]
    fn shatter_fires_once_per_hold_and_rearms_on_release() {
        let hand = Hand::at(0.5, 0.5).gesture(synth::THUMB_UP);
        let mut tracker = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        synth::write(&mut buf, 0, &hand);
        let mut rig = Rig::new();

        let mut bursts = 0;
        for _ in 0..30 {
            tracker.update_hands(&buf, HAND_DT);
            tracker.update_two_hand(HAND_DT);
            bursts += rig.run(&tracker, DT).bursts;
        }
        assert_eq!(tracker.hands()[0].spell, Spell::Shatter);
        assert_eq!(bursts, 1, "holding the gesture re-fired the burst");

        // Drop the hand for long enough to be released, then show it again.
        let empty = synth::buffer();
        for _ in 0..12 {
            tracker.update_hands(&empty, HAND_DT);
            tracker.update_two_hand(HAND_DT);
            bursts += rig.run(&tracker, DT).bursts;
        }
        assert_eq!(bursts, 1, "a burst fired while no hand was present");
        for _ in 0..30 {
            tracker.update_hands(&buf, HAND_DT);
            tracker.update_two_hand(HAND_DT);
            bursts += rig.run(&tracker, DT).bursts;
        }
        assert_eq!(bursts, 2, "the burst did not re-arm after release");

        // And the burst actually did something to the pool.
        assert!(
            (0..rig.particles.active()).any(|i| rig.particles.heat_of(i) > 0.9),
            "no particle came out of the burst hot"
        );
    }

    #[test]
    fn ignite_paints_the_whole_stroke_not_just_the_endpoints() {
        let mut rig = Rig::new();
        let left = holding(&Hand::at(0.15, 0.5).gesture(synth::POINTING_UP));
        assert_eq!(left.hands()[0].spell, Spell::Ignite);
        rig.run(&left, DT);
        let a = to_grid(
            left.hands()[0].index_tip,
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );

        // The finger is now far away; one frame of dye must cover the gap.
        let right = holding(&Hand::at(0.85, 0.5).gesture(synth::POINTING_UP));
        rig.run(&right, DT);
        let b = to_grid(
            right.hands()[0].index_tip,
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        let span = length(b[0] - a[0], b[1] - a[1]);
        assert!(span > 100.0, "the test needs a long stroke, got {span}");

        let red = &rig.fluid.dye()[0];
        for t in [0.25, 0.5, 0.75] {
            let x = lerp(a[0], b[0], t);
            let y = lerp(a[1], b[1], t);
            assert!(
                red.sample(x, y) > 1e-4,
                "no dye at {t} along the stroke ({x}, {y})"
            );
        }
        // Mid-stroke brightness must be the same order as the endpoints, not a
        // faint smear from the splat tails.
        let mid = red.sample(lerp(a[0], b[0], 0.5), lerp(a[1], b[1], 0.5));
        let end = red.sample(b[0], b[1]);
        assert!(mid * 4.0 > end, "stroke is dotted: mid {mid} vs end {end}");
    }

    #[test]
    fn vortex_winds_up_and_the_flow_is_tangential() {
        let hand = Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched();

        // Freshly latched versus held past the windup.
        let mut young = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        synth::write(&mut buf, 0, &hand);
        for _ in 0..GestureConfig::default().commit_frames {
            young.update_hands(&buf, HAND_DT);
            young.update_two_hand(HAND_DT);
        }
        assert_eq!(young.hands()[0].spell, Spell::Vortex);
        let mut fresh_rig = Rig::new();
        fresh_rig.run(&young, DT);
        let fresh = fresh_rig.fluid.energy();

        let old = holding(&hand);
        assert!(old.hands()[0].spell_age > VORTEX_WINDUP);
        let mut wound_rig = Rig::new();
        wound_rig.run(&old, DT);
        let wound = wound_rig.fluid.energy();
        assert!(
            wound > fresh * 4.0,
            "vortex did not wind up: {fresh} -> {wound}"
        );

        // Sample to the right of the pinch point: the flow there must be
        // perpendicular to the radius.
        let tip = old.hands()[0].index_tip;
        let thumb = old.hands()[0].thumb_tip;
        let center = to_grid(
            [(tip[0] + thumb[0]) * 0.5, (tip[1] + thumb[1]) * 0.5],
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        let radius = old.hands()[0].scale * (FLUID_W - 1) as f32 * PALM_RADIUS_SCALE;
        let probe = radius * VORTEX_RADIUS_SCALE * 0.4;
        let (u, v) = wound_rig.velocity_at(center[0] + probe, center[1]);
        assert!(v.abs() > 1.0, "no swirl at the probe: {v}");
        assert!(
            u.abs() < 0.35 * v.abs(),
            "flow is radial, not tangential: u {u} v {v}"
        );
        // Opposite sides of the core circulate opposite ways.
        let (_, v_left) = wound_rig.velocity_at(center[0] - probe, center[1]);
        assert!(v * v_left < 0.0, "not a circulation: {v} vs {v_left}");
    }

    #[test]
    fn repel_pushes_out_and_attract_pulls_in_more_gently() {
        let mut out = Rig::new();
        let repel = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
        let palm = palm_of(&repel);
        // A particle just to the right of the palm, to check the particle path.
        out.particles
            .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
        out.run(&repel, DT);
        let (ru, _) = out.velocity_at(palm[0] + 10.0, palm[1]);
        assert!(ru > 0.0, "repel pushed inward: {ru}");
        assert!(
            out.particles.velocity(0).0 > 0.0,
            "repel did not push particles out"
        );

        let mut inward = Rig::new();
        let attract = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        inward
            .particles
            .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
        inward.run(&attract, DT);
        let (au, _) = inward.velocity_at(palm[0] + 10.0, palm[1]);
        assert!(au < 0.0, "attract pushed outward: {au}");
        assert!(
            inward.particles.velocity(0).0 < 0.0,
            "attract did not pull particles in"
        );
        assert!(
            au.abs() < ru.abs() * 0.8,
            "attract must be the weaker well: {au} vs {ru}"
        );

        // Both should be symmetric about the palm.
        let (lu, _) = out.velocity_at(palm[0] - 10.0, palm[1]);
        assert!((lu + ru).abs() < 0.2 * ru.abs(), "asymmetric push");
    }

    #[test]
    fn every_combo_payload_moves_the_field_and_stays_finite() {
        for effect in [
            ComboEffect::Nova,
            ComboEffect::Tempest,
            ComboEffect::Supernova,
        ] {
            let mut rig = Rig::new();
            let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
            let hit = ComboHit {
                combo: 0,
                effect,
                slot: 0,
            };
            let report = rig.run_with(&tracker, Some(hit), DT);
            assert_eq!(report.bursts, 1, "{effect:?} did not report its burst");
            assert!(
                report.injected > 0.0 && report.injected.is_finite(),
                "{effect:?} injected {}",
                report.injected
            );
            // A payload is an event; it must not leave the solver unstable.
            for _ in 0..30 {
                rig.run(&tracker, DT);
            }
            assert_eq!(rig.fluid.sanitize(), 0, "{effect:?} destabilised the fluid");
        }
    }

    #[test]
    fn freeze_damps_the_fluid_locally_and_leaves_the_rest_alone() {
        let mut rig = Rig::new();
        rig.fluid.velocity_mut().u.data.fill(50.0);
        let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::VICTORY));
        assert_eq!(tracker.hands()[0].spell, Spell::Freeze);
        let palm = palm_of(&tracker);
        for _ in 0..10 {
            rig.run(&tracker, DT);
        }
        let (near, _) = rig.velocity_at(palm[0], palm[1]);
        // Not a "did it go down" test: the always-on palm drag relaxes toward a
        // still hand's zero velocity all by itself, so a loose threshold here
        // passes for `Idle` too and proves nothing about the spell. Measure the
        // spell alone with the drag switched off, against its own rate.
        assert!(near < 40.0, "freeze did not damp the local flow: {near}");
        let (far, _) = rig.velocity_at(5.0, 5.0);
        assert!((far - 50.0).abs() < 1e-3, "freeze damped the whole grid");
        // And it lays down cold dye: hue 0.55 has no red in it.
        let dye = rig.fluid.dye();
        assert!(dye[2].sample(palm[0], palm[1]) > dye[0].sample(palm[0], palm[1]));

        // The falloff weight is ~1 over the middle of the disc, so the centre
        // must decay by exactly `exp(-FREEZE_RATE * t)` — and must do so at any
        // frame rate, since the weight sits inside the decay rate.
        for (hz, frames) in [(60.0f32, 30u32), (240.0, 120)] {
            let mut rig = Rig::new();
            rig.params.hand_force = 0.0;
            rig.fluid.velocity_mut().u.data.fill(50.0);
            for _ in 0..frames {
                rig.run(&tracker, 1.0 / hz);
            }
            let elapsed = frames as f32 / hz;
            let want = 50.0 * decay(FREEZE_RATE, elapsed);
            let got = rig.velocity_at(palm[0], palm[1]).0;
            assert!(
                (got - want).abs() < 0.02 * want,
                "{hz} Hz: freeze centre {got}, analytic {want}"
            );
        }

        // Nothing but the drag: an idle hand must leave the field alone once
        // the drag is off, which is what makes the comparison above meaningful.
        let mut idle = Rig::new();
        idle.params.hand_force = 0.0;
        idle.fluid.velocity_mut().u.data.fill(50.0);
        let idle_tracker = holding(&IDLE_HAND);
        assert_eq!(idle_tracker.hands()[0].spell, Spell::Idle);
        for _ in 0..30 {
            idle.run(&idle_tracker, 1.0 / 60.0);
        }
        let palm = palm_of(&idle_tracker);
        assert!(
            (idle.velocity_at(palm[0], palm[1]).0 - 50.0).abs() < 1e-3,
            "an idle hand damped the field"
        );
    }

    #[test]
    fn an_ignite_stroke_respects_the_impulse_ceiling() {
        // The trail is a line of deliberately overlapping splats: they are
        // spaced `IGNITE_STEP` apart and each is `IGNITE_RADIUS` wide, so
        // clamping them one at a time is not enough — the sum is what lands in
        // the velocity field. Worst case is the module's own `MAX_DT` and the
        // largest `hand_force` `Params::set` allows.
        let mut worst = 0.0f32;
        for cells in [0.0f32, 1.2, 2.2, 3.4, 4.8, 9.0, 40.0, 200.0] {
            let mut rig = Rig::new();
            rig.params.hand_force = 8.0;
            let a = holding(&Hand::at(0.3, 0.5).gesture(synth::POINTING_UP));
            rig.run(&a, MAX_DT);
            let b = holding(
                &Hand::at(0.3 + cells / (FLUID_W - 1) as f32, 0.5).gesture(synth::POINTING_UP),
            );
            rig.run(&b, MAX_DT);
            let vel = rig.fluid.velocity();
            worst = worst.max(vel.u.max_abs()).max(vel.v.max_abs());
        }
        assert!(
            worst <= MAX_IMPULSE,
            "ignite drove the field to {worst} cells/s, past the {MAX_IMPULSE} ceiling"
        );
    }

    #[test]
    fn time_scale_follows_the_two_hand_rotation_and_stays_clamped() {
        let mut rig = Rig::new();
        let forward = rotating(2.5, 90);
        assert!(forward.two_hand().both_present);
        let mut ts = 1.0;
        for _ in 0..60 {
            ts = rig.run(&forward, DT).time_scale;
        }
        assert!(ts > 1.1, "rotation did not warp time: {ts}");
        assert!(ts <= MAX_TIME_SCALE);

        let backward = rotating(-2.5, 90);
        for _ in 0..60 {
            ts = rig.run(&backward, DT).time_scale;
        }
        assert!(ts < 0.9, "reverse rotation should slow time: {ts}");
        assert!(ts >= MIN_TIME_SCALE);

        // An absurd spin must saturate, not run away.
        let wild = rotating(60.0, 120);
        for _ in 0..240 {
            ts = rig.run(&wild, DT).time_scale;
        }
        assert!(
            (MIN_TIME_SCALE..=MAX_TIME_SCALE).contains(&ts),
            "time scale escaped its bounds: {ts}"
        );

        // And it eases back to normal once the hands are gone.
        let none = empty_tracker();
        for _ in 0..120 {
            ts = rig.run(&none, DT).time_scale;
        }
        assert!((ts - 1.0).abs() < 0.05, "time scale did not recover: {ts}");
    }

    #[test]
    fn the_whole_field_drives_push_the_fluid() {
        // Optical flow: model-free, so it must work with no tracker at all.
        let mut rig = Rig::new();
        rig.params.flow_force = 0.45;
        rig.flow.u.data.fill(4.0);
        let report = rig.run(&empty_tracker(), DT);
        assert!(report.injected > 0.0);
        let (u, _) = rig.velocity_at(100.0, 70.0);
        assert!(u > 0.0, "flow drive pushed nothing: {u}");

        // Body edge velocity, injected only where the silhouette moves.
        let mut rig = Rig::new();
        rig.params.body_push = 1.0;
        for y in 60..80 {
            for x in 100..120 {
                rig.body.u.set(x, y, -6.0);
            }
        }
        rig.run(&empty_tracker(), DT);
        let (inside, _) = rig.velocity_at(110.0, 70.0);
        let (outside, _) = rig.velocity_at(10.0, 10.0);
        assert!(inside < 0.0, "body edge did not push: {inside}");
        assert_eq!(outside, 0.0, "body edge leaked across the grid");

        // A zero body field must cost nothing and change nothing.
        let mut rig = Rig::new();
        rig.params.body_push = 1.0;
        assert_eq!(rig.run(&empty_tracker(), DT).injected, 0.0);
    }

    #[test]
    fn forces_are_framerate_independent() {
        // The same simulated interval, split six ways. Both paths must deposit
        // the same energy; a `1 - rate * dt` decay or an unscaled impulse fails
        // this by a factor of six.
        let tracker = sweeping(&IDLE_HAND);
        let mut coarse = Rig::new();
        coarse.run(&tracker, DT);
        let mut fine = Rig::new();
        for _ in 0..6 {
            fine.run(&tracker, DT / 6.0);
        }
        let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
        assert!(a > 0.0);
        assert!(
            (a - b).abs() < 0.005 * a,
            "frame rate changed the outcome: {a} vs {b}"
        );

        // Repel is a pure dt-scaled impulse. Probed outside the palm drag's
        // radius but inside the push's, where nothing else has touched the
        // field, the two paths must agree to floating-point noise.
        let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
        let palm = palm_of(&tracker);
        let mut coarse = Rig::new();
        coarse.run(&tracker, DT);
        let mut fine = Rig::new();
        for _ in 0..6 {
            fine.run(&tracker, DT / 6.0);
        }
        let probe = palm[0] + 50.0;
        let (a, b) = (
            coarse.velocity_at(probe, palm[1]).0,
            fine.velocity_at(probe, palm[1]).0,
        );
        assert!(a > 1.0, "the probe should see the push: {a}");
        assert!(
            (a - b).abs() < 1e-3 * a,
            "impulse not dt-scaled: {a} vs {b}"
        );

        // Over the whole field the two differ by a few percent, and that is
        // operator splitting rather than a dt bug: the fine path interleaves
        // six drag passes with six impulses, so the drag sees — and damps — a
        // field the coarse path only produces after its single drag pass.
        let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
        assert!(
            (a - b).abs() < 0.04 * a,
            "splitting error too large: {a} vs {b}"
        );
    }

    #[test]
    fn a_held_spell_survives_a_long_stepped_run() {
        // The real failure mode this guards: a force that is individually
        // clamped but accumulates every frame until the solver diverges.
        for gesture in [
            synth::OPEN_PALM,
            synth::CLOSED_FIST,
            synth::POINTING_UP,
            synth::VICTORY,
            synth::THUMB_UP,
        ] {
            let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(gesture));
            // A sixteenth of the real grid: this test is about what accumulates
            // over 240 steps, which is resolution independent, and at full size
            // five of these runs take a minute and a half in a debug build.
            let mut rig = Rig::sized(64, 36);
            for _ in 0..240 {
                rig.run(&tracker, DT);
                rig.fluid.step(DT, &rig.params);
            }
            assert_eq!(
                rig.fluid.sanitize(),
                0,
                "gesture {gesture} produced non-finite cells"
            );
            let speed = rig.fluid.max_speed();
            assert!(
                speed.is_finite() && speed < MAX_IMPULSE * 4.0,
                "gesture {gesture} diverged: {speed} cells/s"
            );
        }
    }

    #[test]
    fn garbage_input_never_panics() {
        let tracker = sweeping(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));

        // Absurd timesteps.
        let mut rig = Rig::new();
        for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e9] {
            let report = rig.run(&tracker, dt);
            assert!(report.time_scale.is_finite());
            assert!(report.injected.is_finite());
        }
        assert_eq!(rig.fluid.sanitize(), 0);

        // A non-finite flow field must not reach the velocity grid.
        let mut rig = Rig::new();
        rig.params.flow_force = 1.0;
        rig.params.body_push = 1.0;
        rig.flow.u.data.fill(f32::NAN);
        rig.body.v.data.fill(f32::INFINITY);
        rig.run(&tracker, DT);
        assert_eq!(rig.fluid.sanitize(), 0, "NaN flow leaked into the fluid");

        // A landmark-free "present" hand, and hands off the edge of the frame.
        let mut rig = Rig::new();
        for hand in [
            Hand::at(-4.0, 9.0).gesture(synth::OPEN_PALM),
            Hand::at(0.5, 0.5).gesture(synth::POINTING_UP).scaled(0.9),
            Hand::at(0.5, 0.5).gesture(synth::THUMB_UP).scaled(0.006),
        ] {
            let tracker = holding(&hand);
            rig.run(&tracker, DT);
        }
        assert_eq!(rig.fluid.sanitize(), 0);

        // The smallest grid a Fluid can exist on, and an empty particle pool.
        let mut tiny = Fluid::new(2, 2);
        let mut none = Particles::new(0, 1);
        let mut state = SpellState::new();
        let report = apply(
            &tracker,
            &VecField::new(2, 2),
            &VecField::new(2, 2),
            &mut tiny,
            &mut none,
            &mut state,
            &Params::default(),
            [None; 2],
            [0.0; 2],
            DT,
        );
        assert!(report.injected.is_finite());
        assert_eq!(tiny.sanitize(), 0);
    }

    #[test]
    fn a_sweeping_open_palm_still_repels() {
        let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(synth::OPEN_PALM));
        assert_eq!(tracker.hands()[0].spell, Spell::Repel);
    }

    #[test]
    fn a_closed_fist_gathers_particles_instead_of_slinging_them_past() {
        let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        let hand = &tracker.hands()[0];
        let (sx, sy) = ((FLUID_W - 1) as f32, (FLUID_H - 1) as f32);
        let palm = to_grid(hand.palm, sx, sy);

        let mut rig = Rig::new();
        rig.particles.place(0, palm[0] + 10.0, palm[1], 0.0, 0.0, 30.0);
        let obstacle = Grid::new(FLUID_W, FLUID_H);
        let cfg = ParticleConfig::default();
        for _ in 0..60 {
            rig.run(&tracker, DT);
            rig.particles
                .step(rig.fluid.velocity(), &obstacle, DT, &rig.params, &cfg);
        }
        let (px, py) = rig.particles.position(0);
        let dist = length(px - palm[0], py - palm[1]);
        let (vx, vy) = rig.particles.velocity(0);
        assert!(dist < 10.0, "particle was not gathered: {dist} cells out");
        // Held, not orbiting: the grip has taken the speed off it.
        assert!(length(vx, vy) < 60.0, "particle still slinging: {vx} {vy}");
    }

    #[test]
    fn one_pointing_hand_is_still_an_ignite() {
        let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::POINTING_UP));
        let mut rig = Rig::new();
        let report = rig.run(&tracker, DT);
        assert_eq!(report.spells[0], Spell::Ignite);
    }

    #[test]
    fn opening_a_held_fist_fires_a_ring_once() {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        let mut rig = Rig::new();
        synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        for _ in 0..45 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
            rig.run(&t, HAND_DT);
        }
        assert_eq!(t.hands()[0].spell, Spell::Attract);
        let palm = palm_of(&t);

        // Still the pool first so the only fast particles afterwards are the
        // ring's own, not two seconds of attract pull.
        rig.particles
            .seed_uniform(FLUID_W, FLUID_H, rig.params.particle_life);
        // Open the hand: the tracker takes `commit_frames` to relatch.
        synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
        let mut bursts = 0;
        let mut saw_release = false;
        for _ in 0..12 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
            let report = rig.run(&t, HAND_DT);
            bursts += report.bursts;
            saw_release |= report.spells[0] == Spell::Release;
        }
        assert_eq!(bursts, 1, "the ring must fire exactly once per open");
        assert!(saw_release, "the release should be reported to the HUD");
        // Particles were re-seeded on a ring and fly outward from the palm.
        let mut outward = 0;
        let mut total = 0;
        for i in 0..rig.particles.active() {
            let (px, py) = rig.particles.position(i);
            let (vx, vy) = rig.particles.velocity(i);
            let speed = length(vx, vy);
            if speed < 50.0 {
                continue;
            }
            total += 1;
            if (px - palm[0]) * vx + (py - palm[1]) * vy > 0.0 {
                outward += 1;
            }
        }
        assert!(total > 100, "expected a burst of fast particles, got {total}");
        assert!(outward * 10 > total * 9, "burst is not outward: {outward}/{total}");
    }

    #[test]
    fn a_fist_barely_closed_does_not_fire_a_ring() {
        let mut t = GestureTracker::new(GestureConfig::default());
        let mut buf = synth::buffer();
        let mut rig = Rig::new();
        synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        for _ in 0..4 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
            rig.run(&t, HAND_DT);
        }
        assert_eq!(t.hands()[0].spell, Spell::Attract);
        synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
        let mut bursts = 0;
        for _ in 0..12 {
            t.update_hands(&buf, HAND_DT);
            t.update_two_hand(HAND_DT);
            bursts += rig.run(&t, HAND_DT).bursts;
        }
        assert_eq!(bursts, 0, "an uncharged fist must not fire");
    }
}
