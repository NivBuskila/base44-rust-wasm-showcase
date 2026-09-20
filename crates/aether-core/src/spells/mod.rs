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
use crate::duet::{DuetEffect, DuetFrame};
use crate::field::VecField;
use crate::fluid::Fluid;
use crate::gesture::{GestureTracker, Spell};
use crate::math::{decay, hue_to_rgb, length, lerp, normalize, smoothstep};
use crate::particles::Particles;

mod bursts;
mod drive;
mod duets;
mod geom;
#[cfg(test)]
mod tests;

use bursts::{fire_combo, fire_release};
use drive::drive_fields;
use duets::apply_duet;
use geom::*;

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
/// Seconds the HUD keeps showing "release" after the ring fires.
const RELEASE_SHOW: f32 = 0.45;

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
    // Two-hand duets recognised by `crate::duet`: a continuous tide strength
    // and at most one event, both applied between the hands.
    duet: DuetFrame,
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
            fire_combo(
                hit.effect,
                fluid,
                particles,
                params,
                palm,
                radius,
                &mut report,
            );
        }

        state.last_tip[slot] = tip;
        state.has_tip[slot] = true;
    }

    apply_duet(
        tracker,
        fluid,
        particles,
        params,
        duet,
        sx,
        sy,
        dt,
        &mut report,
    );

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
