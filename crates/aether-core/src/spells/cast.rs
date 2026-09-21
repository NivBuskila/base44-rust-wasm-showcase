//! One hand's cast: the always-on palm drag, the per-spell effect, the release
//! ring and the combo flash — everything `apply` does inside its per-hand loop.
//!
//! Split out because the loop body carried every spell's field arithmetic while
//! the parent only sequences the layer: `apply` now reads as drive, then a cast
//! per hand, then the duets and the time warp.

use super::*;

/// Applies `hand`'s spell for this step. `charging` is its combo wind-up depth
/// and `combo` the sequence, if any, that completed on this hand.
#[allow(clippy::too_many_arguments)]
pub(super) fn cast_hand(
    slot: usize,
    hand: &HandState,
    fluid: &mut Fluid,
    particles: &mut Particles,
    state: &mut SpellState,
    params: &Params,
    combo: Option<ComboHit>,
    charging: f32,
    // Grid dimensions, and the normalised-to-grid position/velocity scale.
    (gw, gh): (usize, usize),
    (sx, sy): (f32, f32),
    base_force: f32,
    dt: f32,
    report: &mut SpellReport,
) {
    // How far this hand is into a sequence, and the matching fade applied to
    // its own spell effects while it winds that sequence up.
    let winding = fin(charging).clamp(0.0, 1.0);
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
            let ramp = smoothstep(0.0, VORTEX_WINDUP, hand.spell_age) * hand.pinch.clamp(0.0, 1.0);
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
            fire_release(fluid, particles, params, palm, radius, charge, report);
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
    if let Some(hit) = combo {
        fire_combo(hit.effect, fluid, particles, params, palm, radius, report);
    }

    state.last_tip[slot] = tip;
    state.has_tip[slot] = true;
}
