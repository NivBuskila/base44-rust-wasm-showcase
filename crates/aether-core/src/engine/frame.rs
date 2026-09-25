//! One simulated frame, in the order the subsystems have to run.
//!
//! The order is not arbitrary: duets are recognised before combos (a charged
//! big bang owns both hands and its release is also each hand's `nova`), the
//! spell layer injects every source before the solve so the same step carries it.
//!
//! Only real time drives the perception and pacing clocks; the two-hand
//! time-warp scales the `dt` handed to the simulation, never the clocks.

use super::{sane_dt, Engine, AMBIENT_AFTER, SANITIZE_EVERY};
use crate::config::{FLUID_H, FLUID_W};
use crate::math::hue_to_rgb;
use crate::par;
use crate::spells;

impl Engine {
    // ----------------------------------------------------------------- step

    /// Advances the whole engine by `dt` real seconds.
    pub fn step(&mut self, dt: f32) {
        // One handoff into the thread pool per frame rather than one per
        // parallel region; see `par::install`. Serial builds call straight in.
        par::install(|| self.advance(dt));
    }

    fn advance(&mut self, dt: f32) {
        let real_dt = sane_dt(dt);
        self.time += real_dt;
        self.frame = self.frame.wrapping_add(1);
        self.since_perception += real_dt;

        // The two-hand time-warp gesture scales simulated time, not real time,
        // so the frame pacing and the perception clocks stay untouched.
        let warped_dt = sane_dt(real_dt * self.params.time_scale * self.report.time_scale);
        self.last_sim_dt = warped_dt;
        // A GPU pool replays last frame's ops from the log; anything still in
        // it now was drained (or never read) and must not be replayed twice.
        self.particles.clear_ops();

        self.body.decay(real_dt);

        let ambient = self.since_perception > AMBIENT_AFTER;
        if ambient {
            self.drive_ambient(warped_dt);
        }

        // Re-uploading the obstacle is a full grid resample plus a wall-mask
        // rebuild, so it is skipped while no body is in view. With one in view
        // the field changes every frame, not just when a mask lands: the fade
        // envelope starts the moment each mask arrives (`BodyMask::decay`), so
        // the frames between masks carry a slightly dimmer copy. The dirty flag
        // carries one upload past `present()` dropping, the one that empties
        // the solver's copy.
        if self.obstacle_dirty || self.body.present() {
            self.fluid.set_obstacle(self.body.obstacle());
            self.obstacle_dirty = self.body.present();
        }

        // A borrow of `self.body` alone, so it can be held across the `&mut`
        // borrows of `fluid` and `particles` that `spells::apply` takes below.
        let body_edge = if self.body.present() {
            self.body.edge_velocity()
        } else {
            &self.idle_field
        };

        // Sequences are recognised before the spell layer runs, so a combo
        // that completes this frame lands in the same step as the gesture that
        // completed it.
        // Duets are recognised first: a charged big bang owns both hands, and
        // its fist -> palm release is also the `nova` combo on each hand. Left
        // alone the pair fires two small novas a frame before the supernova and
        // the charged payoff is lost inside them, so those hits are dropped.
        let mut duet = self.duets.update(&self.tracker, warped_dt);
        let mut hits = self.combos.update(&self.tracker, warped_dt);
        if duet.fire.is_some() || self.duets.claims_hands() {
            hits = [None, None];
        }
        // Practice mode: single gestures still cast, sequences and duets do not.
        // The tutorial teaches one pose after another, and any two poses in a
        // row are a candidate sequence — so without this the lesson fires
        // combos the caster never asked for.
        if self.practice {
            hits = [None, None];
            duet.fire = None;
        }
        let charging = [self.combos.charging(0), self.combos.charging(1)];
        let report = {
            let tracker = &self.tracker;
            let flow = self.flow.flow();
            spells::apply(
                tracker,
                flow,
                body_edge,
                &mut self.fluid,
                &mut self.particles,
                &mut self.spell_state,
                &self.params,
                hits,
                charging,
                duet,
                warped_dt,
            )
        };
        self.report = report;

        self.fluid.step(warped_dt, &self.params);

        let repairs = if self.frame.is_multiple_of(SANITIZE_EVERY) {
            self.fluid.sanitize()
        } else {
            0
        };

        self.particles.step(
            self.fluid.velocity(),
            self.fluid.obstacle(),
            warped_dt,
            &self.params,
            &self.particle_cfg,
        );
        self.particles.build_render_buffer(FLUID_W, FLUID_H);

        self.encode_dye();
        self.collect_stats(repairs, ambient);
    }

    /// When nothing has been perceived for a while, keep the screen alive with
    /// slow wandering vortices. A dead-black canvas reads as "broken app", and
    /// this also gives the headless test something deterministic to assert on.
    fn drive_ambient(&mut self, dt: f32) {
        let t = self.time;
        for k in 0..3 {
            let phase = t * (0.11 + 0.037 * k as f32) + k as f32 * 2.1;
            let x = (FLUID_W as f32) * (0.5 + 0.34 * (phase * 0.7).sin());
            let y = (FLUID_H as f32) * (0.5 + 0.30 * (phase * 0.9).cos());
            let dir = phase * 1.7;
            // Injected every frame, so these are *rates*: the field settles at
            // roughly rate / dissipation. The `dt * 60.0` factor expresses them
            // as "per frame at 60 fps" while staying frame-rate independent.
            let mag = 10.0 * dt * 60.0;
            self.fluid
                .add_force(x, y, dir.cos() * mag, dir.sin() * mag, 12.0);
            let hue = 0.55 + 0.12 * (t * 0.05 + k as f32 * 0.33).sin();
            let rgb = hue_to_rgb(hue);
            let amount = 0.03 * dt * 60.0;
            self.fluid.add_dye(
                x,
                y,
                [rgb[0] * amount, rgb[1] * amount, rgb[2] * amount],
                9.0,
            );
        }
    }
}
