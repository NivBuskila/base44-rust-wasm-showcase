//! Energy shots: a bolt fired *at* the screen rather than cast by a gesture.
//!
//! Every other effect in the engine comes out of perception, which means a
//! caster with no camera (or no hands in frame) can only watch. A shot is the
//! one manual input: the browser hands over a normalised aim point, and a
//! streak of energy travels from just off the near edge of the frame into it and
//! detonates.
//!
//! Shots are queued rather than applied on arrival: a pointer event can land at
//! any moment, and every write to the fluid must happen inside a step, in grid
//! space, with the frame's own `dt`.

use crate::fluid::Fluid;
use crate::math::hue_to_rgb;
use crate::particles::Particles;

/// Most shots kept in the queue. A pointer can fire faster than the render
/// loop consumes them; past this the extra ones are dropped rather than left to
/// detonate a second later, which would feel like lag, not like a volley.
const MAX_QUEUED: usize = 4;

/// Where a bolt is born, normalised: the bottom centre of the frame, a little
/// off-screen so the streak enters rather than appearing mid-air.
const ORIGIN: [f32; 2] = [0.5, 1.06];

/// Samples along the trail. Enough that the streak reads as a line at grid
/// resolution without turning one click into a full-field pass.
const TRAIL_STEPS: usize = 26;
/// Grid-cell radius of the trail's force/dye splat, and its speed and colour.
const TRAIL_RADIUS: f32 = 3.0;
const TRAIL_SPEED: f32 = 240.0;
const TRAIL_DYE: f32 = 0.5;
/// Hue of the bolt: an electric cyan, distinct from every spell tint.
const BOLT_HUE: f32 = 0.5;

/// Impact: an outward puff of particles plus a dye flash.
const IMPACT_RADIUS: f32 = 9.0;
const IMPACT_SPEED: f32 = 300.0;
const IMPACT_DYE: f32 = 2.2;
const IMPACT_FRACTION: usize = 12;
const IMPACT_MAX: usize = 6_000;

/// Queue of aim points waiting for the next step.
#[derive(Clone, Debug, Default)]
pub struct BoltQueue {
    pending: [[f32; 2]; MAX_QUEUED],
    len: usize,
}

impl BoltQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a shot at a normalised `[0, 1]` point. Out-of-range or
    /// non-finite aim is ignored rather than clamped onto an edge.
    pub fn push(&mut self, x: f32, y: f32) {
        if !x.is_finite() || !y.is_finite() || self.len >= MAX_QUEUED {
            return;
        }
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return;
        }
        self.pending[self.len] = [x, y];
        self.len += 1;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Fires and drains everything queued. Returns how many bolts landed.
    pub fn flush(
        &mut self,
        fluid: &mut Fluid,
        particles: &mut Particles,
        particle_life: f32,
        sx: f32,
        sy: f32,
    ) -> u32 {
        let fired = self.len as u32;
        for i in 0..self.len {
            let aim = self.pending[i];
            fire(fluid, particles, particle_life, aim, sx, sy);
        }
        self.len = 0;
        fired
    }
}

/// Draws one bolt's trail and its impact, all in grid space.
fn fire(
    fluid: &mut Fluid,
    particles: &mut Particles,
    particle_life: f32,
    aim: [f32; 2],
    sx: f32,
    sy: f32,
) {
    let from = [ORIGIN[0] * sx, ORIGIN[1] * sy];
    let to = [aim[0] * sx, aim[1] * sy];
    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
    let len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() || len < 1e-3 {
        return;
    }
    let (nx, ny) = (dx / len, dy / len);
    let rgb = hue_to_rgb(BOLT_HUE);

    // The trail brightens and pushes harder towards the impact, so the eye
    // follows it into the hit instead of reading a uniform stripe.
    for i in 0..TRAIL_STEPS {
        let t = (i as f32 + 1.0) / TRAIL_STEPS as f32;
        let x = from[0] + dx * t;
        let y = from[1] + dy * t;
        let w = t * t;
        fluid.add_force(x, y, nx * TRAIL_SPEED * w, ny * TRAIL_SPEED * w, TRAIL_RADIUS);
        let a = TRAIL_DYE * w;
        fluid.add_dye(x, y, [rgb[0] * a, rgb[1] * a, rgb[2] * a], TRAIL_RADIUS);
    }

    let count = (particles.active() / IMPACT_FRACTION).min(IMPACT_MAX);
    particles.spawn_burst(to[0], to[1], count, IMPACT_SPEED, 1.0, particle_life);
    particles.impulse(to[0], to[1], IMPACT_RADIUS * 1.5, 0.0, 1.0);
    fluid.add_dye(
        to[0],
        to[1],
        [
            rgb[0] * IMPACT_DYE,
            rgb[1] * IMPACT_DYE,
            rgb[2] * IMPACT_DYE,
        ],
        IMPACT_RADIUS,
    );
    // An event, not a force: a one-off outward push, deliberately un-scaled by
    // dt so a shot hits as hard at 30 fps as at 144.
    let n = 20;
    for i in 0..n {
        let theta = i as f32 / n as f32 * core::f32::consts::TAU;
        let (cx, cy) = (theta.cos(), theta.sin());
        fluid.add_force(
            to[0] + cx * IMPACT_RADIUS * 0.5,
            to[1] + cy * IMPACT_RADIUS * 0.5,
            cx * IMPACT_SPEED,
            cy * IMPACT_SPEED,
            IMPACT_RADIUS * 0.5,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUID_H, FLUID_W};

    fn rig() -> (Fluid, Particles) {
        let mut particles = Particles::new(4096, 7);
        particles.set_active(4096);
        particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
        (Fluid::new(FLUID_W, FLUID_H), particles)
    }

    #[test]
    fn a_shot_lights_the_field_and_kicks_it() {
        let (mut fluid, mut particles) = rig();
        let mut q = BoltQueue::new();
        q.push(0.5, 0.3);
        let fired = q.flush(
            &mut fluid,
            &mut particles,
            2.0,
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        assert_eq!(fired, 1);
        assert!(fluid.max_speed() > 0.0, "shot injected no velocity");
        assert!(fluid.dye()[2].data.iter().any(|&v| v > 0.01), "shot left no dye");
    }

    #[test]
    fn garbage_and_offscreen_aim_is_ignored_and_the_queue_is_bounded() {
        let mut q = BoltQueue::new();
        q.push(f32::NAN, 0.5);
        q.push(1.5, 0.5);
        q.push(0.5, -0.2);
        assert_eq!(q.len, 0);
        for _ in 0..(MAX_QUEUED + 5) {
            q.push(0.5, 0.5);
        }
        assert_eq!(q.len, MAX_QUEUED);
    }

    #[test]
    fn flush_drains_the_queue() {
        let (mut fluid, mut particles) = rig();
        let mut q = BoltQueue::new();
        q.push(0.4, 0.4);
        let (sx, sy) = ((FLUID_W - 1) as f32, (FLUID_H - 1) as f32);
        assert_eq!(q.flush(&mut fluid, &mut particles, 2.0, sx, sy), 1);
        assert_eq!(q.flush(&mut fluid, &mut particles, 2.0, sx, sy), 0);
    }
}
