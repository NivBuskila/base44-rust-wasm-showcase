//! Energy shots thrown *from the hand*: a fast flick of an open hand hurls a
//! bolt of energy across the screen in the direction it was thrown.
//!
//! Unlike a [`crate::combo`] (one hand's shapes in order) or a [`crate::duet`]
//! (the relationship between the hands), a throw is pure *motion*: the shape
//! only decides whether the hand is allowed to throw at all. That makes it the
//! one spell a caster discovers without being told — you flick, something flies.
//!
//! Speeds here are in **hand widths per second** (palm velocity over the hand
//! scale), so a throw feels the same near and far from the camera.
//!
//! A throw *at the camera* has no on-screen motion to read, so it is read from
//! the hand growing instead: a fast push towards the lens makes the hand swell
//! in the frame, and that growth rate is the throw. Such a bolt has nowhere to
//! fly across the screen, so it detonates at the palm, towards the viewer.
//!
//! The effect is applied here rather than in [`crate::spells`] because a bolt is
//! self-contained: it writes a streak and an impact and nothing else in the
//! engine needs to know about it.

use crate::config::HANDS;
use crate::fluid::Fluid;
use crate::gesture::GestureTracker;
use crate::gun::Shot;
use crate::math::{hue_to_rgb, smoothstep};
use crate::particles::Particles;

/// A hand must be moving at least this fast (hand widths per second) to throw,
/// and `FULL_SPEED` is a full-power throw. It re-arms only once the hand has
/// slowed past `REARM_SPEED`, so one flick is one bolt rather than a stream for
/// as long as the hand keeps moving.
const THROW_SPEED: f32 = 5.5;
const FULL_SPEED: f32 = 16.0;
const REARM_SPEED: f32 = 2.0;

/// The hand must be at least this open. A fist is excluded so squeezing for the
/// big bang, or dragging an attract around, never throws.
const MIN_OPENNESS: f32 = 0.45;

/// Floor on the hand scale used to normalise speeds; a degenerate scale would
/// read every twitch as a throw.
const MIN_SCALE: f32 = 0.02;

/// A push towards the camera: the hand must grow by at least this fraction of
/// itself per second (`FULL_GROWTH` is full power) and re-arms when the growth
/// falls back under `REARM_GROWTH`. The rate is smoothed over `GROWTH_HALF_LIFE`
/// seconds because MediaPipe's hand size jitters frame to frame.
const THROW_GROWTH: f32 = 1.6;
const FULL_GROWTH: f32 = 4.0;
const REARM_GROWTH: f32 = 0.5;
const GROWTH_HALF_LIFE: f32 = 0.04;

/// How far a bolt flies, in normalised units, at zero and at full power.
const REACH: f32 = 0.9;

/// Samples along the trail. Enough that the streak reads as a line at grid
/// resolution without turning one throw into a full-field pass.
const TRAIL_STEPS: usize = 26;
/// Grid-cell radius of the trail's force/dye splat, its drive and its colour.
const TRAIL_RADIUS: f32 = 3.0;
const TRAIL_SPEED: f32 = 260.0;
const TRAIL_DYE: f32 = 0.55;
/// Hue of the bolt: an electric cyan, distinct from every spell tint.
const BOLT_HUE: f32 = 0.5;

/// A finger-gun shot: thinner, hotter and faster than a thrown bolt, and it
/// always flies clean off the screen rather than landing at a reach.
const SHOT_RADIUS: f32 = 1.6;
const SHOT_SPEED: f32 = 420.0;
const SHOT_DYE: f32 = 0.9;
const SHOT_STEPS: usize = 40;
/// A white-hot amber, so a gunshot never reads as a small throw.
const SHOT_HUE: f32 = 0.09;

/// Impact at the far end: an outward puff of particles plus a dye flash.
const IMPACT_RADIUS: f32 = 9.0;
const IMPACT_SPEED: f32 = 320.0;
const IMPACT_DYE: f32 = 2.2;
const IMPACT_FRACTION: usize = 12;
const IMPACT_MAX: usize = 6_000;

/// A bolt thrown this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Throw {
    /// Palm the bolt leaves, normalised `[0, 1]`.
    pub from: [f32; 2],
    /// Unit direction it was thrown in across the screen, or `[0, 0]` for a
    /// throw at the camera, which detonates at the palm instead of flying.
    pub dir: [f32; 2],
    /// 0..1, from how hard the flick was.
    pub power: f32,
}

/// Cross-frame state, per hand: arming flags plus the hand-size filter that
/// reads a push towards the camera.
#[derive(Clone, Debug)]
pub struct ThrowTracker {
    armed: [bool; HANDS],
    /// Last frame's hand scale, `None` until the hand has been seen once.
    last_scale: [Option<f32>; HANDS],
    /// Smoothed relative growth of the hand scale, per second.
    growth: [f32; HANDS],
}

impl Default for ThrowTracker {
    fn default() -> Self {
        Self {
            armed: [true; HANDS],
            last_scale: [None; HANDS],
            growth: [0.0; HANDS],
        }
    }
}

impl ThrowTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns whichever hands threw this step. Both may throw at once.
    pub fn update(&mut self, tracker: &GestureTracker, dt: f32) -> [Option<Throw>; HANDS] {
        let mut thrown: [Option<Throw>; HANDS] = [None; HANDS];
        let dt = if dt.is_finite() && dt > 0.0 { dt } else { 1.0 / 60.0 };
        for (slot, hand) in tracker.hands().iter().enumerate().take(HANDS) {
            if !hand.present {
                // A hand leaving the frame is not a throw, but the next one to
                // appear must be able to throw straight away.
                self.armed[slot] = true;
                self.last_scale[slot] = None;
                self.growth[slot] = 0.0;
                continue;
            }
            let scale = fin(hand.scale).max(MIN_SCALE);

            // Across the screen: palm speed.
            let (vx, vy) = (fin(hand.velocity[0]), fin(hand.velocity[1]));
            let speed = (vx * vx + vy * vy).sqrt() / scale;

            // Towards the camera: how fast the hand is growing in the frame.
            let raw = match self.last_scale[slot] {
                Some(prev) => (scale - prev) / (prev.max(MIN_SCALE) * dt),
                None => 0.0,
            };
            self.last_scale[slot] = Some(scale);
            let k = 1.0 - (0.5f32).powf(dt / GROWTH_HALF_LIFE);
            self.growth[slot] += (raw - self.growth[slot]) * k;
            let growth = self.growth[slot];

            if speed < REARM_SPEED && growth < REARM_GROWTH {
                self.armed[slot] = true;
            }
            if !self.armed[slot] || fin(hand.openness) < MIN_OPENNESS {
                continue;
            }
            let across = smoothstep(THROW_SPEED, FULL_SPEED, speed);
            let towards = smoothstep(THROW_GROWTH, FULL_GROWTH, growth);
            if speed < THROW_SPEED && growth < THROW_GROWTH {
                continue;
            }
            self.armed[slot] = false;
            // Whichever reading is the stronger throw decides the kind: a push
            // at the lens always drifts a little on screen too, and that drift
            // must not turn it into a sideways bolt.
            thrown[slot] = Some(if towards >= across {
                Throw {
                    from: hand.palm,
                    dir: [0.0, 0.0],
                    power: towards,
                }
            } else {
                let unit = speed * scale;
                Throw {
                    from: hand.palm,
                    dir: [vx / unit, vy / unit],
                    power: across,
                }
            });
        }
        thrown
    }
}

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

/// Draws one finger-gun shot: a tight tracer from the fingertip to the edge of
/// the field and a small hard impact where it leaves.
pub fn shoot(fluid: &mut Fluid, particles: &mut Particles, particle_life: f32, shot: Shot, sx: f32, sy: f32) {
    let (nx, ny) = (fin(shot.dir[0]), fin(shot.dir[1]));
    if nx == 0.0 && ny == 0.0 {
        return;
    }
    let from = [
        (fin(shot.from[0]) * sx).clamp(0.0, sx),
        (fin(shot.from[1]) * sy).clamp(0.0, sy),
    ];
    // Distance to the first field edge along the aim; the tracer stops there.
    let edge = |p: f32, d: f32, max: f32| -> f32 {
        if d > 0.0 {
            (max - p) / d
        } else if d < 0.0 {
            -p / d
        } else {
            f32::INFINITY
        }
    };
    let t = edge(from[0], nx * sx, 1.0).min(edge(from[1], ny * sy, 1.0)).min(1.0);
    let to = [
        (from[0] + nx * sx * t).clamp(0.0, sx),
        (from[1] + ny * sy * t).clamp(0.0, sy),
    ];
    let rgb = hue_to_rgb(SHOT_HUE);
    // Uniform along its length: a tracer is a line, not a comet.
    for i in 0..SHOT_STEPS {
        let k = (i as f32 + 0.5) / SHOT_STEPS as f32;
        let x = from[0] + (to[0] - from[0]) * k;
        let y = from[1] + (to[1] - from[1]) * k;
        fluid.add_force(x, y, nx * SHOT_SPEED, ny * SHOT_SPEED, SHOT_RADIUS);
        fluid.add_dye(
            x,
            y,
            [rgb[0] * SHOT_DYE, rgb[1] * SHOT_DYE, rgb[2] * SHOT_DYE],
            SHOT_RADIUS,
        );
    }
    // Muzzle flash at the fingertip, hit at the edge.
    fluid.add_dye(
        from[0],
        from[1],
        [rgb[0] * IMPACT_DYE, rgb[1] * IMPACT_DYE, rgb[2] * IMPACT_DYE],
        IMPACT_RADIUS * 0.5,
    );
    impact(fluid, particles, particle_life, to, 1.0, 0.6);
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
    let count = ((particles.active() / IMPACT_FRACTION) as f32 * (0.5 + 0.5 * power) * size) as usize;
    particles.spawn_burst(at[0], at[1], count.min(IMPACT_MAX), burst, 1.0, particle_life);
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

#[inline]
fn fin(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUID_H, FLUID_W};
    use crate::gesture::{synth, GestureConfig};

    const DT: f32 = 1.0 / 60.0;
    const SCALE: f32 = 0.1;

    struct Rig {
        gestures: GestureTracker,
        throws: ThrowTracker,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                gestures: GestureTracker::new(GestureConfig::default()),
                throws: ThrowTracker::new(),
            }
        }

        /// One hand at `x`, stepped through the tracker and the recogniser.
        fn frame(&mut self, gesture: i32, x: f32) -> [Option<Throw>; HANDS] {
            self.frame_scaled(gesture, x, SCALE)
        }

        fn frame_scaled(&mut self, gesture: i32, x: f32, scale: f32) -> [Option<Throw>; HANDS] {
            let mut buf = synth::buffer();
            let hand = synth::Hand::at(x, 0.5).gesture(gesture).scaled(scale);
            synth::write(&mut buf, 0, &hand);
            self.gestures.update_hands(&buf, DT);
            self.gestures.update_two_hand(DT);
            self.throws.update(&self.gestures, DT)
        }

        /// The hand grows in place from `SCALE` to `to` over `frames`: a push
        /// straight at the lens.
        fn push(&mut self, gesture: i32, to: f32, frames: usize) -> Option<Throw> {
            let mut thrown = None;
            for i in 0..frames {
                let s = SCALE + (to - SCALE) * (i as f32 + 1.0) / frames as f32;
                thrown = thrown.or(self.frame_scaled(gesture, 0.5, s)[0]);
            }
            thrown
        }

        /// Holds the hand still, so the gesture latches and the speed decays.
        fn settle(&mut self, gesture: i32, x: f32) {
            for _ in 0..40 {
                self.frame(gesture, x);
            }
        }

        /// One fast sweep from `from` to `to`, returning any throw.
        fn flick(&mut self, gesture: i32, from: f32, to: f32, frames: usize) -> Option<Throw> {
            let mut thrown = None;
            for i in 0..frames {
                let x = from + (to - from) * (i as f32 + 1.0) / frames as f32;
                thrown = thrown.or(self.frame(gesture, x)[0]);
            }
            thrown
        }
    }

    #[test]
    fn a_flick_of_an_open_hand_throws_along_the_motion() {
        let mut rig = Rig::new();
        rig.settle(synth::OPEN_PALM, 0.25);
        let throw = rig
            .flick(synth::OPEN_PALM, 0.25, 0.75, 8)
            .expect("a fast open-hand flick did not throw");
        assert!(
            throw.dir[0] > 0.8,
            "bolt did not fly along the flick: {:?}",
            throw.dir
        );
        assert!(throw.power > 0.0);
    }

    #[test]
    fn a_slow_hand_and_a_fist_never_throw() {
        let mut rig = Rig::new();
        rig.settle(synth::OPEN_PALM, 0.4);
        assert!(
            rig.flick(synth::OPEN_PALM, 0.4, 0.46, 40).is_none(),
            "a slow drift threw a bolt"
        );
        let mut fist = Rig::new();
        fist.settle(synth::CLOSED_FIST, 0.25);
        assert!(
            fist.flick(synth::CLOSED_FIST, 0.25, 0.75, 8).is_none(),
            "a fist threw a bolt"
        );
    }

    #[test]
    fn one_flick_throws_once_until_the_hand_slows_down() {
        let mut rig = Rig::new();
        rig.settle(synth::OPEN_PALM, 0.2);
        assert!(rig.flick(synth::OPEN_PALM, 0.1, 0.5, 6).is_some());
        // Still sweeping the same way: one flick is one bolt. (Reversing *is* a
        // new throw — the hand has to stop to turn around.)
        assert!(
            rig.flick(synth::OPEN_PALM, 0.5, 0.9, 6).is_none(),
            "a continuous sweep machine-gunned bolts"
        );
        rig.settle(synth::OPEN_PALM, 0.2);
        assert!(
            rig.flick(synth::OPEN_PALM, 0.2, 0.8, 8).is_some(),
            "the throw never re-armed"
        );
    }

    #[test]
    fn a_push_at_the_camera_throws_at_the_viewer() {
        let mut rig = Rig::new();
        rig.settle(synth::OPEN_PALM, 0.5);
        let throw = rig
            .push(synth::OPEN_PALM, SCALE * 1.8, 10)
            .expect("a fast push at the camera did not throw");
        assert_eq!(throw.dir, [0.0, 0.0], "a push at the lens flew sideways");
        assert!(throw.power > 0.0);
        // A slow approach is just moving closer, not a throw.
        let mut slow = Rig::new();
        slow.settle(synth::OPEN_PALM, 0.5);
        assert!(
            slow.push(synth::OPEN_PALM, SCALE * 1.3, 90).is_none(),
            "leaning in threw a bolt"
        );
    }

    #[test]
    fn a_bolt_at_the_viewer_detonates_at_the_palm() {
        let mut fluid = Fluid::new(FLUID_W, FLUID_H);
        let mut particles = Particles::new(4096, 7);
        particles.set_active(4096);
        particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
        fire(
            &mut fluid,
            &mut particles,
            2.0,
            Throw {
                from: [0.5, 0.5],
                dir: [0.0, 0.0],
                power: 1.0,
            },
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        assert!(fluid.max_speed() > 0.0, "the bolt injected no velocity");
    }

    #[test]
    fn a_thrown_bolt_lights_the_field_and_kicks_it() {
        let mut fluid = Fluid::new(FLUID_W, FLUID_H);
        let mut particles = Particles::new(4096, 7);
        particles.set_active(4096);
        particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
        fire(
            &mut fluid,
            &mut particles,
            2.0,
            Throw {
                from: [0.3, 0.5],
                dir: [1.0, 0.0],
                power: 1.0,
            },
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        assert!(fluid.max_speed() > 0.0, "the bolt injected no velocity");
        assert!(
            fluid.dye()[2].data.iter().any(|&v| v > 0.01),
            "the bolt left no dye"
        );
    }

    #[test]
    fn a_gunshot_traces_to_the_edge_of_the_field() {
        let mut fluid = Fluid::new(FLUID_W, FLUID_H);
        let mut particles = Particles::new(4096, 7);
        particles.set_active(4096);
        particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
        let (sx, sy) = ((FLUID_W - 1) as f32, (FLUID_H - 1) as f32);
        shoot(
            &mut fluid,
            &mut particles,
            2.0,
            Shot {
                from: [0.5, 0.9],
                dir: [0.0, -1.0],
            },
            sx,
            sy,
        );
        assert!(fluid.max_speed() > 0.0, "the shot injected no velocity");
        // Dye must reach the top of the field (row 0 is the boundary and stays
        // clear), i.e. the tracer went all the way.
        let red = &fluid.dye()[0];
        let top: f32 = (0..FLUID_W)
            .map(|x| red.data[FLUID_W + x])
            .fold(0.0, f32::max);
        assert!(top > 0.01, "the tracer stopped short of the edge");
    }

    #[test]
    fn garbage_throw_data_is_harmless() {
        let mut fluid = Fluid::new(FLUID_W, FLUID_H);
        let mut particles = Particles::new(256, 3);
        particles.set_active(256);
        fire(
            &mut fluid,
            &mut particles,
            2.0,
            Throw {
                from: [f32::NAN, 0.5],
                dir: [f32::NAN, f32::NAN],
                power: f32::NAN,
            },
            (FLUID_W - 1) as f32,
            (FLUID_H - 1) as f32,
        );
        assert!(fluid.max_speed().is_finite());
    }
}
