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

mod fire;
#[cfg(test)]
mod tests;

pub use fire::fire;

use crate::config::HANDS;
use crate::gesture::GestureTracker;
use crate::math::smoothstep;

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
/// The bar is high on purpose: a hand simply moving around the frame changes its
/// apparent depth all the time, and a low threshold read most of that as a push
/// at the lens.
const THROW_GROWTH: f32 = 2.0;
const FULL_GROWTH: f32 = 3.6;
const REARM_GROWTH: f32 = 0.6;
const GROWTH_HALF_LIFE: f32 = 0.05;

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
        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            1.0 / 60.0
        };
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
            // Whichever reading is the stronger throw decides the kind. A
            // sideways flick also grows the hand a little (the arm extends as it
            // swings), so the push must actually beat the drift to win —
            // favouring it turned every throw into a push at the lens.
            // Sideways motion always wins: a push at the lens is only a push
            // when the hand is *not* travelling across the frame, which is the
            // one reading a wandering hand cannot fake.
            thrown[slot] = Some(if growth >= THROW_GROWTH && speed < THROW_SPEED {
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

#[inline]
fn fin(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}
