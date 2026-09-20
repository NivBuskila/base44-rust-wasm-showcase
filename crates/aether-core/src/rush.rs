//! The lens rush: what a bolt thrown *at the camera* looks like.
//!
//! The simulation is two-dimensional — the fluid is an XY grid and a particle
//! carries a screen position and a screen velocity, nothing else — so there is
//! no direction that means "towards the viewer". A push at the lens therefore
//! cannot be animated like a throw across the screen; it has to be *staged*,
//! and the staging is what this module owns.
//!
//! Two halves, and both are needed for the illusion:
//!
//! 1. **In the field** (here): an expanding shell. Instead of one burst at the
//!    palm, the rush pushes particles and fluid outward from the origin with a
//!    radius that grows over its whole life, which is what the frame of an
//!    object passing the camera looks like — everything sliding off the edges,
//!    faster the nearer it gets.
//! 2. **In the composite** (the renderer, from [`RushState`]): the frame dollies
//!    in on the origin with a radial smear, and a hot core and shell of light
//!    grow through it. That is the part the eye reads as depth; a 2D field
//!    cannot supply it.
//!
//! One rush at a time: a second push restarts it rather than stacking, because
//! two overlapping dollies read as a camera shake, not as two bolts.

use crate::fluid::Fluid;
use crate::math::{hue_to_rgb, smoothstep};
use crate::particles::Particles;

/// Which staging is running. A bolt or palm pushed at the lens is energy
/// *arriving* — a slow swelling approach. A finger-gun shot is a discharge
/// *leaving*: it has to read as a gunshot, so it is a single hard crack at the
/// muzzle, over in a fraction of the time, with the frame kicking back from the
/// recoil instead of dollying in. Same module because both stage a 2D field as
/// depth; different numbers and a different composite branch because they are
/// nothing alike to look at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RushKind {
    /// Energy arriving at the lens: dolly in, cyan shell.
    #[default]
    Push,
    /// A shot fired at the lens: muzzle blast, recoil, amber-white flash.
    Shot,
}

impl RushKind {
    /// How the renderer reads the kind out of the packed state.
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Push => 0.0,
            Self::Shot => 1.0,
        }
    }

    fn duration(self) -> f32 {
        match self {
            Self::Push => DURATION,
            Self::Shot => SHOT_DURATION,
        }
    }
}

/// A gunshot is a crack, not an approach: short enough that the eye reads it as
/// an event rather than a movement, and its blast stays near the muzzle instead
/// of sweeping the whole frame.
const SHOT_DURATION: f32 = 0.24;
const SHOT_SHELL_TO: f32 = 0.42;
/// Muzzle blast: a harder, tighter shove than the rush shell, and the hot amber
/// of burning powder rather than the bolt's cyan.
const SHOT_PUSH: f32 = 520.0;
const SHOT_DRIVE: f32 = 320.0;
const SHOT_HUE: f32 = 0.09;

/// Seconds from the push to the frame settling back down. Long enough to read
/// as an approach rather than a flash, short enough not to hold the view.
const DURATION: f32 = 0.62;

/// Radius of the expanding shell, in normalised units, at the start and at the
/// end of the rush.
const SHELL_FROM: f32 = 0.06;
const SHELL_TO: f32 = 1.25;

/// Outward speed given to particles the shell passes, in grid cells per second
/// at full power, and the fluid drive that carries the dye out with them.
const SHELL_PUSH: f32 = 340.0;
const SHELL_DRIVE: f32 = 210.0;
/// Samples around the shell that inject fluid force each frame. The shell is a
/// ring, and a ring drawn with too few samples reads as a star.
const SHELL_SAMPLES: usize = 24;
/// Dye laid down along the shell, and the hue of the whole effect: the same
/// electric cyan as a thrown bolt, because it is the same energy arriving.
const SHELL_DYE: f32 = 0.5;
const RUSH_HUE: f32 = 0.5;

/// What the renderer needs to stage the rush in the composite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RushState {
    /// Origin in normalised view space, the palm the bolt came from.
    pub at: [f32; 2],
    /// 0 at the push, 1 when it is over. Meaningless while `power` is 0.
    pub progress: f32,
    /// How hard the push was, 0..1, and 0 exactly when no rush is running.
    pub power: f32,
    /// Which staging this is.
    pub kind: RushKind,
}

impl RushState {
    pub const IDLE: Self = Self {
        at: [0.5, 0.5],
        progress: 0.0,
        power: 0.0,
        kind: RushKind::Push,
    };
}

/// The single in-flight rush, if any.
#[derive(Clone, Debug, Default)]
pub struct Rush {
    at: [f32; 2],
    power: f32,
    kind: RushKind,
    /// Seconds since the push; `None` when nothing is running.
    age: Option<f32>,
}

impl Rush {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Starts an energy rush from `at` (normalised) at `power` 0..1.
    pub fn trigger(&mut self, at: [f32; 2], power: f32) {
        self.start(at, power, RushKind::Push);
    }

    /// Starts a gunshot at the lens, fired from `at`.
    pub fn trigger_shot(&mut self, at: [f32; 2], power: f32) {
        self.start(at, power, RushKind::Shot);
    }

    fn start(&mut self, at: [f32; 2], power: f32, kind: RushKind) {
        self.kind = kind;
        let power = if power.is_finite() {
            power.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.at = [fin(at[0]).clamp(0.0, 1.0), fin(at[1]).clamp(0.0, 1.0)];
        // A weak push is still a push: the floor keeps the staging visible
        // rather than letting a barely-triggered rush do nothing at all.
        self.power = power.max(0.35);
        self.age = Some(0.0);
    }

    pub fn state(&self) -> RushState {
        match self.age {
            Some(age) => RushState {
                at: self.at,
                progress: (age / self.kind.duration()).clamp(0.0, 1.0),
                power: self.power,
                kind: self.kind,
            },
            None => RushState::IDLE,
        }
    }

    /// Advances the rush and drives the field for this frame.
    ///
    /// `sx`/`sy` are the grid extents the normalised origin maps onto, the same
    /// convention [`crate::bolt::fire`] takes.
    pub fn step(
        &mut self,
        fluid: &mut Fluid,
        particles: &mut Particles,
        dt: f32,
        sx: f32,
        sy: f32,
    ) {
        let Some(age) = self.age else { return };
        let dt = if dt.is_finite() && dt > 0.0 {
            dt.min(0.05)
        } else {
            1.0 / 60.0
        };
        let age = age + dt;
        let duration = self.kind.duration();
        if age >= duration {
            self.age = None;
            return;
        }
        self.age = Some(age);

        let shot = self.kind == RushKind::Shot;
        let p = age / duration;
        // The shell accelerates: an object approaching the lens covers more of
        // the frame per unit time the closer it gets, so a linear radius would
        // read as a slow ripple. The squared ramp is that acceleration.
        let shell_to = if shot { SHOT_SHELL_TO } else { SHELL_TO };
        let radius = SHELL_FROM + (shell_to - SHELL_FROM) * p * p;
        // ...and it fades as it leaves, so the field is calm by the end.
        let strength = self.power * (1.0 - smoothstep(0.55, 1.0, p));
        if strength <= 0.0 {
            return;
        }

        let (push, drive, hue) = if shot {
            (SHOT_PUSH, SHOT_DRIVE, SHOT_HUE)
        } else {
            (SHELL_PUSH, SHELL_DRIVE, RUSH_HUE)
        };
        let cx = fin(self.at[0]) * sx;
        let cy = fin(self.at[1]) * sy;
        let grid_r = (radius * sx).max(1.0);
        let rgb = hue_to_rgb(hue);

        // Particles: an outward push at the shell and a gentler inward one just
        // behind it, so what sweeps past is a *shell* rather than one shove that
        // empties the middle of the frame.
        let kick = push * strength * dt * 60.0;
        particles.impulse(cx, cy, grid_r, kick, strength);
        particles.impulse(cx, cy, (grid_r * 0.62).max(1.0), -kick * 0.45, 0.0);

        // Fluid: force and dye around the ring, so the dye field opens up with
        // the particles instead of staying a static blob at the palm.
        let dye = SHELL_DYE * strength * dt * 60.0;
        for i in 0..SHELL_SAMPLES {
            let theta = i as f32 / SHELL_SAMPLES as f32 * core::f32::consts::TAU;
            let (ux, uy) = (theta.cos(), theta.sin());
            let x = cx + ux * grid_r;
            let y = cy + uy * grid_r;
            fluid.add_force(
                x,
                y,
                ux * drive * strength,
                uy * drive * strength,
                (grid_r * 0.3).clamp(2.0, 10.0),
            );
            fluid.add_dye(
                x,
                y,
                [rgb[0] * dye, rgb[1] * dye, rgb[2] * dye],
                (grid_r * 0.25).clamp(2.0, 9.0),
            );
        }
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

    const DT: f32 = 1.0 / 60.0;

    fn rig() -> (Fluid, Particles) {
        let fluid = Fluid::new(FLUID_W, FLUID_H);
        let mut particles = Particles::new(4096, 11);
        particles.set_active(4096);
        particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
        (fluid, particles)
    }

    fn run(rush: &mut Rush, fluid: &mut Fluid, particles: &mut Particles, frames: usize) {
        for _ in 0..frames {
            rush.step(
                fluid,
                particles,
                DT,
                (FLUID_W - 1) as f32,
                (FLUID_H - 1) as f32,
            );
        }
    }

    #[test]
    fn an_idle_rush_touches_nothing() {
        let (mut fluid, mut particles) = rig();
        let mut rush = Rush::new();
        run(&mut rush, &mut fluid, &mut particles, 30);
        assert_eq!(rush.state().power, 0.0);
        assert_eq!(fluid.max_speed(), 0.0, "an idle rush drove the field");
    }

    #[test]
    fn a_rush_expands_then_ends() {
        let (mut fluid, mut particles) = rig();
        let mut rush = Rush::new();
        rush.trigger([0.5, 0.5], 1.0);
        assert_eq!(rush.state().progress, 0.0);
        run(&mut rush, &mut fluid, &mut particles, 6);
        let early = rush.state();
        assert!(early.power > 0.0 && early.progress > 0.0 && early.progress < 0.5);
        assert!(fluid.max_speed() > 0.0, "the rush drove no fluid");
        // Well past its duration it is gone, and gone means fully idle rather
        // than a stuck dolly holding the composite warped.
        run(&mut rush, &mut fluid, &mut particles, 60);
        assert_eq!(rush.state(), RushState::IDLE);
    }

    #[test]
    fn a_second_push_restarts_rather_than_stacking() {
        let (mut fluid, mut particles) = rig();
        let mut rush = Rush::new();
        rush.trigger([0.2, 0.3], 1.0);
        run(&mut rush, &mut fluid, &mut particles, 20);
        rush.trigger([0.8, 0.7], 0.5);
        let s = rush.state();
        assert!(s.progress < 0.1, "the restart kept the old age: {s:?}");
        assert_eq!(s.at, [0.8, 0.7]);
    }

    #[test]
    fn garbage_input_is_harmless() {
        let (mut fluid, mut particles) = rig();
        let mut rush = Rush::new();
        rush.trigger([f32::NAN, 2.0], f32::NAN);
        run(&mut rush, &mut fluid, &mut particles, 10);
        assert!(fluid.max_speed().is_finite());
        let s = rush.state();
        assert!(s.at[0].is_finite() && s.at[1] <= 1.0 && s.power > 0.0);
    }
}
