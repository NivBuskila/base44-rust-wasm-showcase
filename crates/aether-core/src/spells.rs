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

use crate::config::Params;
use crate::field::VecField;
use crate::fluid::Fluid;
use crate::gesture::{GestureTracker, Spell};
use crate::particles::Particles;

/// Hard ceiling on any single velocity impulse, in grid cells per second.
pub const MAX_IMPULSE: f32 = 900.0;

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
#[derive(Clone, Debug, Default)]
pub struct SpellState {
    /// Whether each hand's one-shot spell has already fired this hold.
    fired: [bool; 2],
    /// Previous index-tip position per hand, for continuous trails.
    last_tip: [[f32; 2]; 2],
    has_tip: [bool; 2],
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
///
/// AGENT-IMPLEMENT: the placeholder injects only the optical-flow drive, which
/// is enough for the app to visibly react to motion while the spell layer is
/// being written.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    tracker: &GestureTracker,
    flow: &VecField,
    body_edge: &VecField,
    fluid: &mut Fluid,
    particles: &mut Particles,
    state: &mut SpellState,
    params: &Params,
    dt: f32,
) -> SpellReport {
    let _ = (tracker, body_edge, particles, state);
    let mut report = SpellReport {
        time_scale: 1.0,
        ..Default::default()
    };

    // Model-free drive: push the fluid with measured pixel motion.
    if params.flow_force > 0.0 {
        let gain = params.flow_force * dt * 60.0;
        let (fw, fh) = (flow.w(), flow.h());
        let (gw, gh) = (fluid.width(), fluid.height());
        let sx = (fw - 1) as f32 / (gw - 1) as f32;
        let sy = (fh - 1) as f32 / (gh - 1) as f32;
        let vel = fluid.velocity_mut();
        for y in 0..gh {
            let fy = y as f32 * sy;
            for x in 0..gw {
                let (u, v) = flow.sample(x as f32 * sx, fy);
                let du = (u * gain).clamp(-MAX_IMPULSE, MAX_IMPULSE);
                let dv = (v * gain).clamp(-MAX_IMPULSE, MAX_IMPULSE);
                vel.add(x, y, du, dv);
                report.injected += du * du + dv * dv;
            }
        }
    }

    report
}
