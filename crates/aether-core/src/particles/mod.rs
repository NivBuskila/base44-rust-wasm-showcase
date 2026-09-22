//! Structure-of-arrays particle system advected by the fluid velocity field.
//!
//! SoA rather than array-of-structs so each `step` pass strides linearly over
//! one `Vec<f32>`, which is both cache friendly and auto-vectorisable.
//!
//! ## Contract
//!
//! `step` must, for each of the first `active` particles:
//! - advect by the fluid velocity (RK2 midpoint; plain Euler visibly smears at
//!   the speeds hand gestures produce),
//! - add a curl-driven tangential swirl scaled by `curl_influence`,
//! - age by `dt` and respawn at a fresh random position when `life` runs out or
//!   when the particle lands inside an obstacle or leaves the grid,
//! - keep `heat` as the 0..1 colour-ramp coordinate, decaying toward 0.
//!
//! Positions are in **grid space**; `build_render_buffer` converts to the
//! normalised `[0, 1]` coordinates the vertex shader wants.
//!
//! ## What `vx, vy` mean
//!
//! The stored per-particle velocity is the particle's **own** velocity *on top
//! of* the fluid transport, not its absolute velocity. Transport is
//! `particle_drag * fluid(x) + own`, so a particle that has never been kicked
//! carries `own == 0` and traces a fluid streamline exactly — which is the
//! whole point of drawing 200k of them over a fluid solve. Everything the
//! fluid did not do (gesture impulses, bursts, gravity, curl swirl) lives in
//! `own`, and `own` relaxes toward zero at rate `cfg.damping`; relaxing the
//! residual to zero *is* blending the absolute velocity into the flow, and it
//! is what recaptures a kicked particle instead of letting it coast forever.
//!
//! Doing it the other way round — storing absolute velocity and relaxing it
//! toward the sampled fluid velocity — makes every particle lag the field by
//! the coupling time constant, so a swipe drags a smeared ghost of itself
//! across the screen and fine vortex filaments never form at all.
//!
//! ## Respawn budget
//!
//! `params.spawn_rate` is a *rate*, honoured through an accumulator: a particle
//! whose life ran out only comes back when there is credit for it. Respawning
//! every corpse immediately instead makes the population breathe in lockstep
//! with whatever killed it (a hand sweeping through, a silhouette appearing),
//! and that synchronised pulsing is the tell of a cheap particle system. The
//! bank is capped at one frame's worth so a paused tab cannot buy a burst.

use crate::config::{Params, PARTICLE_STRIDE};
use crate::field::{Grid, VecField};
use crate::math::{decay, length, lerp, normalize, smoothstep};
use crate::par;
use crate::rng::Rng;

/// Per-second decay applied to `heat`. Fast enough that an `Ignite` trail
/// cools within about a second of the hand leaving, slow enough that the
/// colour ramp still reads as a gradient along the trail.
const HEAT_DECAY: f32 = 0.85;

/// Transport speed, in grid cells per second, that reads as fully hot.
///
/// `heat` is the colour-ramp coordinate, and decaying it to zero everywhere
/// meant that the overwhelming majority of particles — the ones the fluid is
/// simply carrying, with no spell on them — sat at the cold end of the ramp
/// and rendered near-black. Screenshots showed beautiful vortex filaments in
/// almost invisible deep blue.
///
/// So heat has a floor derived from how fast the particle is actually moving:
/// the flow colours itself, fast filaments read hot, still regions stay cool,
/// and spells add heat on top of that rather than being the only source of it.
/// In a still field the floor is zero and the decay is unchanged.
const HEAT_SPEED_FULL: f32 = 220.0;

/// Rate, per second, at which `heat` rises toward the speed-derived floor.
/// Slower than the decay so a cooling trail still reads as a gradient rather
/// than snapping to the ambient value.
const HEAT_RISE: f32 = 3.5;

/// Ceiling on a particle's own velocity, in grid cells per second.
///
/// The fluid is clamped elsewhere, but gravity and repeated gesture impulses
/// accumulate here with nothing else to bound them; a particle moving faster
/// than this crosses the whole grid in one frame anyway, so the clamp costs
/// nothing visually and keeps `dt * v` from overflowing to infinity.
const MAX_OWN_SPEED: f32 = 2_000.0;

/// Longest timestep a single `step` integrates. The engine already clamps its
/// `dt`, but `step` is public and must not be made to integrate a tab that was
/// backgrounded for a minute.
const MAX_STEP: f32 = 0.25;

/// Attempts made to find a respawn point outside the obstacle before giving
/// up. A particle respawned inside the silhouette is culled again next frame,
/// which wastes the budget it just spent; three tries make that vanishingly
/// unlikely without putting an unbounded loop in the frame path.
const RESPAWN_TRIES: usize = 3;

/// Multiplicative jitter band applied to a respawned particle's lifetime.
/// Matches [`Particles::seed_uniform`] so the initial pool and every later
/// generation are drawn from the same distribution.
const LIFE_JITTER: (f32, f32) = (0.6, 1.4);

/// Grid-space radius over which [`Particles::spawn_burst`] scatters the
/// particles it emits. A burst from a single exact point reads as one bright
/// dot for its first few frames; a cell or two of spread reads as a puff.
pub const BURST_SCATTER: f32 = 1.5;

/// Per-step tunables that do not belong in the global [`Params`].
#[derive(Clone, Copy, Debug)]
pub struct ParticleConfig {
    /// Tangential swirl from the fluid's curl.
    pub curl_influence: f32,
    /// Constant downward acceleration in grid units per second squared.
    pub gravity: f32,
    /// Fraction of velocity kept per second when outside the fluid's influence.
    pub damping: f32,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            curl_influence: 2.2,
            gravity: 0.0,
            damping: 0.6,
        }
    }
}

/// A fixed-capacity particle pool.
pub struct Particles {
    capacity: usize,
    active: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    life: Vec<f32>,
    max_life: Vec<f32>,
    heat: Vec<f32>,
    render: Vec<f32>,
    rng: Rng,
    /// Round-robin cursor so bursts overwrite the oldest slots.
    cursor: usize,
    alive: usize,
    /// Fractional respawn budget carried between frames, in particles.
    respawn_credit: f32,
    /// When set, the pool is simulated on the GPU: every mutation is recorded
    /// in `ops` for the browser to replay, and `step` does nothing here.
    offloaded: bool,
    ops: OpLog,
}

mod forces;
mod lane;
pub mod offload;
mod render;
mod spawn;
mod step;
#[cfg(test)]
mod tests;

use lane::*;
pub use offload::{OpLog, MAX_OPS, OP_KINDS, OP_STRIDE};

impl Particles {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            capacity,
            active: 0,
            x: vec![0.0; capacity],
            y: vec![0.0; capacity],
            vx: vec![0.0; capacity],
            vy: vec![0.0; capacity],
            life: vec![0.0; capacity],
            max_life: vec![1.0; capacity],
            heat: vec![0.0; capacity],
            render: vec![0.0; capacity * PARTICLE_STRIDE],
            rng: Rng::new(seed),
            cursor: 0,
            alive: 0,
            respawn_credit: 0.0,
            offloaded: false,
            ops: OpLog::new(),
        }
    }

    /// Hands the simulation to the GPU (or takes it back).
    ///
    /// While offloaded, `spawn_*`, `impulse`, `damp` and `grip` append to the op
    /// log instead of touching the streams, `step` and `build_render_buffer`
    /// are no-ops, and `alive` is whatever the browser last reported through
    /// [`Particles::set_alive_hint`]. The streams keep their last CPU state so
    /// switching back mid-session resumes from a sane pool.
    pub fn set_offloaded(&mut self, on: bool) {
        self.offloaded = on;
        self.ops.clear();
    }

    #[inline]
    pub fn offloaded(&self) -> bool {
        self.offloaded
    }

    /// The op log for this frame; the caller clears it once drained.
    #[inline]
    pub fn ops(&self) -> &OpLog {
        &self.ops
    }

    #[inline]
    pub fn clear_ops(&mut self) {
        self.ops.clear();
    }

    /// The GPU's own live count, fed back so the HUD stays truthful.
    pub fn set_alive_hint(&mut self, alive: usize) {
        if self.offloaded {
            self.alive = alive.min(self.active);
        }
    }

    /// Reserves `count` round-robin slots and returns the first, exactly as the
    /// CPU spawners would have taken them. Recorded so the compute shader
    /// overwrites the same particles the CPU would have.
    fn reserve_slots(&mut self, count: usize) -> usize {
        let start = if self.active == 0 {
            0
        } else {
            self.cursor % self.active
        };
        self.cursor = self.cursor.wrapping_add(count);
        start
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn active(&self) -> usize {
        self.active
    }

    /// Number of particles with `life > 0`, recomputed each `step`.
    #[inline]
    pub fn alive(&self) -> usize {
        self.alive
    }

    pub fn set_active(&mut self, n: usize) {
        self.active = n.min(self.capacity);
    }

    // --- Test and spell accessors ---

    #[inline]
    pub fn position(&self, i: usize) -> (f32, f32) {
        (self.x[i], self.y[i])
    }

    #[inline]
    pub fn velocity(&self, i: usize) -> (f32, f32) {
        (self.vx[i], self.vy[i])
    }

    #[inline]
    pub fn life_of(&self, i: usize) -> f32 {
        self.life[i]
    }

    #[inline]
    pub fn heat_of(&self, i: usize) -> f32 {
        self.heat[i]
    }

    /// Places a particle explicitly. Tests use this to build exact scenarios.
    pub fn place(&mut self, i: usize, x: f32, y: f32, vx: f32, vy: f32, life: f32) {
        self.x[i] = x;
        self.y[i] = y;
        self.vx[i] = vx;
        self.vy[i] = vy;
        self.max_life[i] = life.max(1e-4);
        self.life[i] = life;
    }
}
