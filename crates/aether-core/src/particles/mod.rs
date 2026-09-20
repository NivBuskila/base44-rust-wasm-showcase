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

mod lane;
pub mod offload;
#[cfg(test)]
mod tests;

use lane::*;
pub use offload::{OpLog, MAX_OPS, OP_STRIDE};

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

    /// Scatters every active particle uniformly over a `w * h` grid with a
    /// randomised life phase, so the pool does not die in lockstep.
    pub fn seed_uniform(&mut self, w: usize, h: usize, life: f32) {
        for i in 0..self.active {
            self.x[i] = self.rng.range(0.0, (w - 1) as f32);
            self.y[i] = self.rng.range(0.0, (h - 1) as f32);
            self.vx[i] = 0.0;
            self.vy[i] = 0.0;
            self.max_life[i] = life * self.rng.range(0.6, 1.4);
            self.life[i] = self.max_life[i] * self.rng.next_f32();
            self.heat[i] = 0.0;
        }
        self.alive = self.active;
    }

    /// Emits `count` particles from `(x, y)` in grid space with radial speed
    /// `speed` and the given colour-ramp coordinate.
    ///
    /// Direction comes from [`Rng::in_disc`], so the sampled offsets are
    /// area-uniform rather than centre-heavy, and each particle is placed along
    /// its own velocity: the burst starts already slightly expanded, and
    /// "outward" is true of every particle rather than only on average.
    pub fn spawn_burst(&mut self, x: f32, y: f32, count: usize, speed: f32, heat: f32, life: f32) {
        if self.active == 0 || !x.is_finite() || !y.is_finite() {
            return;
        }
        let speed = finite_or(speed, 0.0).clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED);
        let heat = finite_or(heat, 0.0).clamp(0.0, 1.0);
        let base = finite_or(life, 1.0).max(0.05);
        let count = count.min(self.active);

        if self.offloaded {
            let slot = self.reserve_slots(count) as f32;
            self.ops.push(
                offload::OP_BURST,
                &[x, y, count as f32, speed, heat, base, slot],
            );
            return;
        }

        for _ in 0..count {
            let i = self.cursor % self.active;
            self.cursor = self.cursor.wrapping_add(1);
            let (dx, dy) = self.rng.in_disc();
            self.x[i] = x + dx * BURST_SCATTER;
            self.y[i] = y + dy * BURST_SCATTER;
            self.vx[i] = dx * speed;
            self.vy[i] = dy * speed;
            // Jittered even here: a burst is the most likely thing to put a few
            // thousand particles on exactly the same clock.
            self.max_life[i] = base * self.rng.range(LIFE_JITTER.0, LIFE_JITTER.1);
            self.life[i] = self.max_life[i];
            self.heat[i] = heat;
        }
    }

    /// Emits `count` particles on a circle of radius `ring` around `(x, y)`,
    /// each flying straight outward at `speed`: an expanding shell rather than
    /// a filled puff. Uniform in angle, so the ring has no bright spokes.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ring(
        &mut self,
        x: f32,
        y: f32,
        count: usize,
        ring: f32,
        speed: f32,
        heat: f32,
        life: f32,
    ) {
        if self.active == 0 || !x.is_finite() || !y.is_finite() || !ring.is_finite() {
            return;
        }
        let speed = finite_or(speed, 0.0).clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED);
        let heat = finite_or(heat, 0.0).clamp(0.0, 1.0);
        let base = finite_or(life, 1.0).max(0.05);
        let ring = ring.max(0.0);
        let count = count.min(self.active);

        if self.offloaded {
            let slot = self.reserve_slots(count) as f32;
            self.ops.push(
                offload::OP_RING,
                &[x, y, count as f32, ring, speed, heat, base, slot],
            );
            return;
        }

        for _ in 0..count {
            let i = self.cursor % self.active;
            self.cursor = self.cursor.wrapping_add(1);
            let theta = self.rng.next_f32() * core::f32::consts::TAU;
            let (dx, dy) = (theta.cos(), theta.sin());
            let (jx, jy) = self.rng.in_disc();
            self.x[i] = x + dx * ring + jx * BURST_SCATTER;
            self.y[i] = y + dy * ring + jy * BURST_SCATTER;
            self.vx[i] = dx * speed;
            self.vy[i] = dy * speed;
            self.max_life[i] = base * self.rng.range(LIFE_JITTER.0, LIFE_JITTER.1);
            self.life[i] = self.max_life[i];
            self.heat[i] = heat;
        }
    }

    /// Advances every active particle.
    ///
    /// One pass, no allocation, no per-particle branching beyond the cull test.
    /// Position comes from an RK2 midpoint of the fluid field: the first-order
    /// backtrace cuts corners exactly where a gesture is most visible — a fast
    /// curved swipe — and the second sample is cheaper than the smaller `dt` it
    /// would take to hide that.
    pub fn step(
        &mut self,
        vel: &VecField,
        obstacle: &Grid,
        dt: f32,
        params: &Params,
        cfg: &ParticleConfig,
    ) {
        // On the GPU the browser advances the pool; the log it replays was
        // filled by the spells that ran before this call.
        if self.offloaded {
            return;
        }
        let (gw, gh) = (vel.w(), vel.h());
        let cells = gw.saturating_mul(gh);
        // Defensive: `VecField::new` cannot produce a grid this small, but the
        // fields are public and `step` must not index out of a hand-built one.
        if self.active == 0
            || gw < 2
            || gh < 2
            || vel.u.data.len() < cells
            || vel.v.data.len() < cells
        {
            self.alive = 0;
            return;
        }

        let dt = finite_or(dt, 0.0).clamp(0.0, MAX_STEP);
        let maxx = (gw - 1) as f32;
        let maxy = (gh - 1) as f32;

        let usable_obstacle = obstacle.w >= 1
            && obstacle.h >= 1
            && obstacle.data.len() >= obstacle.w * obstacle.h
            && obstacle.max_abs() >= 0.5;
        let frame = Frame {
            vel,
            obstacle: if usable_obstacle {
                Some(obstacle)
            } else {
                None
            },
            osx: (obstacle.w.max(1) - 1) as f32 / maxx,
            osy: (obstacle.h.max(1) - 1) as f32 / maxy,
            maxx,
            maxy,
            life: finite_or(params.particle_life, 1.0).clamp(0.05, 60.0),
            dt,
            drag: finite_or(params.particle_drag, 1.0).clamp(0.0, 8.0),
            // `decay`, never `1 - rate * dt`: the latter flips sign at large dt
            // and makes the 30 fps session look different from the 144 fps one.
            keep: decay(finite_or(cfg.damping, 0.0).max(0.0), dt),
            heat_keep: decay(HEAT_DECAY, dt),
            heat_rise: 1.0 - decay(HEAT_RISE, dt),
            inv_dt: 1.0 / dt,
            swirl: finite_or(cfg.curl_influence, 0.0) * dt,
            gravity: finite_or(cfg.gravity, 0.0) * dt,
        };

        let rate = finite_or(params.spawn_rate, 0.0).max(0.0);
        let per_frame = rate * dt;
        let credit = (self.respawn_credit + per_frame).min(per_frame + 1.0);

        // The pool is cut into lanes — contiguous bands of every SoA stream —
        // and each lane is advanced independently, on its own thread when the
        // `parallel` feature is on. Nothing is shared between lanes: each gets
        // its own RNG stream seeded from the pool's, and an equal slice of the
        // respawn budget. With one thread this is exactly the old single pass.
        let lane_len = par::chunk_len(self.active, LANE_MIN);
        let mut lanes = self.lanes(lane_len, credit);
        let alive = par::sum_mut(&mut lanes, |lane| lane.run(&frame));

        self.respawn_credit = lanes.iter().map(|l| l.credit).sum();
        self.alive = alive;
    }

    /// Splits the first `active` particles into lanes of `len`, handing each
    /// an equal share of `credit` and a fresh RNG stream.
    fn lanes(&mut self, len: usize, credit: f32) -> Vec<Lane<'_>> {
        let n = self.active;
        let count = n.div_ceil(len.max(1)).max(1);
        let share = credit / count as f32;
        let Self {
            x,
            y,
            vx,
            vy,
            life,
            max_life,
            heat,
            rng,
            ..
        } = self;
        let streams = x[..n]
            .chunks_mut(len)
            .zip(y[..n].chunks_mut(len))
            .zip(vx[..n].chunks_mut(len))
            .zip(vy[..n].chunks_mut(len))
            .zip(life[..n].chunks_mut(len))
            .zip(max_life[..n].chunks_mut(len))
            .zip(heat[..n].chunks_mut(len));
        let mut out = Vec::with_capacity(count);
        for ((((((x, y), vx), vy), life), max_life), heat) in streams {
            out.push(Lane {
                x,
                y,
                vx,
                vy,
                life,
                max_life,
                heat,
                rng: Rng::new(rng.next_u64()),
                credit: share,
            });
        }
        out
    }
    /// Packs `x, y, heat, life` per particle, with position normalised to
    /// `[0, 1]` against the grid the simulation runs on and `life` normalised
    /// against that particle's own lifetime.
    pub fn build_render_buffer(&mut self, grid_w: usize, grid_h: usize) {
        // The GPU pool draws straight from its own storage buffer.
        if self.offloaded {
            return;
        }
        let sx = 1.0 / (grid_w - 1) as f32;
        let sy = 1.0 / (grid_h - 1) as f32;
        let n = self.active;
        let lane = par::chunk_len(n, LANE_MIN);
        let (x, y, heat, life, max_life) = (
            &self.x[..n],
            &self.y[..n],
            &self.heat[..n],
            &self.life[..n],
            &self.max_life[..n],
        );
        par::chunks_mut(
            &mut self.render[..n * PARTICLE_STRIDE],
            lane * PARTICLE_STRIDE,
            |c, out| {
                let base = c * lane;
                for (k, px) in out.chunks_exact_mut(PARTICLE_STRIDE).enumerate() {
                    let i = base + k;
                    let norm_life = if max_life[i] > 0.0 {
                        (life[i] / max_life[i]).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    px[0] = x[i] * sx;
                    px[1] = y[i] * sy;
                    px[2] = heat[i];
                    px[3] = norm_life;
                }
            },
        );
    }

    /// The interleaved buffer, truncated to the active particle count.
    #[inline]
    pub fn render_buffer(&self) -> &[f32] {
        &self.render[..self.active * PARTICLE_STRIDE]
    }

    /// Raw pointer to the render buffer, for zero-copy access from JS.
    #[inline]
    pub fn render_ptr(&self) -> *const f32 {
        self.render.as_ptr()
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

    /// Adds a radial impulse of `strength` grid cells per second to every live
    /// particle within `radius` of `(cx, cy)`, easing to zero at the edge.
    ///
    /// The falloff is a smoothstep rather than a hard cutoff: a constant push
    /// truncated at `radius` leaves a sharp ring of particles moving at full
    /// speed next to stationary ones, and that ring is visible as a circle
    /// every single time the spell fires.
    pub fn impulse(&mut self, cx: f32, cy: f32, radius: f32, strength: f32, heat: f32) {
        if self.active == 0
            || !cx.is_finite()
            || !cy.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
        {
            return;
        }
        let strength = finite_or(strength, 0.0).clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED);
        let heat = finite_or(heat, 0.0).clamp(0.0, 1.0);
        if self.offloaded {
            self.ops
                .push(offload::OP_IMPULSE, &[cx, cy, radius, strength, heat]);
            return;
        }
        let r2 = radius * radius;

        for i in 0..self.active {
            if self.life[i] <= 0.0 {
                continue;
            }
            let dx = self.x[i] - cx;
            let dy = self.y[i] - cy;
            let d2 = dx * dx + dy * dy;
            // The `is_nan` arm matters: `place` can park a particle at a NaN
            // position, and `d2 > r2` is false for NaN, which would let it
            // through into `normalize` and spread the NaN into its velocity.
            if d2 > r2 || d2.is_nan() {
                continue;
            }
            // 1 at the centre, 0 with zero slope at the edge.
            let falloff = smoothstep(radius, 0.0, d2.sqrt());
            let (nx, ny) = normalize(dx, dy);
            self.vx[i] = tame(self.vx[i] + nx * strength * falloff);
            self.vy[i] = tame(self.vy[i] + ny * strength * falloff);
            // `max`, not `+=`: spells hold for many frames, and accumulating
            // would saturate every particle in range to full heat.
            self.heat[i] = self.heat[i].max(heat * falloff);
        }
    }

    /// Bleeds the own-velocity of every particle within `radius` of
    /// `(cx, cy)` at `rate` per second, strongest at the centre.
    ///
    /// A pure inward impulse cannot gather anything: a particle arrives at the
    /// centre carrying all the speed the pull gave it and sails straight out
    /// the far side, which reads as an orbit rather than a grip. Damping is
    /// what turns the same pull into a hold.
    pub fn damp(&mut self, cx: f32, cy: f32, radius: f32, rate: f32, dt: f32) {
        if self.active == 0
            || !cx.is_finite()
            || !cy.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
        {
            return;
        }
        let dt = finite_or(dt, 0.0).clamp(0.0, MAX_STEP);
        let k = 1.0 - decay(finite_or(rate, 0.0).max(0.0), dt);
        if self.offloaded {
            self.ops.push(offload::OP_DAMP, &[cx, cy, radius, k]);
            return;
        }
        let r2 = radius * radius;

        for i in 0..self.active {
            if self.life[i] <= 0.0 {
                continue;
            }
            let dx = self.x[i] - cx;
            let dy = self.y[i] - cy;
            let d2 = dx * dx + dy * dy;
            if d2 > r2 || d2.is_nan() {
                continue;
            }
            let w = k * smoothstep(radius, 0.0, d2.sqrt());
            self.vx[i] = tame(self.vx[i] * (1.0 - w));
            self.vy[i] = tame(self.vy[i] * (1.0 - w));
        }
    }

    /// Drags every particle within `radius` of `(cx, cy)` *positionally* toward
    /// the centre at `rate` per second, killing its own velocity as it goes.
    ///
    /// Damping alone cannot hold a clump: transport is `drag * fluid + own`, so
    /// the flow the fist injected keeps carrying the gathered particles once the
    /// hand moves on. Moving the position itself is what makes a closed fist
    /// read as a grip the clump follows.
    pub fn grip(&mut self, cx: f32, cy: f32, radius: f32, rate: f32, dt: f32) {
        if self.active == 0
            || !cx.is_finite()
            || !cy.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
        {
            return;
        }
        let dt = finite_or(dt, 0.0).clamp(0.0, MAX_STEP);
        let k = 1.0 - decay(finite_or(rate, 0.0).max(0.0), dt);
        if self.offloaded {
            self.ops.push(offload::OP_GRIP, &[cx, cy, radius, k]);
            return;
        }
        let r2 = radius * radius;

        for i in 0..self.active {
            if self.life[i] <= 0.0 {
                continue;
            }
            let dx = self.x[i] - cx;
            let dy = self.y[i] - cy;
            let d2 = dx * dx + dy * dy;
            if d2 > r2 || d2.is_nan() {
                continue;
            }
            let w = k * smoothstep(radius, 0.0, d2.sqrt());
            self.x[i] = lerp(self.x[i], cx, w);
            self.y[i] = lerp(self.y[i], cy, w);
            self.vx[i] = tame(self.vx[i] * (1.0 - w));
            self.vy[i] = tame(self.vy[i] * (1.0 - w));
        }
    }
}
