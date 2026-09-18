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
}

/// Everything one `step` needs that is not per-particle, resolved once: the
/// fields, the bounds, and every tunable already folded through `dt`. Bundling
/// them keeps the per-particle helpers pure and short-signatured, and keeps the
/// clamping and `decay` calls out of the loop.
struct Frame<'a> {
    vel: &'a VecField,
    /// `None` when there is no usable obstacle field, which skips the test
    /// entirely for the common no-body case.
    obstacle: Option<&'a Grid>,
    /// Grid-space -> obstacle-index scale, for the case where the obstacle
    /// field is not the same resolution as the velocity field.
    osx: f32,
    osy: f32,
    /// Inclusive upper bounds of the velocity grid, `(w - 1, h - 1)`.
    maxx: f32,
    maxy: f32,
    /// Base lifetime for respawns, before jitter.
    life: f32,
    dt: f32,
    /// `params.particle_drag`, clamped: the multiplier on the fluid velocity.
    drag: f32,
    /// Fraction of a particle's own velocity surviving this step.
    keep: f32,
    /// Fraction of `heat` surviving this step.
    heat_keep: f32,
    /// Fraction of the gap to the speed-derived heat floor closed this step.
    heat_rise: f32,
    /// `1 / dt`, for turning this step's displacement into a speed.
    inv_dt: f32,
    /// `curl_influence * dt`.
    swirl: f32,
    /// `gravity * dt`.
    gravity: f32,
}

/// One particle's candidate next state.
#[derive(Clone, Copy)]
struct Advance {
    nx: f32,
    ny: f32,
    ox: f32,
    oy: f32,
    /// False when the particle left the grid, landed in an obstacle, or went
    /// non-finite — all of which mean "recycle it".
    ok: bool,
}

impl Advance {
    /// The state of a particle that is not worth integrating at all.
    const RECYCLE: Self = Self {
        nx: 0.0,
        ny: 0.0,
        ox: 0.0,
        oy: 0.0,
        ok: false,
    };
}

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
        }
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

        for _ in 0..count.min(self.active) {
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

        for _ in 0..count.min(self.active) {
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
}

impl Frame<'_> {
    /// Integrates one particle for a step: RK2 midpoint advection through the
    /// fluid, plus the particle's own velocity carrying gravity, curl swirl and
    /// whatever gestures kicked into it.
    ///
    /// Pure, so the caller can run two of these back to back and let the core
    /// interleave them.
    #[inline(always)]
    fn advance(&self, px: f32, py: f32, ovx: f32, ovy: f32) -> Advance {
        let vel = self.vel;
        let (fu1, fv1) = sample_uv(vel, px, py);

        let mut ox = ovx * self.keep;
        let mut oy = ovy * self.keep;

        // Swirl: perpendicular to the direction of travel, signed by the local
        // curl, so a particle entering a vortex is bent onto a spiral instead
        // of crossing it in a straight line. Scaled off the *unit* travel
        // direction, which keeps the magnitude proportional to omega alone —
        // taken off the unscaled velocity instead, fast particles spin and slow
        // ones (the ones actually trapped in the vortex) barely turn at all.
        if self.swirl != 0.0 {
            let tvx = self.drag * fu1 + ox;
            let tvy = self.drag * fv1 + oy;
            let len2 = tvx * tvx + tvy * tvy;
            // Same guard as `math::normalize` (len > 1e-6), but folding the
            // normalisation into the scale keeps one division off the critical
            // path instead of two.
            if len2 > 1e-12 {
                let k = curl_at(vel, px, py) * self.swirl / len2.sqrt();
                ox -= tvy * k;
                oy += tvx * k;
            }
        }
        oy += self.gravity;
        ox = tame(ox);
        oy = tame(oy);

        // RK2 midpoint: sample, step half, sample again, apply. The own
        // velocity is constant across the step so it needs no correction; only
        // the field sample does, and it is exactly the correction that keeps a
        // fast curved gesture from cutting the corner.
        let dt = self.dt;
        let hx = px + 0.5 * dt * (self.drag * fu1 + ox);
        let hy = py + 0.5 * dt * (self.drag * fv1 + oy);
        let (fu2, fv2) = sample_uv(vel, hx, hy);
        let nx = px + dt * (self.drag * fu2 + ox);
        let ny = py + dt * (self.drag * fv2 + oy);

        let ok = nx.is_finite()
            && ny.is_finite()
            && nx >= 0.0
            && nx <= self.maxx
            && ny >= 0.0
            && ny <= self.maxy
            && !self.is_solid(nx, ny);
        Advance { nx, ny, ox, oy, ok }
    }

    /// Whether grid-space `(x, y)` falls in an obstacle cell.
    ///
    /// Nearest-cell rather than bilinear: the obstacle is a hard binary
    /// predicate, the caller only needs to know which side of 0.5 it is on,
    /// and this is one load instead of four in a loop that runs 220k times a
    /// frame.
    #[inline(always)]
    fn is_solid(&self, x: f32, y: f32) -> bool {
        match self.obstacle {
            None => false,
            Some(o) => {
                // `as usize` saturates: NaN and negatives land on 0, huge
                // values on usize::MAX, and `min` brings both into range.
                let xi = ((x * self.osx + 0.5) as usize).min(o.w - 1);
                let yi = ((y * self.osy + 0.5) as usize).min(o.h - 1);
                o.data[yi * o.w + xi] >= 0.5
            }
        }
    }
}

/// Bilinear sample of both velocity components at once.
///
/// Identical in result to two [`Grid::sample`] calls, but the clamp, floor and
/// index arithmetic are shared: the compiler cannot prove `u.w == v.w`, so
/// going through `VecField::sample` computes all of it twice, and `step` does
/// two of these per particle.
#[inline(always)]
fn sample_uv(vel: &VecField, x: f32, y: f32) -> (f32, f32) {
    let w = vel.u.w;
    let h = vel.u.h;
    // Clamp and integer split exactly as `Grid::sample` does them, for the
    // reasons documented there: `max`/`min` fold NaN to the origin without a
    // branch, and truncation avoids `f32::floor`, which is a libm call on the
    // SSE2 baseline. Four of those per particle measured at 25 ns here, more
    // than the rest of the step put together.
    let x = x.max(0.0).min((w - 1) as f32);
    let y = y.max(0.0).min((h - 1) as f32);

    let ix0 = x as usize;
    let iy0 = y as usize;
    let tx = x - ix0 as f32;
    let ty = y - iy0 as f32;

    let ix1 = (ix0 + 1).min(w - 1);
    let iy1 = (iy0 + 1).min(h - 1);
    let row0 = iy0 * w;
    let row1 = iy1 * w;

    // Nested `lerp`, bit-for-bit what `Grid::sample` computes: the fluid
    // backtraces through the same field, and a particle that disagreed with
    // the solver in the last bit would be a needless source of drift between
    // what the dye shows and where the particles are. (A flattened
    // weighted-sum form measured no faster here, so there is nothing to trade
    // that agreement for.)
    let u = lerp(
        lerp(vel.u.data[row0 + ix0], vel.u.data[row0 + ix1], tx),
        lerp(vel.u.data[row1 + ix0], vel.u.data[row1 + ix1], tx),
        ty,
    );
    let v = lerp(
        lerp(vel.v.data[row0 + ix0], vel.v.data[row0 + ix1], tx),
        lerp(vel.v.data[row1 + ix0], vel.v.data[row1 + ix1], tx),
        ty,
    );
    (u, v)
}

/// Curl `omega = dv/dx - du/dy` at the lattice point nearest `(x, y)`.
///
/// The fluid solver keeps its own curl grid, but `step` is handed only the
/// velocity field, and recomputing it here from four loads is cheaper than the
/// eight bilinear samples an interpolated curl would cost. Matches
/// `Fluid::compute_curl` at interior lattice points, and degrades to a
/// one-sided difference on the border.
#[inline(always)]
fn curl_at(vel: &VecField, x: f32, y: f32) -> f32 {
    let w = vel.u.w;
    let h = vel.u.h;
    let xi = ((x + 0.5) as usize).min(w - 1);
    let yi = ((y + 0.5) as usize).min(h - 1);
    let xm = xi.saturating_sub(1);
    let xp = (xi + 1).min(w - 1);
    let ym = yi.saturating_sub(1);
    let yp = (yi + 1).min(h - 1);
    // The span is 2 in the interior and 1 against a border, so the per-cell
    // normalisation is a select on an integer, not a divide. Two `divss` here
    // measured at ~7 ns per particle, a tenth of the entire step: this runs
    // once per particle per frame, 220k times, on the critical path.
    let inv_dx = if xp - xm == 2 { 0.5 } else { 1.0 };
    let inv_dy = if yp - ym == 2 { 0.5 } else { 1.0 };
    let dvdx = (vel.v.data[yi * w + xp] - vel.v.data[yi * w + xm]) * inv_dx;
    let dudy = (vel.u.data[yp * w + xi] - vel.u.data[ym * w + xi]) * inv_dy;
    dvdx - dudy
}

/// Smallest lane worth scheduling on its own thread: below this the bilinear
/// gathers cost less than the task hand-off.
const LANE_MIN: usize = 4_096;

/// One contiguous band of the pool — every SoA stream for particles
/// `[start, start + len)` — plus the private state advancing it needs.
///
/// Lanes never overlap, so any number of them can run at once without a lock:
/// a lane reads only the shared, immutable [`Frame`] and writes only its own
/// slices. The RNG is per lane so respawn positions never serialise on one
/// generator, and the respawn budget is split up front rather than shared.
struct Lane<'a> {
    x: &'a mut [f32],
    y: &'a mut [f32],
    vx: &'a mut [f32],
    vy: &'a mut [f32],
    life: &'a mut [f32],
    max_life: &'a mut [f32],
    heat: &'a mut [f32],
    rng: Rng,
    /// Respawns this lane may still perform, in particles.
    credit: f32,
}

impl Lane<'_> {
    /// Advances every particle in the lane; returns how many are alive after.
    ///
    /// One linear pass, seven independent streams, no branch in the common
    /// case beyond the cull test.
    ///
    /// The per-particle cost is dominated by the two bilinear gathers RK2
    /// needs, which are dependent (the second samples the midpoint the
    /// first produced). Measured on a 2.8 GHz Xeon, one gather of both
    /// velocity components costs ~10.5 ns whether the grid fits in L1 or
    /// not, and the whole step ~58 ns; hand-pipelining two particles per
    /// iteration to overlap the chains made no difference there, so the
    /// simple form is what stays. On a core that does reorder across
    /// iterations there is nothing here to stop it: `advance` is pure and
    /// every iteration is independent.
    fn run(&mut self, frame: &Frame) -> usize {
        let mut alive = 0usize;
        for i in 0..self.x.len() {
            let life = self.life[i] - frame.dt;
            // The one branch worth taking: a starved pool would otherwise pay
            // two bilinear gathers per frame for particles nobody can see.
            let adv = if life > 0.0 {
                frame.advance(self.x[i], self.y[i], self.vx[i], self.vy[i])
            } else {
                Advance::RECYCLE
            };
            alive += usize::from(self.commit(i, adv, life, frame));
        }
        alive
    }

    /// Writes one particle's advanced state back, or recycles it.
    ///
    /// `life` is the already-aged lifetime and `adv` the candidate state from
    /// [`Frame::advance`]. Returns whether the particle is alive afterwards, so
    /// the caller's `alive` tally costs an add rather than a second pass.
    #[inline(always)]
    fn commit(&mut self, i: usize, adv: Advance, life: f32, f: &Frame) -> bool {
        if life > 0.0 && adv.ok {
            // Speed from the displacement actually applied, which already
            // includes fluid transport, the particle's own residual velocity
            // and the curl swirl — no extra field sample needed.
            let speed = length(
                (adv.nx - self.x[i]) * f.inv_dt,
                (adv.ny - self.y[i]) * f.inv_dt,
            );
            let floor = (speed * (1.0 / HEAT_SPEED_FULL)).min(1.0);

            self.x[i] = adv.nx;
            self.y[i] = adv.ny;
            self.vx[i] = adv.ox;
            self.vy[i] = adv.oy;
            let cooled = self.heat[i] * f.heat_keep;
            // Whichever is hotter: a spell's heat decaying away, or the
            // ambient heat this particle's own motion earns it.
            self.heat[i] = if floor > cooled {
                cooled + (floor - cooled) * f.heat_rise
            } else {
                cooled
            };
            self.life[i] = life;
            return true;
        }
        // Old age, out of the world, or inside the silhouette: all the same
        // fate, and all subject to the same respawn budget.
        if self.credit >= 1.0 {
            self.credit -= 1.0;
            return self.respawn(i, f);
        }
        // No budget: park it. Position is kept in bounds so a dead particle can
        // never render off-screen if the shader ignores `life`, and the velocity
        // is dropped so it does not resume mid-flight several frames later.
        self.life[i] = 0.0;
        self.vx[i] = 0.0;
        self.vy[i] = 0.0;
        self.heat[i] = 0.0;
        self.x[i] = clamp_finite(self.x[i], f.maxx);
        self.y[i] = clamp_finite(self.y[i], f.maxy);
        false
    }

    /// Scatters particle `i` to a fresh random position with a fresh jittered
    /// lifetime, returning whether it came back alive. Uniform over the grid
    /// rather than over the borders: the pool has to stay spatially uniform for
    /// the fluid to be legible everywhere, and edge respawns leave a visible
    /// dead band in the middle of the screen whenever the flow is slow.
    ///
    /// Every call spends one unit of budget whether or not it found a free
    /// cell. Refunding a failed attempt instead would let a silhouette that
    /// covers most of the frame put the *whole* pool through `RESPAWN_TRIES`
    /// samples every frame; spending it caps the work at the budget and merely
    /// thins the population while the obstacle is that large.
    #[cold]
    #[inline(never)]
    fn respawn(&mut self, i: usize, frame: &Frame) -> bool {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut free = false;
        for _ in 0..RESPAWN_TRIES {
            x = self.rng.range(0.0, frame.maxx);
            y = self.rng.range(0.0, frame.maxy);
            if !frame.is_solid(x, y) {
                free = true;
                break;
            }
        }
        self.x[i] = x;
        self.y[i] = y;
        self.vx[i] = 0.0;
        self.vy[i] = 0.0;
        self.heat[i] = 0.0;
        self.max_life[i] = frame.life * self.rng.range(LIFE_JITTER.0, LIFE_JITTER.1);
        self.life[i] = if free { self.max_life[i] } else { 0.0 };
        free
    }
}

#[inline(always)]
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Clamps a particle's own velocity component into the representable band,
/// mapping a non-finite value to rest rather than propagating it.
#[inline(always)]
fn tame(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED)
    } else {
        0.0
    }
}

#[inline(always)]
fn clamp_finite(v: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, max)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 60.0;

    fn const_field(w: usize, h: usize, u: f32, v: f32) -> VecField {
        let mut f = VecField::new(w, h);
        f.u.data.fill(u);
        f.v.data.fill(v);
        f
    }

    /// Rigid rotation about `(cx, cy)`: `u = -omega * (y - cy)`,
    /// `v = omega * (x - cx)`. Each component is linear, so bilinear sampling
    /// reproduces it exactly and the analytic circle is the exact solution.
    fn rot_field(w: usize, h: usize, cx: f32, cy: f32, omega: f32) -> VecField {
        let mut f = VecField::new(w, h);
        for j in 0..h {
            for i in 0..w {
                f.u.set(i, j, -omega * (j as f32 - cy));
                f.v.set(i, j, omega * (i as f32 - cx));
            }
        }
        f
    }

    fn params(drag: f32, life: f32, spawn: f32) -> Params {
        Params {
            particle_drag: drag,
            particle_life: life,
            spawn_rate: spawn,
            ..Params::default()
        }
    }

    fn cfg(curl: f32, gravity: f32, damping: f32) -> ParticleConfig {
        ParticleConfig {
            curl_influence: curl,
            gravity,
            damping,
        }
    }

    /// Advection only: no swirl, no gravity, immortal particles.
    fn pure_advection() -> (Params, ParticleConfig) {
        (params(1.0, 10.0, 0.0), cfg(0.0, 0.0, 0.6))
    }

    #[test]
    fn constant_field_transports_at_the_field_velocity() {
        let vel = const_field(64, 32, 5.0, 0.0);
        let obs = Grid::new(64, 32);
        let (p_params, p_cfg) = pure_advection();
        let mut p = Particles::new(4, 1);
        p.set_active(1);
        p.place(0, 10.0, 16.0, 0.0, 0.0, 100.0);

        let steps = 60;
        for _ in 0..steps {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }

        let (x, y) = p.position(0);
        let expect = 10.0 + 5.0 * DT * steps as f32;
        assert!(
            (x - expect).abs() < 1e-3,
            "x drifted: {x} vs {expect} (sign or scale error)"
        );
        assert!((y - 16.0).abs() < 1e-4, "y should not have moved: {y}");
        assert_eq!(p.alive(), 1);
    }

    #[test]
    fn negative_field_transports_left_and_down() {
        let vel = const_field(64, 64, -3.0, -4.0);
        let obs = Grid::new(64, 64);
        let (p_params, p_cfg) = pure_advection();
        let mut p = Particles::new(1, 2);
        p.set_active(1);
        p.place(0, 40.0, 40.0, 0.0, 0.0, 100.0);
        for _ in 0..30 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }
        let (x, y) = p.position(0);
        assert!((x - (40.0 - 3.0 * DT * 30.0)).abs() < 1e-3, "x {x}");
        assert!((y - (40.0 - 4.0 * DT * 30.0)).abs() < 1e-3, "y {y}");
    }

    #[test]
    fn rk2_tracks_the_analytic_circle_far_better_than_euler() {
        let (w, h) = (128, 128);
        let (cx, cy) = (64.0, 64.0);
        let omega = 1.0;
        let radius = 30.0;
        let vel = rot_field(w, h, cx, cy, omega);
        let obs = Grid::new(w, h);
        let (p_params, p_cfg) = pure_advection();

        let mut p = Particles::new(1, 3);
        p.set_active(1);
        p.place(0, cx + radius, cy, 0.0, 0.0, 1000.0);

        // The same field integrated with one Euler step per frame, as the
        // reference for "what the naive integrator would have done".
        let (mut ex, mut ey) = (cx + radius, cy);

        let steps = 600;
        for _ in 0..steps {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
            let (eu, ev) = (-omega * (ey - cy), omega * (ex - cx));
            ex += DT * eu;
            ey += DT * ev;
        }

        let (x, y) = p.position(0);
        let r_rk2 = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        let r_euler = ((ex - cx).powi(2) + (ey - cy).powi(2)).sqrt();

        // RK2's radius error is O(dt^4) per step, Euler's is O(dt^2).
        assert!(
            (r_rk2 - radius).abs() < 0.01,
            "RK2 radius drifted {} cells",
            r_rk2 - radius
        );
        assert!(
            r_euler - radius > 1.0,
            "reference Euler should have visibly spiralled out, drift {}",
            r_euler - radius
        );

        // Phase must also track: total rotation is omega * t radians.
        let want = omega * DT * steps as f32;
        let got = (y - cy).atan2(x - cx);
        let err = ((got - want) + core::f32::consts::PI).rem_euclid(core::f32::consts::TAU)
            - core::f32::consts::PI;
        assert!(err.abs() < 0.01, "phase error {err} rad");
    }

    #[test]
    fn curl_swirl_turns_the_particle_with_the_vortex() {
        let (w, h) = (64, 64);
        let (cx, cy) = (32.0, 32.0);
        let omega = 3.0;
        let vel = rot_field(w, h, cx, cy, omega);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 10.0, 0.0);

        // At the centre the fluid velocity is zero, so the only thing that can
        // deflect a radially moving particle is the curl swirl.
        let mut with = Particles::new(1, 4);
        with.set_active(1);
        with.place(0, cx, cy, 5.0, 0.0, 100.0);
        with.step(&vel, &obs, DT, &p_params, &cfg(2.0, 0.0, 0.6));

        let (_, vy) = with.velocity(0);
        // curl = dv/dx - du/dy = 2 * omega for a rigid rotation.
        let expect = 2.0 * omega * 2.0 * DT;
        assert!(
            (vy - expect).abs() < 1e-4,
            "swirl magnitude {vy} vs {expect}"
        );
        assert!(vy > 0.0, "swirl must follow the vortex sense");

        let mut without = Particles::new(1, 4);
        without.set_active(1);
        without.place(0, cx, cy, 5.0, 0.0, 100.0);
        without.step(&vel, &obs, DT, &p_params, &cfg(0.0, 0.0, 0.6));
        assert_eq!(
            without.velocity(0).1,
            0.0,
            "curl_influence 0 must not swirl"
        );
    }

    #[test]
    fn own_velocity_decay_is_framerate_independent() {
        // drag 0 removes the fluid entirely, leaving only the damping term.
        let vel = const_field(32, 32, 0.0, 0.0);
        let obs = Grid::new(32, 32);
        let p_params = params(0.0, 10.0, 0.0);
        let p_cfg = cfg(0.0, 0.0, 1.5);

        let run = |dt: f32, steps: usize| {
            let mut p = Particles::new(1, 5);
            p.set_active(1);
            p.place(0, 16.0, 16.0, 10.0, 0.0, 100.0);
            for _ in 0..steps {
                p.step(&vel, &obs, dt, &p_params, &p_cfg);
            }
            p.velocity(0).0
        };

        let slow = run(1.0 / 30.0, 30);
        let fast = run(1.0 / 240.0, 240);
        let analytic = 10.0 * (-1.5f32).exp();
        assert!((slow - fast).abs() < 1e-3, "{slow} vs {fast}");
        assert!((slow - analytic).abs() < 1e-3, "{slow} vs {analytic}");
    }

    #[test]
    fn live_particles_never_end_a_step_inside_an_obstacle() {
        let (w, h) = (64, 64);
        let vel = const_field(w, h, 20.0, 6.0);
        let mut obs = Grid::new(w, h);
        for y in 20..44 {
            for x in 20..44 {
                obs.set(x, y, 1.0);
            }
        }
        let p_params = params(1.0, 3.0, 50_000.0);
        let p_cfg = cfg(2.2, 0.0, 0.6);

        let mut p = Particles::new(2_000, 6);
        p.set_active(2_000);
        p.seed_uniform(w, h, 3.0);

        for _ in 0..400 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
            for i in 0..p.active() {
                if p.life_of(i) <= 0.0 {
                    continue;
                }
                let (x, y) = p.position(i);
                let xi = (x + 0.5) as usize;
                let yi = (y + 0.5) as usize;
                assert!(
                    obs.get(xi, yi) < 0.5,
                    "live particle {i} inside the obstacle at ({x}, {y})"
                );
            }
        }
        assert!(p.alive() > 1_000, "the pool collapsed: {}", p.alive());
    }

    #[test]
    fn no_particle_ever_ends_a_step_outside_the_grid() {
        let (w, h) = (48, 24);
        // Strong outward flow plus gravity: everything is pushed at the walls.
        let mut vel = VecField::new(w, h);
        for y in 0..h {
            for x in 0..w {
                vel.u.set(x, y, (x as f32 - 24.0) * 8.0);
                vel.v.set(x, y, (y as f32 - 12.0) * 8.0);
            }
        }
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 2.0, 30_000.0);
        let p_cfg = cfg(4.0, 60.0, 0.6);

        let mut p = Particles::new(1_500, 7);
        p.set_active(1_500);
        p.seed_uniform(w, h, 2.0);

        for _ in 0..300 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
            for i in 0..p.active() {
                let (x, y) = p.position(i);
                assert!(
                    (0.0..=(w - 1) as f32).contains(&x) && (0.0..=(h - 1) as f32).contains(&y),
                    "particle {i} escaped to ({x}, {y})"
                );
            }
        }
        // And the normalised buffer stays in [0, 1], which is what the shader
        // assumes.
        p.build_render_buffer(w, h);
        for q in p.render_buffer().chunks(PARTICLE_STRIDE) {
            assert!((0.0..=1.0).contains(&q[0]) && (0.0..=1.0).contains(&q[1]));
            assert!((0.0..=1.0).contains(&q[3]));
        }
    }

    #[test]
    fn everything_stays_finite_under_a_violent_field() {
        let (w, h) = (32, 32);
        let mut vel = VecField::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let s = if (x + y) % 2 == 0 { 9_000.0 } else { -9_000.0 };
                vel.u.set(x, y, s);
                vel.v.set(x, y, -s);
            }
        }
        let mut obs = Grid::new(w, h);
        for y in 10..22 {
            for x in 10..22 {
                obs.set(x, y, 1.0);
            }
        }
        let p_params = params(4.0, 1.0, 200_000.0);
        let p_cfg = cfg(20.0, 5_000.0, 0.6);

        let mut p = Particles::new(512, 8);
        p.set_active(512);
        p.seed_uniform(w, h, 1.0);

        for _ in 0..1_000 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }
        for i in 0..p.active() {
            let (x, y) = p.position(i);
            let (vx, vy) = p.velocity(i);
            assert!(
                x.is_finite() && y.is_finite() && vx.is_finite() && vy.is_finite(),
                "particle {i} went non-finite: ({x}, {y}) ({vx}, {vy})"
            );
            assert!(p.life_of(i).is_finite() && p.heat_of(i).is_finite());
        }
        p.build_render_buffer(w, h);
        assert!(p.render_buffer().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn alive_count_is_stable_over_thousands_of_steps() {
        let (w, h) = (64, 36);
        let vel = const_field(w, h, 2.0, -1.0);
        let obs = Grid::new(w, h);
        let n = 2_000;
        // Steady-state death rate is n / mean_life = 2000/s; give the budget
        // some headroom and the population should sit at the pool size.
        let p_params = params(1.0, 1.0, 3_000.0);
        let p_cfg = cfg(2.2, 0.0, 0.6);

        let mut p = Particles::new(n, 9);
        p.set_active(n);
        p.seed_uniform(w, h, 1.0);

        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for s in 0..6_000 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
            // Skip the first second: the seeded generation dies off on its own
            // schedule before the respawn loop reaches equilibrium.
            if s > 60 {
                lo = lo.min(p.alive());
                hi = hi.max(p.alive());
            }
        }
        assert!(lo > (n as f32 * 0.9) as usize, "pool died down to {lo}");
        assert!(hi <= n, "more alive than active: {hi}");
    }

    #[test]
    fn spawn_rate_is_a_budget_not_a_suggestion() {
        let (w, h) = (32, 32);
        let vel = const_field(w, h, 0.0, 0.0);
        let obs = Grid::new(w, h);
        let n = 2_000;
        // 100 respawns/s against a 1 s lifetime can only sustain ~100 alive.
        let p_params = params(1.0, 1.0, 100.0);
        let p_cfg = cfg(0.0, 0.0, 0.6);

        let mut p = Particles::new(n, 10);
        p.set_active(n);
        p.seed_uniform(w, h, 1.0);
        for _ in 0..600 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }
        let alive = p.alive();
        assert!(
            alive > 20,
            "the budget starved the pool completely: {alive}"
        );
        assert!(
            alive < 400,
            "spawn_rate was ignored: {alive} alive on a 100/s budget"
        );

        // With no budget at all the pool must be allowed to empty.
        let mut dead = Particles::new(64, 11);
        dead.set_active(64);
        dead.seed_uniform(w, h, 0.5);
        let starved = params(1.0, 1.0, 0.0);
        for _ in 0..120 {
            dead.step(&vel, &obs, DT, &starved, &p_cfg);
        }
        assert_eq!(dead.alive(), 0);
    }

    #[test]
    fn lifetimes_are_staggered_after_seeding_and_after_respawn() {
        let (w, h) = (64, 36);
        let vel = const_field(w, h, 0.0, 0.0);
        let obs = Grid::new(w, h);
        let n = 600;

        let spread = |p: &Particles| {
            let lives: Vec<f32> = (0..p.active()).map(|i| p.life_of(i)).collect();
            let mean = lives.iter().sum::<f32>() / lives.len() as f32;
            let var =
                lives.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / lives.len() as f32;
            (mean, var.sqrt())
        };

        let mut p = Particles::new(n, 12);
        p.set_active(n);
        p.seed_uniform(w, h, 2.0);
        let (mean, sd) = spread(&p);
        assert!(sd > 0.2 * mean, "seeded lifetimes are bunched: sd {sd}");

        // Now force the whole pool onto one clock and check the respawn path
        // breaks it up again.
        for i in 0..n {
            p.place(i, 10.0, 10.0, 0.0, 0.0, 0.5);
        }
        let p_params = params(1.0, 2.0, 100_000.0);
        let p_cfg = cfg(0.0, 0.0, 0.6);
        for _ in 0..40 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
        }
        let (mean2, sd2) = spread(&p);
        assert_eq!(p.alive(), n, "generous budget should refill the pool");
        assert!(
            sd2 > 0.1 * mean2,
            "respawn put the pool back in lockstep: sd {sd2}, mean {mean2}"
        );
        // Positions must be scattered too, not all sitting on (10, 10).
        let distinct = (0..n).filter(|&i| p.position(i).0 != 10.0).count();
        assert!(distinct > n - 10, "respawn did not scatter: {distinct}");
    }

    #[test]
    fn spawn_burst_is_local_outward_and_capacity_safe() {
        let mut p = Particles::new(64, 13);
        p.set_active(64);
        let (cx, cy) = (30.0, 20.0);
        // Ask for more than the pool holds: must wrap, not panic.
        p.spawn_burst(cx, cy, 500, 40.0, 0.8, 1.5);

        for i in 0..p.active() {
            let (x, y) = p.position(i);
            let (vx, vy) = p.velocity(i);
            let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            assert!(d <= BURST_SCATTER + 1e-4, "particle {i} landed {d} away");
            // Placed along its own velocity, so the offset is strictly outward.
            let dot = (x - cx) * vx + (y - cy) * vy;
            assert!(dot >= 0.0, "particle {i} moves inward: {dot}");
            assert!(p.life_of(i) > 0.0);
            assert!((p.heat_of(i) - 0.8).abs() < 1e-6);
        }
        let speeds: Vec<f32> = (0..p.active())
            .map(|i| {
                let (vx, vy) = p.velocity(i);
                (vx * vx + vy * vy).sqrt()
            })
            .collect();
        assert!(speeds.iter().any(|&s| s > 1.0), "burst has no speed at all");
        assert!(
            speeds.iter().all(|&s| s <= 40.0 + 1e-3),
            "burst exceeded the requested speed"
        );

        // Degenerate input must be ignored, not propagated.
        p.spawn_burst(f32::NAN, cy, 10, 5.0, 0.5, 1.0);
        p.spawn_burst(cx, cy, 10, f32::INFINITY, 0.5, 1.0);
        for i in 0..p.active() {
            let (vx, vy) = p.velocity(i);
            assert!(vx.is_finite() && vy.is_finite());
            assert!(p.position(i).0.is_finite());
        }
    }

    #[test]
    fn impulse_pushes_near_particles_out_and_leaves_far_ones_alone() {
        let mut p = Particles::new(4, 14);
        p.set_active(4);
        let (cx, cy) = (20.0, 20.0);
        p.place(0, 22.0, 20.0, 0.0, 0.0, 5.0); // right, close
        p.place(1, 20.0, 14.0, 0.0, 0.0, 5.0); // below, near the edge
        p.place(2, 40.0, 20.0, 0.0, 0.0, 5.0); // far outside
        p.place(3, 17.0, 17.0, 0.0, 0.0, 5.0); // diagonal, close
        p.impulse(cx, cy, 8.0, 50.0, 0.9);

        for i in [0usize, 1, 3] {
            let (x, y) = p.position(i);
            let (vx, vy) = p.velocity(i);
            let dot = (x - cx) * vx + (y - cy) * vy;
            assert!(dot > 0.0, "particle {i} was not pushed outward: {dot}");
            assert!(p.heat_of(i) > 0.0, "particle {i} got no heat");
        }
        assert_eq!(p.velocity(2), (0.0, 0.0), "distant particle was disturbed");
        assert_eq!(p.heat_of(2), 0.0);

        // Falloff must be monotonic in distance and reach zero at the edge.
        let near = p.velocity(0);
        let edge = p.velocity(1);
        let near_mag = (near.0 * near.0 + near.1 * near.1).sqrt();
        let edge_mag = (edge.0 * edge.0 + edge.1 * edge.1).sqrt();
        assert!(
            near_mag > edge_mag,
            "falloff is not monotonic: {near_mag} at r=2 vs {edge_mag} at r=6"
        );

        let mut boundary = Particles::new(1, 15);
        boundary.set_active(1);
        boundary.place(0, cx + 8.0, cy, 0.0, 0.0, 5.0);
        boundary.impulse(cx, cy, 8.0, 50.0, 0.9);
        assert_eq!(
            boundary.velocity(0),
            (0.0, 0.0),
            "a hard cutoff at the edge leaves a visible ring"
        );

        // Degenerate radius / non-finite centre must be no-ops.
        let before = p.velocity(0);
        p.impulse(cx, cy, 0.0, 50.0, 0.5);
        p.impulse(f32::NAN, cy, 8.0, 50.0, 0.5);
        p.impulse(cx, cy, 8.0, f32::NAN, 0.5);
        assert_eq!(p.velocity(0), before);
    }

    #[test]
    fn identical_seeds_and_inputs_produce_identical_buffers() {
        let (w, h) = (48, 32);
        let vel = const_field(w, h, 7.0, -3.0);
        let mut obs = Grid::new(w, h);
        for y in 12..20 {
            for x in 12..20 {
                obs.set(x, y, 1.0);
            }
        }
        let p_params = params(1.0, 0.8, 40_000.0);
        let p_cfg = cfg(2.2, 3.0, 0.6);

        let run = || {
            let mut p = Particles::new(300, 0xDEAD_BEEF);
            p.set_active(300);
            p.seed_uniform(w, h, 0.8);
            for s in 0..200 {
                if s == 50 {
                    p.spawn_burst(24.0, 16.0, 40, 30.0, 1.0, 0.6);
                }
                if s == 90 {
                    p.impulse(24.0, 16.0, 10.0, 25.0, 0.7);
                }
                p.step(&vel, &obs, DT, &p_params, &p_cfg);
            }
            p.build_render_buffer(w, h);
            p.render_buffer().to_vec()
        };

        let a = run();
        let b = run();
        assert_eq!(a, b, "same seed and inputs diverged");
        assert_eq!(a.len(), 300 * PARTICLE_STRIDE);
    }

    #[test]
    fn empty_pools_and_absurd_timesteps_are_survivable() {
        let (w, h) = (16, 16);
        let vel = const_field(w, h, 3.0, 3.0);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 1.0, 1_000.0);
        let p_cfg = cfg(2.2, 10.0, 0.6);

        // Zero-capacity pool: every entry point must be a no-op.
        let mut empty = Particles::new(0, 16);
        empty.set_active(100);
        assert_eq!(empty.active(), 0);
        empty.seed_uniform(w, h, 1.0);
        empty.spawn_burst(4.0, 4.0, 10, 5.0, 1.0, 1.0);
        empty.impulse(4.0, 4.0, 5.0, 5.0, 1.0);
        empty.step(&vel, &obs, DT, &p_params, &p_cfg);
        empty.build_render_buffer(w, h);
        assert_eq!(empty.alive(), 0);
        assert!(empty.render_buffer().is_empty());

        // Deactivated pool: nothing moves, nothing panics.
        let mut idle = Particles::new(8, 17);
        idle.set_active(8);
        idle.seed_uniform(w, h, 1.0);
        let before = idle.position(0);
        idle.set_active(0);
        idle.step(&vel, &obs, DT, &p_params, &p_cfg);
        assert_eq!(idle.alive(), 0);
        assert_eq!(idle.position(0), before);
        assert!(idle.render_buffer().is_empty());

        // Absurd timesteps.
        let mut p = Particles::new(64, 18);
        p.set_active(64);
        p.seed_uniform(w, h, 1.0);
        for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e9, 1.0 / 480.0] {
            p.step(&vel, &obs, dt, &p_params, &p_cfg);
            for i in 0..p.active() {
                let (x, y) = p.position(i);
                assert!(
                    (0.0..=(w - 1) as f32).contains(&x) && (0.0..=(h - 1) as f32).contains(&y),
                    "dt {dt} moved particle {i} to ({x}, {y})"
                );
                assert!(p.life_of(i).is_finite(), "dt {dt} broke life");
            }
        }
        // A NaN dt must not be read as "everything died".
        let mut nan_dt = Particles::new(32, 19);
        nan_dt.set_active(32);
        nan_dt.seed_uniform(w, h, 5.0);
        nan_dt.step(&vel, &obs, f32::NAN, &p_params, &p_cfg);
        assert_eq!(nan_dt.alive(), 32, "a NaN dt wiped the pool");
    }

    #[test]
    fn a_particle_placed_at_nan_is_culled_not_propagated() {
        let (w, h) = (16, 16);
        let vel = const_field(w, h, 1.0, 1.0);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 1.0, 10_000.0);
        let p_cfg = cfg(2.2, 0.0, 0.6);

        let mut p = Particles::new(2, 20);
        p.set_active(2);
        p.place(0, f32::NAN, 8.0, 0.0, 0.0, 5.0);
        p.place(1, 8.0, 8.0, f32::INFINITY, 0.0, 5.0);
        p.step(&vel, &obs, DT, &p_params, &p_cfg);
        for i in 0..2 {
            let (x, y) = p.position(i);
            let (vx, vy) = p.velocity(i);
            assert!(x.is_finite() && y.is_finite(), "({x}, {y})");
            assert!(vx.is_finite() && vy.is_finite(), "({vx}, {vy})");
            assert!((0.0..=(w - 1) as f32).contains(&x));
        }
    }

    #[test]
    fn heat_decays_toward_zero() {
        let (w, h) = (16, 16);
        let vel = const_field(w, h, 0.0, 0.0);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 100.0, 0.0);
        let p_cfg = cfg(0.0, 0.0, 0.6);

        let mut p = Particles::new(1, 21);
        p.set_active(1);
        p.place(0, 8.0, 8.0, 0.0, 0.0, 100.0);
        p.impulse(8.0, 8.0, 4.0, 0.0, 1.0);
        assert!((p.heat_of(0) - 1.0).abs() < 1e-6);

        let mut prev = p.heat_of(0);
        for _ in 0..60 {
            p.step(&vel, &obs, DT, &p_params, &p_cfg);
            let now = p.heat_of(0);
            assert!(now < prev, "heat did not decay: {now} >= {prev}");
            prev = now;
        }
        let analytic = (-HEAT_DECAY).exp();
        assert!((prev - analytic).abs() < 1e-3, "{prev} vs {analytic}");
        assert!(prev > 0.0);
    }

    #[test]
    fn a_fast_flow_makes_particles_glow() {
        // The counterpart to `heat_decays_toward_zero`: heat is the colour-ramp
        // coordinate, so a particle the fluid is carrying quickly has to earn
        // colour on its own. Without this, the only coloured particles are the
        // ones a spell touched, and the flow renders as near-black filaments.
        //
        // The grid is wide enough that neither particle reaches the far edge in
        // the measured second. A narrow one recycles the fast particle
        // mid-test, and respawn resets heat — which is how this test failed
        // the first time, reporting 0 for a particle that was in fact glowing.
        let (w, h) = (256, 32);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 100.0, 0.0);
        let p_cfg = cfg(0.0, 0.0, 0.6);
        let steps = 60;

        let run = |speed: f32, seed: u64| {
            let field = const_field(w, h, speed, 0.0);
            let mut p = Particles::new(1, seed);
            p.set_active(1);
            p.place(0, 1.0, 16.0, 0.0, 0.0, 100.0);
            for _ in 0..steps {
                p.step(&field, &obs, DT, &p_params, &p_cfg);
            }
            // Confirm it never recycled, or the heat reading means nothing.
            assert!(p.position(0).0 > 1.0, "particle did not move");
            p.heat_of(0)
        };

        let hot = run(HEAT_SPEED_FULL, 31);
        let warm = run(HEAT_SPEED_FULL * 0.25, 32);

        // Well into the hot half of the ramp. Not tighter than that: the
        // actual figure depends on how far the rise converges in the measured
        // second and on the particle's own velocity relaxing, and pinning the
        // bar to the current output would make this a test of today's
        // arithmetic rather than of the intended behaviour.
        assert!(
            hot > 0.6,
            "a particle at the full-heat speed stayed cold: {hot}"
        );
        // A quarter of the speed must read as distinctly cooler, or the ramp
        // carries no information and every moving particle is one colour.
        assert!(
            warm < hot - 0.4,
            "heat does not discriminate speed: {warm} vs {hot}"
        );
        assert!(warm > 0.0, "a moving particle should have some heat: {warm}");
    }

    #[test]
    fn heat_never_leaves_the_unit_interval() {
        // The renderer indexes a colour ramp with this, so a value outside
        // [0, 1] is a garbage pixel. An absurd field is the stress case.
        let (w, h) = (32, 32);
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 100.0, 0.0);
        let p_cfg = cfg(8.0, 0.0, 0.6);
        let violent = const_field(w, h, 50_000.0, -50_000.0);

        let mut p = Particles::new(64, 33);
        p.set_active(64);
        p.seed_uniform(w, h, 100.0);
        p.impulse(16.0, 16.0, 40.0, 5000.0, 1.0);
        for _ in 0..120 {
            p.step(&violent, &obs, DT, &p_params, &p_cfg);
        }
        for i in 0..64 {
            let heat = p.heat_of(i);
            assert!(
                (0.0..=1.0).contains(&heat),
                "particle {i} heat {heat} outside the colour ramp"
            );
        }
    }

    /// A `Frame` that does nothing but look up obstacles, for unit-testing the
    /// cull predicate directly.
    fn obstacle_frame<'a>(vel: &'a VecField, obs: &'a Grid) -> Frame<'a> {
        Frame {
            vel,
            obstacle: Some(obs),
            osx: (obs.w - 1) as f32 / (vel.w() - 1) as f32,
            osy: (obs.h - 1) as f32 / (vel.h() - 1) as f32,
            maxx: (vel.w() - 1) as f32,
            maxy: (vel.h() - 1) as f32,
            life: 1.0,
            dt: DT,
            drag: 1.0,
            keep: 1.0,
            heat_keep: 1.0,
            heat_rise: 0.0,
            inv_dt: 1.0 / DT,
            swirl: 0.0,
            gravity: 0.0,
        }
    }

    #[test]
    fn obstacle_test_matches_the_solver_threshold() {
        // The fluid treats `>= 0.5` as solid; the particle cull must agree, or
        // particles pile up in the half-cell the two disagree about.
        let vel = const_field(8, 8, 0.0, 0.0);
        let mut obs = Grid::new(8, 8);
        obs.set(4, 4, 0.49);
        assert!(!obstacle_frame(&vel, &obs).is_solid(4.0, 4.0));
        obs.set(4, 4, 0.5);
        let frame = obstacle_frame(&vel, &obs);
        assert!(frame.is_solid(4.0, 4.0));
        assert!(frame.is_solid(4.49, 4.49), "nearest cell is still (4, 4)");
        assert!(!frame.is_solid(3.49, 4.0));
        // Out of range and non-finite coordinates must clamp, not index wild.
        assert!(!frame.is_solid(f32::NAN, f32::NAN));
        assert!(!frame.is_solid(1e9, 1e9));
        assert!(!frame.is_solid(-1e9, -1e9));
    }

    #[test]
    fn sample_uv_agrees_with_grid_sample() {
        let mut vel = VecField::new(9, 5);
        for y in 0..5 {
            for x in 0..9 {
                vel.u.set(x, y, (x * 3 + y) as f32);
                vel.v.set(x, y, (x as f32) * -0.5 + y as f32);
            }
        }
        for &(x, y) in &[
            (0.0, 0.0),
            (3.25, 1.75),
            (8.0, 4.0),
            (-5.0, 2.0),
            (100.0, 100.0),
            (f32::NAN, 1.0),
        ] {
            let fused = sample_uv(&vel, x, y);
            let reference = vel.sample(x, y);
            assert_eq!(fused, reference, "mismatch at ({x}, {y})");
        }
    }

    #[test]
    fn curl_matches_an_analytic_rigid_rotation() {
        let vel = rot_field(32, 32, 16.0, 16.0, 2.5);
        // omega = dv/dx - du/dy = 2 * 2.5 in the interior.
        assert!((curl_at(&vel, 16.0, 16.0) - 5.0).abs() < 1e-4);
        assert!((curl_at(&vel, 4.2, 27.8) - 5.0).abs() < 1e-4);
        // A pure translation has no curl.
        let flat = const_field(16, 16, 9.0, -4.0);
        assert_eq!(curl_at(&flat, 8.0, 8.0), 0.0);
        // Border lookups must clamp instead of panicking.
        assert!(curl_at(&vel, 0.0, 0.0).is_finite());
        assert!(curl_at(&vel, 31.0, 31.0).is_finite());
        assert!(curl_at(&vel, f32::NAN, 1e9).is_finite());
    }

    #[test]
    fn transport_scales_linearly_with_particle_drag() {
        // Every other advection test runs at drag 1.0, where a factor of one is
        // invisible. `particle_drag` is a HUD slider over 0..4, so the multiplier
        // has to be exact across the range, not just non-zero.
        let vel = const_field(64, 64, 10.0, 0.0);
        let obs = Grid::new(64, 64);
        for drag in [0.0, 0.5, 1.0, 2.0] {
            let p_params = params(drag, 10.0, 0.0);
            let mut p = Particles::new(1, 1);
            p.set_active(1);
            p.place(0, 5.0, 32.0, 0.0, 0.0, 100.0);
            for _ in 0..60 {
                p.step(&vel, &obs, DT, &p_params, &cfg(0.0, 0.0, 0.6));
            }
            let want = 5.0 + drag * 10.0 * DT * 60.0;
            let got = p.position(0).0;
            assert!((got - want).abs() < 1e-3, "drag {drag}: {got} vs {want}");
        }
    }

    #[test]
    fn the_whole_step_agrees_at_30_and_144_fps() {
        // `own_velocity_decay_is_framerate_independent` isolates the damping with
        // the fluid switched off. This is the composite: advection, curl swirl and
        // damping together at the shipped defaults, which is the only combination
        // a player ever sees.
        let (w, h) = (256, 144);
        let mut vel = VecField::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - 128.0, y as f32 - 72.0);
                vel.u.set(x, y, -0.35 * dy + 6.0);
                vel.v.set(x, y, 0.35 * dx);
            }
        }
        let obs = Grid::new(w, h);
        let p_params = params(1.0, 10.0, 0.0);
        let p_cfg = ParticleConfig::default();
        let run = |dt: f32, steps: usize| {
            let mut p = Particles::new(1, 3);
            p.set_active(1);
            p.place(0, 100.0, 40.0, 0.0, 0.0, 100.0);
            for _ in 0..steps {
                p.step(&vel, &obs, dt, &p_params, &p_cfg);
            }
            p.position(0)
        };
        // One second of travel is ~19 cells here, so the tolerance is 0.3% of
        // the path: a swirl or gravity term missing its `dt`, or an advection
        // step that integrated per frame instead of per second, lands orders of
        // magnitude outside it. (The damping form is pinned separately, by
        // `own_velocity_decay_is_framerate_independent`, which isolates it.)
        let (ax, ay) = run(1.0 / 30.0, 30);
        let (bx, by) = run(1.0 / 144.0, 144);
        assert!(
            (ax - bx).abs() < 0.05 && (ay - by).abs() < 0.05,
            "30 fps ({ax}, {ay}) vs 144 fps ({bx}, {by})"
        );
    }

    #[test]
    fn a_lower_resolution_obstacle_still_culls_correctly() {
        // The engine hands `step` an obstacle at the fluid's resolution, so the
        // `osx`/`osy` rescale is dead code there — and dead code is where a
        // transposed or off-by-one factor hides until someone changes a constant.
        let (w, h) = (64, 36);
        let vel = const_field(w, h, 15.0, 0.0);
        let (ow, oh) = (32, 18);
        let mut obs = Grid::new(ow, oh);
        for y in 0..oh {
            for x in ow / 2..ow {
                obs.set(x, y, 1.0);
            }
        }
        let p_params = params(1.0, 5.0, 60_000.0);
        let mut p = Particles::new(400, 5);
        p.set_active(400);
        p.seed_uniform(w, h, 5.0);

        let mut live_seen = 0;
        let mut deepest = 0.0f32;
        for _ in 0..200 {
            p.step(&vel, &obs, DT, &p_params, &ParticleConfig::default());
            for i in 0..p.active() {
                if p.life_of(i) <= 0.0 {
                    continue;
                }
                live_seen += 1;
                let (x, y) = p.position(i);
                let ox = x * (ow - 1) as f32 / (w - 1) as f32;
                let oy = y * (oh - 1) as f32 / (h - 1) as f32;
                assert!(
                    obs.get((ox + 0.5) as usize, (oy + 0.5) as usize) < 0.5,
                    "live particle at ({x}, {y}) is inside the half-res obstacle"
                );
                deepest = deepest.max(x);
            }
        }
        assert!(live_seen > 10_000, "nothing survived to test: {live_seen}");
        // The wall in grid space sits where `round(x * (ow - 1) / (w - 1))`
        // first reaches `ow / 2`, i.e. x = 31.5. Asserting only "no live
        // particle is inside" would pass just as happily with the scale off by
        // a cell in the *conservative* direction, so pin both sides: the flow
        // pushes particles at the wall every frame, so the deepest live one has
        // to end up in the last free half-cell.
        let wall = 15.5 * (w - 1) as f32 / (ow - 1) as f32;
        assert!(
            deepest > wall - 0.5 && deepest < wall,
            "the obstacle rescale is off: deepest live x {deepest}, wall at {wall}"
        );
    }

    #[test]
    fn garbage_fields_and_parameters_keep_every_invariant() {
        // The fluid sanitises itself once every few frames, so `step` can and
        // does get handed a field with a NaN or an infinity in it. None of the
        // per-particle state may inherit it, and `alive()` must keep agreeing
        // with the number of particles that actually have life left.
        let mut r = Rng::new(0xF00D);
        let (w, h) = (40, 24);
        for round in 0..120 {
            let mut vel = VecField::new(w, h);
            for i in 0..w * h {
                let pick = r.next_f32();
                vel.u.data[i] = match pick {
                    p if p < 0.02 => f32::NAN,
                    p if p < 0.04 => f32::INFINITY,
                    p if p < 0.06 => f32::NEG_INFINITY,
                    _ => r.range(-400.0, 400.0),
                };
                vel.v.data[i] = r.range(-400.0, 400.0);
            }
            let mut obs = Grid::new(w, h);
            for i in 0..w * h {
                obs.data[i] = if r.next_f32() < 0.3 { 1.0 } else { 0.0 };
            }
            let p_params = Params {
                particle_drag: r.range(-1.0, 6.0),
                particle_life: r.range(-1.0, 40.0),
                // Every third round starves the respawn budget, which is the
                // only way to exercise the "park it in bounds" branch.
                spawn_rate: if round % 3 == 0 {
                    0.0
                } else {
                    r.range(-100.0, 90_000.0)
                },
                ..Params::default()
            };
            let p_cfg = ParticleConfig {
                curl_influence: r.range(-20.0, 20.0),
                gravity: r.range(-300.0, 300.0),
                damping: r.range(-2.0, 25.0),
            };
            let mut p = Particles::new(120, round as u64);
            p.set_active(120);
            p.seed_uniform(w, h, 2.0);

            for s in 0..12 {
                let dt = match s % 7 {
                    0 => f32::NAN,
                    1 => 1e9,
                    2 => -0.01,
                    3 => 0.0,
                    _ => r.range(1.0 / 480.0, 1.0 / 20.0),
                };
                if s % 5 == 0 {
                    p.spawn_burst(
                        r.range(-5.0, 45.0),
                        r.range(-5.0, 30.0),
                        20,
                        r.range(-80.0, 80.0),
                        r.range(-0.5, 1.5),
                        r.range(-1.0, 3.0),
                    );
                }
                if s % 4 == 0 {
                    p.impulse(
                        r.range(-5.0, 45.0),
                        r.range(-5.0, 30.0),
                        r.range(-2.0, 30.0),
                        r.range(-200.0, 200.0),
                        r.range(-0.5, 1.5),
                    );
                }
                p.step(&vel, &obs, dt, &p_params, &p_cfg);

                let counted = (0..p.active()).filter(|&i| p.life_of(i) > 0.0).count();
                assert_eq!(counted, p.alive(), "round {round} step {s}");
                for i in 0..p.active() {
                    let (x, y) = p.position(i);
                    let (vx, vy) = p.velocity(i);
                    assert!(
                        x.is_finite() && y.is_finite() && vx.is_finite() && vy.is_finite(),
                        "round {round} step {s} particle {i}: ({x}, {y}) ({vx}, {vy})"
                    );
                    assert!(
                        (0.0..=(w - 1) as f32).contains(&x)
                            && (0.0..=(h - 1) as f32).contains(&y),
                        "round {round} step {s} particle {i} escaped to ({x}, {y})"
                    );
                    let heat = p.heat_of(i);
                    assert!((0.0..=1.0).contains(&heat), "heat {heat} out of range");
                    if p.life_of(i) > 0.0 {
                        assert!(
                            obs.get((x + 0.5) as usize, (y + 0.5) as usize) < 0.5,
                            "live particle inside the obstacle at ({x}, {y})"
                        );
                    }
                }
                p.build_render_buffer(w, h);
                for q in p.render_buffer().chunks(PARTICLE_STRIDE) {
                    assert!(
                        q.iter().all(|v| (0.0..=1.0).contains(v)),
                        "render buffer out of the shader's range: {q:?}"
                    );
                }
            }
        }
    }
}
