//! The per-step machinery, split by what it touches.
//!
//! - [`Lane`] (here) is a contiguous band of the pool that one thread advances
//!   without locks: lanes never overlap, so each reads only the shared,
//!   immutable [`Frame`] and writes only its own slices. The RNG is per lane so
//!   respawn positions never serialise on one generator, and the respawn budget
//!   is split up front rather than shared.
//! - `frame.rs` is that shared frame plus the pure RK2 integrate of one
//!   particle.
//! - `sample.rs` is the bilinear velocity gather and the curl read, the two
//!   field reads on the critical path.
//! - `guards.rs` is the finite clamping every write goes through.

mod frame;
mod guards;
mod sample;

pub(super) use frame::{Advance, Frame};
pub(super) use guards::{clamp_finite, finite_or, tame};
pub(super) use sample::{curl_at, sample_uv};

use super::*;

/// Smallest lane worth scheduling on its own thread: below this the bilinear
/// gathers cost less than the task hand-off.
pub(super) const LANE_MIN: usize = 4_096;

/// One contiguous band of the pool — every SoA stream for particles
/// `[start, start + len)` — plus the private state advancing it needs.
///
/// Lanes never overlap, so any number of them can run at once without a lock:
/// a lane reads only the shared, immutable [`Frame`] and writes only its own
/// slices. The RNG is per lane so respawn positions never serialise on one
/// generator, and the respawn budget is split up front rather than shared.
pub(super) struct Lane<'a> {
    pub(super) x: &'a mut [f32],
    pub(super) y: &'a mut [f32],
    pub(super) vx: &'a mut [f32],
    pub(super) vy: &'a mut [f32],
    pub(super) life: &'a mut [f32],
    pub(super) max_life: &'a mut [f32],
    pub(super) heat: &'a mut [f32],
    pub(super) rng: Rng,
    /// Respawns this lane may still perform, in particles.
    pub(super) credit: f32,
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
    pub(super) fn run(&mut self, frame: &Frame) -> usize {
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
    pub(super) fn commit(&mut self, i: usize, adv: Advance, life: f32, f: &Frame) -> bool {
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
    pub(super) fn respawn(&mut self, i: usize, frame: &Frame) -> bool {
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
