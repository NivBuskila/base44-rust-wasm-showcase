//! The per-frame advance and the lane split it is parallelised over.
//!
//! Split out of `particles/mod.rs`: the parent keeps the pool, its tuning
//! constants and the small accessors, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl Particles {
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
        // and each lane is advanced independently, several per thread when the
        // `parallel` feature is on so a thread that finishes early can steal.
        // Nothing is shared between lanes: each gets its own RNG stream seeded
        // from the pool's, and an equal slice of the respawn budget. With one
        // thread this is exactly the old single pass.
        let lane_len = par::balanced_len(self.active, LANE_MIN);
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
}
