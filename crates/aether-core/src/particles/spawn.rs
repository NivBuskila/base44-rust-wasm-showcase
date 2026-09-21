//! Population: the initial scatter and the burst/ring emitters.
//!
//! Split out of `particles/mod.rs`: the parent keeps the pool, its tuning
//! constants and the small accessors, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl Particles {
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
}
