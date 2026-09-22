//! Spell forces applied to a region of the pool: impulse, damp, grip.
//!
//! Split out of `particles/mod.rs`: the parent keeps the pool, its tuning
//! constants and the small accessors, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl Particles {
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
