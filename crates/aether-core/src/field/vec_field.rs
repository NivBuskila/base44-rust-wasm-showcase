//! The planar two-component velocity field over a pair of [`Grid`]s.

use super::Grid;

/// A two-component (velocity) field stored as two planar grids, which is what
/// lets the SIMD fluid kernels stride linearly over each component.
#[derive(Clone, Debug)]
pub struct VecField {
    pub u: Grid,
    pub v: Grid,
}

impl VecField {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            u: Grid::new(w, h),
            v: Grid::new(w, h),
        }
    }

    #[inline]
    pub fn w(&self) -> usize {
        self.u.w
    }

    #[inline]
    pub fn h(&self) -> usize {
        self.u.h
    }

    #[inline]
    pub fn zero(&mut self) {
        self.u.zero();
        self.v.zero();
    }

    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        (self.u.sample(x, y), self.v.sample(x, y))
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> (f32, f32) {
        (self.u.get(x, y), self.v.get(x, y))
    }

    #[inline]
    pub fn add(&mut self, x: usize, y: usize, du: f32, dv: f32) {
        self.u.add(x, y, du);
        self.v.add(x, y, dv);
    }

    /// Adds a radial velocity impulse of `(fx, fy)` centred at `(cx, cy)`.
    pub fn splat(&mut self, cx: f32, cy: f32, radius: f32, fx: f32, fy: f32) {
        self.u.splat(cx, cy, radius, fx);
        self.v.splat(cx, cy, radius, fy);
    }

    #[inline]
    pub fn scale(&mut self, k: f32) {
        self.u.scale(k);
        self.v.scale(k);
    }

    /// Mean kinetic energy per cell, for the HUD and for divergence tests.
    pub fn energy(&self) -> f32 {
        let n = self.u.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f32 = self
            .u
            .data
            .iter()
            .zip(&self.v.data)
            .map(|(u, v)| u * u + v * v)
            .sum();
        0.5 * sum / n
    }

    /// Scales down any cell whose speed exceeds `limit`, leaving slower cells
    /// untouched.
    ///
    /// A per-cell scale is not divergence-free, unlike the uniform one in
    /// `scale`, so this leaves a little divergence behind wherever it bites —
    /// the next step's projection removes it. That is the cheaper trade: the
    /// alternative is a field that ratchets upward for as long as someone keeps
    /// gesturing, until advection outruns the grid and every colour smears into
    /// one flat wash.
    pub fn clamp_speed(&mut self, limit: f32) {
        if limit.is_nan() || limit <= 0.0 {
            return;
        }
        let max_sq = limit * limit;
        for (u, v) in self.u.data.iter_mut().zip(&mut self.v.data) {
            let sq = *u * *u + *v * *v;
            if sq > max_sq {
                let k = limit / sq.sqrt();
                *u *= k;
                *v *= k;
            }
        }
    }

    pub fn max_speed(&self) -> f32 {
        self.u
            .data
            .iter()
            .zip(&self.v.data)
            .fold(0.0f32, |m, (u, v)| m.max((u * u + v * v).sqrt()))
    }

    pub fn sanitize(&mut self) -> usize {
        self.u.sanitize() + self.v.sanitize()
    }
}
