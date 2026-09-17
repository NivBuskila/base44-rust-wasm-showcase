//! Scalar and vector grids, plus the sampling primitives the whole engine
//! shares.
//!
//! # Coordinate convention
//!
//! A grid value is defined **at the integer lattice point** `(i, j)`, so a
//! `w * h` grid spans `[0, w - 1] x [0, h - 1]` in grid space. [`Grid::sample`]
//! bilinearly interpolates and clamps outside that box — clamping rather than
//! wrapping is what keeps semi-Lagrangian advection stable at the borders.
//!
//! Every module in this crate works in grid space; only the engine boundary
//! converts to and from normalised `[0, 1]` screen space.

use crate::math::lerp;

/// A single-channel float field.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Grid {
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 2 && h >= 2, "grid must be at least 2x2, got {w}x{h}");
        Self {
            w,
            h,
            data: vec![0.0; w * h],
        }
    }

    pub fn filled(w: usize, h: usize, value: f32) -> Self {
        let mut g = Self::new(w, h);
        g.data.fill(value);
        g
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    /// Reads a lattice point, clamping the indices into range.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        let x = x.min(self.w - 1);
        let y = y.min(self.h - 1);
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        if x < self.w && y < self.h {
            self.data[y * self.w + x] = v;
        }
    }

    #[inline]
    pub fn add(&mut self, x: usize, y: usize, v: f32) {
        if x < self.w && y < self.h {
            self.data[y * self.w + x] += v;
        }
    }

    /// Bilinear sample in grid space, clamped at the borders.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        // NaN-safe: clamp() with a NaN input yields the lower bound here
        // because the comparisons all fail, which beats indexing out of range.
        let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, (self.w - 1) as f32) };
        let y = if y.is_nan() { 0.0 } else { y.clamp(0.0, (self.h - 1) as f32) };

        let x0 = x.floor();
        let y0 = y.floor();
        let tx = x - x0;
        let ty = y - y0;

        let ix0 = x0 as usize;
        let iy0 = y0 as usize;
        let ix1 = (ix0 + 1).min(self.w - 1);
        let iy1 = (iy0 + 1).min(self.h - 1);

        let row0 = iy0 * self.w;
        let row1 = iy1 * self.w;
        let top = lerp(self.data[row0 + ix0], self.data[row0 + ix1], tx);
        let bottom = lerp(self.data[row1 + ix0], self.data[row1 + ix1], tx);
        lerp(top, bottom, ty)
    }

    /// Adds a smooth radial blob centred at `(cx, cy)` in grid space.
    ///
    /// The falloff is `exp(-d^2 / r^2 * 3)`, i.e. a Gaussian truncated at
    /// `radius`, so splats stay local and the loop stays O(radius^2).
    pub fn splat(&mut self, cx: f32, cy: f32, radius: f32, amount: f32) {
        if radius <= 0.0 || !amount.is_finite() || !cx.is_finite() || !cy.is_finite() {
            return;
        }
        let r = radius.max(0.5);
        let inv_r2 = 3.0 / (r * r);

        let x0 = ((cx - r).floor().max(0.0)) as usize;
        let y0 = ((cy - r).floor().max(0.0)) as usize;
        let x1 = ((cx + r).ceil().min((self.w - 1) as f32)) as usize;
        let y1 = ((cy + r).ceil().min((self.h - 1) as f32)) as usize;
        if x0 > x1 || y0 > y1 {
            return;
        }

        for y in y0..=y1 {
            let dy = y as f32 - cy;
            let row = y * self.w;
            for x in x0..=x1 {
                let dx = x as f32 - cx;
                let d2 = dx * dx + dy * dy;
                if d2 > r * r {
                    continue;
                }
                self.data[row + x] += amount * (-d2 * inv_r2).exp();
            }
        }
    }

    /// Bilinearly rescales `src` into `self`, whatever the two resolutions are.
    pub fn resample_from(&mut self, src: &Grid) {
        if src.w == self.w && src.h == self.h {
            self.data.copy_from_slice(&src.data);
            return;
        }
        let sx = (src.w - 1) as f32 / (self.w - 1) as f32;
        let sy = (src.h - 1) as f32 / (self.h - 1) as f32;
        for y in 0..self.h {
            let fy = y as f32 * sy;
            let row = y * self.w;
            for x in 0..self.w {
                self.data[row + x] = src.sample(x as f32 * sx, fy);
            }
        }
    }

    /// Fills the grid from an 8-bit plane of the same dimensions, scaling to
    /// `[0, 1]`. Used for the camera luma plane and segmentation masks.
    pub fn fill_from_u8(&mut self, src: &[u8]) {
        assert_eq!(src.len(), self.data.len(), "u8 plane size mismatch");
        for (dst, &s) in self.data.iter_mut().zip(src) {
            *dst = s as f32 * (1.0 / 255.0);
        }
    }

    /// Separable binomial blur, `passes` applications of a `[1, 2, 1] / 4`
    /// kernel per axis. `scratch` must have the same dimensions as `self`.
    ///
    /// Cheap, unconditionally stable, and mass-preserving in the interior —
    /// which is all the obstacle and mask smoothing needs.
    pub fn blur(&mut self, scratch: &mut Grid, passes: usize) {
        assert_eq!(
            (scratch.w, scratch.h),
            (self.w, self.h),
            "blur scratch must match grid dimensions"
        );
        for _ in 0..passes {
            // Horizontal pass: self -> scratch.
            for y in 0..self.h {
                let row = y * self.w;
                for x in 0..self.w {
                    let l = self.data[row + x.saturating_sub(1)];
                    let c = self.data[row + x];
                    let r = self.data[row + (x + 1).min(self.w - 1)];
                    scratch.data[row + x] = 0.25 * l + 0.5 * c + 0.25 * r;
                }
            }
            // Vertical pass: scratch -> self.
            for y in 0..self.h {
                let row = y * self.w;
                let up = y.saturating_sub(1) * self.w;
                let down = (y + 1).min(self.h - 1) * self.w;
                for x in 0..self.w {
                    let u = scratch.data[up + x];
                    let c = scratch.data[row + x];
                    let d = scratch.data[down + x];
                    self.data[row + x] = 0.25 * u + 0.5 * c + 0.25 * d;
                }
            }
        }
    }

    /// Central-difference gradient at a lattice point, one-sided at the border.
    #[inline]
    pub fn gradient(&self, x: usize, y: usize) -> (f32, f32) {
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(self.w - 1);
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(self.h - 1);
        let dx = (self.get(xp, y) - self.get(xm, y)) / (xp - xm).max(1) as f32;
        let dy = (self.get(x, yp) - self.get(x, ym)) / (yp - ym).max(1) as f32;
        (dx, dy)
    }

    /// Multiplies every cell by `k`.
    #[inline]
    pub fn scale(&mut self, k: f32) {
        for v in &mut self.data {
            *v *= k;
        }
    }

    /// Sum of all cells; used by tests and the HUD energy readout.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Largest absolute value in the grid — the cheap "is this diverging?" check.
    pub fn max_abs(&self) -> f32 {
        self.data.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// Replaces any non-finite cell with zero and reports how many it fixed.
    ///
    /// The simulation should never produce NaN, but one escaped NaN otherwise
    /// spreads through the whole field within a few frames and blanks the
    /// screen, so the engine scrubs defensively once per frame.
    pub fn sanitize(&mut self) -> usize {
        let mut fixed = 0;
        for v in &mut self.data {
            if !v.is_finite() {
                *v = 0.0;
                fixed += 1;
            }
        }
        fixed
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_reproduces_lattice_values() {
        let mut g = Grid::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                g.set(x, y, (x * 10 + y) as f32);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                let s = g.sample(x as f32, y as f32);
                assert!(
                    (s - (x * 10 + y) as f32).abs() < 1e-4,
                    "lattice mismatch at {x},{y}: {s}"
                );
            }
        }
    }

    #[test]
    fn sample_interpolates_linearly() {
        let mut g = Grid::new(4, 4);
        g.set(0, 0, 0.0);
        g.set(1, 0, 10.0);
        assert!((g.sample(0.5, 0.0) - 5.0).abs() < 1e-5);
        assert!((g.sample(0.25, 0.0) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn sample_clamps_outside_the_grid() {
        let mut g = Grid::new(4, 4);
        g.set(0, 0, 3.0);
        g.set(3, 3, 7.0);
        assert!((g.sample(-100.0, -100.0) - 3.0).abs() < 1e-6);
        assert!((g.sample(1e6, 1e6) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn sample_survives_nan() {
        let mut g = Grid::new(4, 4);
        g.set(0, 0, 1.0);
        // Must not panic, must not return NaN.
        assert!(g.sample(f32::NAN, f32::NAN).is_finite());
        assert!(g.sample(f32::INFINITY, 0.0).is_finite());
    }

    #[test]
    fn splat_is_centred_and_local() {
        let mut g = Grid::new(32, 32);
        g.splat(16.0, 16.0, 4.0, 1.0);
        // Peak at the centre.
        assert!(g.get(16, 16) > g.get(18, 16));
        assert!(g.get(18, 16) > g.get(20, 16));
        // Nothing leaked far away.
        assert_eq!(g.get(0, 0), 0.0);
        assert_eq!(g.get(31, 31), 0.0);
        assert!(g.sum() > 0.0);
    }

    #[test]
    fn splat_at_the_border_does_not_panic() {
        let mut g = Grid::new(16, 16);
        g.splat(0.0, 0.0, 6.0, 1.0);
        g.splat(15.0, 15.0, 6.0, 1.0);
        g.splat(-50.0, 100.0, 6.0, 1.0);
        assert!(g.get(0, 0) > 0.0);
    }

    #[test]
    fn splat_ignores_non_finite_input() {
        let mut g = Grid::new(8, 8);
        g.splat(f32::NAN, 4.0, 2.0, 1.0);
        g.splat(4.0, 4.0, 2.0, f32::INFINITY);
        assert_eq!(g.sanitize(), 0, "no NaN should have been written");
        assert_eq!(g.sum(), 0.0);
    }

    #[test]
    fn resample_preserves_a_constant_field() {
        let src = Grid::filled(16, 9, 0.75);
        let mut dst = Grid::new(64, 36);
        dst.resample_from(&src);
        for v in &dst.data {
            assert!((v - 0.75).abs() < 1e-5);
        }
    }

    #[test]
    fn resample_preserves_a_linear_ramp() {
        let mut src = Grid::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                src.set(x, y, x as f32 / 15.0);
            }
        }
        let mut dst = Grid::new(31, 31);
        dst.resample_from(&src);
        // A bilinear resample must reproduce a linear ramp exactly.
        for y in 0..31 {
            for x in 0..31 {
                let expect = x as f32 / 30.0;
                assert!(
                    (dst.get(x, y) - expect).abs() < 1e-4,
                    "ramp broken at {x},{y}: {} vs {expect}",
                    dst.get(x, y)
                );
            }
        }
    }

    #[test]
    fn resample_same_size_is_a_copy() {
        let mut src = Grid::new(8, 8);
        src.set(3, 4, 9.0);
        let mut dst = Grid::new(8, 8);
        dst.resample_from(&src);
        assert_eq!(src, dst);
    }

    #[test]
    fn blur_spreads_and_conserves_interior_mass() {
        let mut g = Grid::new(32, 32);
        g.set(16, 16, 1.0);
        let before = g.sum();
        let mut scratch = Grid::new(32, 32);
        g.blur(&mut scratch, 3);
        assert!((g.sum() - before).abs() < 1e-4, "mass changed");
        assert!(g.get(16, 16) < 1.0, "peak should have flattened");
        assert!(g.get(17, 16) > 0.0, "should have spread to the neighbour");
    }

    #[test]
    fn blur_leaves_a_constant_field_alone() {
        let mut g = Grid::filled(16, 16, 0.4);
        let mut scratch = Grid::new(16, 16);
        g.blur(&mut scratch, 5);
        for v in &g.data {
            assert!((v - 0.4).abs() < 1e-6, "constant field changed: {v}");
        }
    }

    #[test]
    fn gradient_matches_an_analytic_ramp() {
        let mut g = Grid::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                g.set(x, y, 2.0 * x as f32 + 3.0 * y as f32);
            }
        }
        let (dx, dy) = g.gradient(8, 8);
        assert!((dx - 2.0).abs() < 1e-4, "dx {dx}");
        assert!((dy - 3.0).abs() < 1e-4, "dy {dy}");
    }

    #[test]
    fn sanitize_scrubs_non_finite_cells() {
        let mut g = Grid::new(4, 4);
        g.data[0] = f32::NAN;
        g.data[1] = f32::INFINITY;
        g.data[2] = 1.0;
        assert_eq!(g.sanitize(), 2);
        assert_eq!(g.data[0], 0.0);
        assert_eq!(g.data[1], 0.0);
        assert_eq!(g.data[2], 1.0);
    }

    #[test]
    fn fill_from_u8_normalizes() {
        let mut g = Grid::new(2, 2);
        g.fill_from_u8(&[0, 255, 128, 64]);
        assert_eq!(g.data[0], 0.0);
        assert_eq!(g.data[1], 1.0);
        assert!((g.data[2] - 0.5019608).abs() < 1e-6);
    }

    #[test]
    fn vecfield_energy_is_zero_at_rest() {
        let f = VecField::new(8, 8);
        assert_eq!(f.energy(), 0.0);
        assert_eq!(f.max_speed(), 0.0);
    }

    #[test]
    fn vecfield_energy_matches_hand_calculation() {
        let mut f = VecField::new(2, 2);
        f.u.data.fill(3.0);
        f.v.data.fill(4.0);
        // 0.5 * (9 + 16) = 12.5 per cell.
        assert!((f.energy() - 12.5).abs() < 1e-5);
        assert!((f.max_speed() - 5.0).abs() < 1e-5);
    }
}
