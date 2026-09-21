//! The scalar grid: storage, indexing, sampling, splats, resampling, blur and
//! the sanity scrubs every other module builds on.

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
    ///
    /// This is the hottest function in the engine — MacCormack advection alone
    /// calls it several million times per frame — so both the bounds handling
    /// and the integer split are written for the instruction count.
    ///
    /// `f32::max` and `f32::min` ignore NaN and return the other operand, so
    /// `x.max(0.0).min(hi)` maps NaN to 0, `-inf` to 0 and `+inf` to `hi` in
    /// two instructions, where an explicit `is_nan` test costs a
    /// mispredictable branch on every call.
    ///
    /// The integer part comes from truncation rather than `floor`. They agree
    /// for a non-negative argument, which the clamp above guarantees, and
    /// `f32::floor` compiles to a *libm call* on the SSE2 baseline this crate
    /// targets — there is no `roundss` without SSE4.1. (wasm32 has a native
    /// `f32.floor`, so this is a native-only win, but it is free either way.)
    #[inline(always)]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x = x.max(0.0).min((self.w - 1) as f32);
        let y = y.max(0.0).min((self.h - 1) as f32);

        let ix0 = x as usize;
        let iy0 = y as usize;
        let tx = x - ix0 as f32;
        let ty = y - iy0 as f32;

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
