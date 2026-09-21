//! The Horn–Schunck solve itself: the coarse-to-fine sweep, the linearised
//! system each level is built from, the Jacobi iteration, and the deadband
//! that turns cells per frame into the published cells per second.

use super::pyramid::prolong;
use super::work::Work;
use super::OpticalFlow;
use super::{BORDER_FADE, FLOW_SMOOTH, INV_LUMA8, MAX_FLOW, MAX_STEP, MIN_ALPHA2};
use crate::field::Grid;

impl OpticalFlow {
    /// Coarse-to-fine Horn–Schunck. Leaves the full-resolution answer, in
    /// cells per frame, in `work[0].u` / `work[0].v`.
    pub(super) fn solve(&mut self) {
        let n = self.work.len();
        // See `INV_LUMA8`: without this conversion `alpha^2` (144) swamps
        // `Ix^2 + Iy^2` (~1e-2 on normalised luma) by four orders of
        // magnitude, every update is a no-op and the flow field reads zero.
        let alpha = if self.cfg.alpha.is_finite() {
            self.cfg.alpha.abs() * INV_LUMA8
        } else {
            0.0
        };
        let alpha2 = (alpha * alpha).max(MIN_ALPHA2);
        let iters = self.cfg.iters.clamp(1, 64);

        for k in (0..n).rev() {
            if k + 1 == n {
                self.work[k].u.zero();
                self.work[k].v.zero();
            } else {
                let (fine, coarse) = self.work.split_at_mut(k + 1);
                prolong(&mut fine[k], &coarse[0]);
            }
            let prev = &self.prev_pyr[k];
            let cur = &self.cur_pyr[k];
            let work = &mut self.work[k];
            linearise(prev, cur, work);
            jacobi(work, alpha2, iters);
            work.u.blur(&mut work.tmp, FLOW_SMOOTH);
            work.v.blur(&mut work.tmp, FLOW_SMOOTH);
        }
    }

    /// Deadbands, rescales to cells/second and publishes the statistics.
    pub(super) fn finish(&mut self, dt: f32) {
        let floor = if self.cfg.noise_floor.is_finite() {
            self.cfg.noise_floor.max(0.0)
        } else {
            0.0
        };
        let inv_dt = 1.0 / dt;
        let src = &self.work[0];
        let flow = &mut self.flow;

        // f64 accumulators: the per-second magnitudes square up to ~1e5 and a
        // 9k-cell f32 sum of those loses enough precision to make the mean
        // flow readout visibly jitter.
        let mut sum_u = 0.0f64;
        let mut sum_v = 0.0f64;
        let mut sum_sq = 0.0f64;

        let (w, h) = (flow.u.w, flow.u.h);
        for y in 0..h {
            let fade_y = border_fade(y, h);
            for x in 0..w {
                let i = y * w + x;
                let fade = fade_y * border_fade(x, w);
                let mut u = src.u.data[i] * fade;
                let mut v = src.v.data[i] * fade;
                if !u.is_finite() || !v.is_finite() {
                    u = 0.0;
                    v = 0.0;
                }
                let mag = (u * u + v * v).sqrt();
                // A gain that ramps 0 -> 1 across the deadband, rather than
                // subtracting the floor from the magnitude. Both keep the field
                // continuous, but subtracting leaves a residual bias of
                // `floor / dt` in the published cells/second, and that term
                // grows with the frame rate: at the default 0.06 cells/frame it
                // costs 1.8 cells/s at 30 fps and 8.6 cells/s at 144 fps, so the
                // same physical motion reads differently on different hardware.
                // A unity-gain-above-the-band gate rejects exactly the same
                // per-frame noise without touching motion the sensor resolved.
                let gain = crate::math::smoothstep(floor, floor * 2.0, mag);
                let (fu, fv) = if gain > 0.0 && mag > 1e-12 {
                    let per_frame = (mag * gain).min(MAX_STEP);
                    let scale = (per_frame * inv_dt).min(MAX_FLOW) / mag;
                    (u * scale, v * scale)
                } else {
                    (0.0, 0.0)
                };
                flow.u.data[i] = fu;
                flow.v.data[i] = fv;
                sum_u += fu as f64;
                sum_v += fv as f64;
                sum_sq += (fu as f64) * (fu as f64) + (fv as f64) * (fv as f64);
            }
        }

        let inv_n = 1.0 / flow.u.data.len() as f64;
        self.motion_energy = (sum_sq * inv_n) as f32;
        self.dominant = ((sum_u * inv_n) as f32, (sum_v * inv_n) as f32);
    }
}

/// Weight in `[0, 1]` that fades the flow out over [`BORDER_FADE`] cells at
/// both ends of an axis of length `n`.
#[inline]
fn border_fade(i: usize, n: usize) -> f32 {
    let d = i.min(n - 1 - i) as f32;
    crate::math::smoothstep(0.0, BORDER_FADE, d)
}

/// Warps `cur` by the initial guess and builds the linearised system.
///
/// Brightness constancy is `cur(x + u, y + v) = prev(x, y)`. Linearising about
/// the initial guess `(u0, v0)` and rearranging into the plain Horn–Schunck
/// form `Ix*u + Iy*v + It = 0` for the **total** flow gives
/// `It = warp - prev - (Ix*u0 + Iy*v0)`, which is why the residual is not
/// simply `cur - prev` on every level but the coarsest.
fn linearise(prev: &Grid, cur: &Grid, w: &mut Work) {
    let (lw, lh) = (prev.w, prev.h);
    for y in 0..lh {
        let row = y * lw;
        for x in 0..lw {
            let i = row + x;
            let u0 = w.u.data[i];
            let v0 = w.v.data[i];
            w.warp.data[i] = cur.sample(x as f32 + u0, y as f32 + v0);
        }
    }
    for y in 0..lh {
        let row = y * lw;
        for x in 0..lw {
            let i = row + x;
            // Averaging the two frames' gradients centres the derivative in
            // time; using only one frame biases the estimate towards it and
            // shows up as a systematic magnitude error.
            let (px, py) = prev.gradient(x, y);
            let (wx, wy) = w.warp.gradient(x, y);
            let ix = 0.5 * (px + wx);
            let iy = 0.5 * (py + wy);
            w.ix.data[i] = ix;
            w.iy.data[i] = iy;
            w.it.data[i] = w.warp.data[i] - prev.data[i] - (ix * w.u.data[i] + iy * w.v.data[i]);
        }
    }
}

/// Jacobi sweeps of the Horn–Schunck update.
fn jacobi(w: &mut Work, alpha2: f32, iters: usize) {
    let (lw, lh) = (w.u.w, w.u.h);
    for _ in 0..iters {
        for y in 0..lh {
            let row = y * lw;
            let up = y.saturating_sub(1) * lw;
            let down = (y + 1).min(lh - 1) * lw;
            for x in 0..lw {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(lw - 1);
                let i = row + x;
                // Clamped neighbours = Neumann boundary: the flow is free to
                // slide along the frame edge instead of being pinned to zero.
                let ubar = 0.25
                    * (w.u.data[row + xm]
                        + w.u.data[row + xp]
                        + w.u.data[up + x]
                        + w.u.data[down + x]);
                let vbar = 0.25
                    * (w.v.data[row + xm]
                        + w.v.data[row + xp]
                        + w.v.data[up + x]
                        + w.v.data[down + x]);
                let ix = w.ix.data[i];
                let iy = w.iy.data[i];
                // `alpha2 >= MIN_ALPHA2`, so a gradient-free (flat) region
                // divides by alpha2 and simply inherits the local average.
                let den = alpha2 + ix * ix + iy * iy;
                let k = (ix * ubar + iy * vbar + w.it.data[i]) / den;
                w.un.data[i] = ubar - ix * k;
                w.vn.data[i] = vbar - iy * k;
            }
        }
        core::mem::swap(&mut w.u, &mut w.un);
        core::mem::swap(&mut w.v, &mut w.vn);
    }
}
