//! Dense optical flow in pure Rust — no model, no inference, no network.
//!
//! This is the engine's independent perception path: even with every MediaPipe
//! model disabled or failed to load, Aether still reacts to whatever moves in
//! front of the camera, because motion is measured straight from the pixels.
//!
//! ## Contract
//!
//! Multi-scale Horn–Schunck: build a Gaussian pyramid of the previous and
//! current luma frames, solve for the smooth flow field coarse-to-fine,
//! prolonging each level's solution as the next level's initial guess.
//!
//! [`OpticalFlow::flow`] is in **grid cells per second** (not per frame), so
//! the caller can scale it by force without caring about the frame rate.
//! [`OpticalFlow::update`] must be robust to a duplicate frame (flow -> 0) and
//! to the very first frame (no history -> flow 0, `has_history() == false`).
//!
//! ## Sign convention
//!
//! A positive `u` means image content moved towards +x (to the right in grid
//! space) between `prev` and `cur`; positive `v` means it moved towards +y
//! (down a row). That is the same convention the fluid grid uses, so the flow
//! field can be sampled straight into a velocity impulse.

use crate::field::{Grid, VecField};

/// Solver settings.
#[derive(Clone, Copy, Debug)]
pub struct FlowConfig {
    /// Pyramid levels, including the full-resolution one.
    pub levels: usize,
    /// Horn–Schunck iterations per level.
    pub iters: usize,
    /// Smoothness weight: higher is smoother and more diffuse.
    pub alpha: f32,
    /// Flow magnitudes below this (cells/frame) are treated as sensor noise.
    pub noise_floor: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            iters: 8,
            alpha: 12.0,
            noise_floor: 0.06,
        }
    }
}

/// Hard ceiling on pyramid depth, so a bad config cannot allocate forever.
const MAX_LEVELS: usize = 6;
/// Below this, a level has too few samples left for a central difference to
/// mean anything, so the pyramid stops there regardless of `cfg.levels`.
const MIN_LEVEL_DIM: usize = 4;
/// Prefilter on the finest level. Central differences of raw sensor noise are
/// garbage, and one binomial pass costs almost nothing.
const PRE_BLUR: usize = 1;
/// Two binomial passes is the classic `[1, 4, 6, 4, 1] / 16` pyramid kernel.
/// Decimating with less than that aliases the texture and the coarse levels
/// end up tracking nothing.
const DOWN_BLUR: usize = 2;
/// `alpha` is quoted in 8-bit luma units — the scale every Horn–Schunck
/// reference is written in — while the pyramid holds `[0, 1]` samples.
const INV_LUMA8: f32 = 1.0 / 255.0;
/// Floor on `alpha^2`. Nothing here divides by zero with the default config,
/// but `alpha = 0` plus a gradient-free region would, and a NaN in the flow
/// field reaches the fluid and blanks the screen.
const MIN_ALPHA2: f32 = 1e-6;
/// Displacement ceiling in cells/frame, applied before the `1 / dt` scaling.
/// The coarsest level can only resolve a couple of cells of motion, which is
/// ~8 at full resolution, so anything past this is a solver artefact and not
/// something the pyramid actually measured.
const MAX_STEP: f32 = 8.0;
/// Ceiling on the reported flow, in cells/second. One dropped or torn camera
/// frame must not be able to inject an unbounded impulse into the fluid.
const MAX_FLOW: f32 = 400.0;
/// Binomial passes applied to each level's solution before it is prolonged or
/// published. The smoothness term only couples four neighbours per sweep, so a
/// handful of Jacobi iterations cannot damp an isolated spike in a cell whose
/// gradient is near zero — this is the cheap stand-in for the median filter
/// modern Horn–Schunck descendants rely on.
const FLOW_SMOOTH: usize = 1;
/// Cells of fade applied at each frame border. Content entering or leaving the
/// frame has no counterpart in the other frame, so the solve is unconstrained
/// there and habitually spikes; without the fade a camera pan paints a
/// permanent wall of force down the edge of the fluid.
const BORDER_FADE: f32 = 3.0;

/// Per-level scratch. Allocated once and reused: rebuilding a dozen `Grid`s
/// every camera frame would churn the wasm heap at 30 Hz for no reason.
struct Work {
    /// Current flow estimate at this level, in cells per frame.
    u: Grid,
    v: Grid,
    /// Jacobi ping-pong targets.
    un: Grid,
    vn: Grid,
    /// Spatial derivatives, averaged between `prev` and the warped `cur`.
    ix: Grid,
    iy: Grid,
    /// Temporal residual, linearised about the initial guess.
    it: Grid,
    /// `cur` warped by the initial guess.
    warp: Grid,
    /// Blurred copy of this level, the source of the next coarser one.
    smooth: Grid,
    /// Blur scratch.
    tmp: Grid,
}

impl Work {
    fn new(w: usize, h: usize) -> Self {
        Self {
            u: Grid::new(w, h),
            v: Grid::new(w, h),
            un: Grid::new(w, h),
            vn: Grid::new(w, h),
            ix: Grid::new(w, h),
            iy: Grid::new(w, h),
            it: Grid::new(w, h),
            warp: Grid::new(w, h),
            smooth: Grid::new(w, h),
            tmp: Grid::new(w, h),
        }
    }
}

/// Dense motion estimator over successive camera luma planes.
pub struct OpticalFlow {
    w: usize,
    h: usize,
    cfg: FlowConfig,
    prev: Grid,
    cur: Grid,
    flow: VecField,
    /// Gaussian pyramids, index 0 = full resolution.
    prev_pyr: Vec<Grid>,
    cur_pyr: Vec<Grid>,
    work: Vec<Work>,
    /// False until `prev_pyr` is known to be the pyramid of `prev`.
    pyr_ready: bool,
    has_history: bool,
    motion_energy: f32,
    dominant: (f32, f32),
}

impl OpticalFlow {
    /// `w` and `h` are clamped up to the smallest grid the sampler supports.
    pub fn new(w: usize, h: usize, cfg: FlowConfig) -> Self {
        let w = w.max(2);
        let h = h.max(2);
        let mut me = Self {
            w,
            h,
            cfg,
            prev: Grid::new(w, h),
            cur: Grid::new(w, h),
            flow: VecField::new(w, h),
            prev_pyr: Vec::new(),
            cur_pyr: Vec::new(),
            work: Vec::new(),
            pyr_ready: false,
            has_history: false,
            motion_energy: 0.0,
            dominant: (0.0, 0.0),
        };
        me.ensure_levels();
        me
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    #[inline]
    pub fn config(&self) -> &FlowConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: FlowConfig) {
        self.cfg = cfg;
        self.ensure_levels();
    }

    /// Flow field in grid cells per second.
    ///
    /// "Cells" here are cells of *this* grid ([`FLOW_W` x `FLOW_H`][crate::config],
    /// 128x72), not of the fluid grid: a consumer that resamples the field onto
    /// the 256x144 fluid needs to scale the vectors by the resolution ratio for
    /// physically exact velocities. `params.flow_force` absorbs that constant
    /// today, which is why the drive path in [`crate::spells`] samples the field
    /// without rescaling it.
    #[inline]
    pub fn flow(&self) -> &VecField {
        &self.flow
    }

    /// Mean squared flow magnitude — the "how much is happening" scalar the
    /// visuals use to scale global intensity. In `(cells/second)^2`, i.e. the
    /// same units as [`OpticalFlow::flow`], squared.
    #[inline]
    pub fn motion_energy(&self) -> f32 {
        self.motion_energy
    }

    /// Average flow direction and magnitude over the frame.
    #[inline]
    pub fn dominant_motion(&self) -> (f32, f32) {
        self.dominant
    }

    #[inline]
    pub fn has_history(&self) -> bool {
        self.has_history
    }

    /// Pyramid levels actually in use, which is [`FlowConfig::levels`] capped
    /// by how far this resolution can usefully be decimated.
    #[inline]
    pub fn levels(&self) -> usize {
        self.work.len()
    }

    pub fn reset(&mut self) {
        self.prev.zero();
        self.cur.zero();
        self.flow.zero();
        // The cached pyramid describes frames that no longer exist.
        self.pyr_ready = false;
        self.has_history = false;
        self.motion_energy = 0.0;
        self.dominant = (0.0, 0.0);
    }

    /// Feeds a new `w * h` luma plane and recomputes the flow.
    ///
    /// `dt` is the time since the previous plane, in seconds; it is clamped
    /// into a sane range rather than trusted, because a tab that was
    /// backgrounded hands back whatever it likes.
    pub fn update(&mut self, luma: &[u8], dt: f32) {
        assert_eq!(luma.len(), self.w * self.h, "luma plane size mismatch");

        core::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.fill_from_u8(luma);

        self.ensure_levels();
        if self.pyr_ready {
            // Last frame's `cur` pyramid *is* this frame's `prev` pyramid, so
            // only one of the two is ever built per update.
            core::mem::swap(&mut self.prev_pyr, &mut self.cur_pyr);
            build_pyramid(&self.cur, &mut self.cur_pyr, &mut self.work);
        } else {
            build_pyramid(&self.prev, &mut self.prev_pyr, &mut self.work);
            build_pyramid(&self.cur, &mut self.cur_pyr, &mut self.work);
            self.pyr_ready = true;
        }

        if !self.has_history {
            // One frame is not motion. Report stillness and keep the pyramid
            // so the next call has something to difference against.
            self.has_history = true;
            self.flow.zero();
            self.motion_energy = 0.0;
            self.dominant = (0.0, 0.0);
            return;
        }

        self.solve();
        self.finish(sane_dt(dt));
    }

    /// Reallocates the pyramid if the effective level count changed.
    fn ensure_levels(&mut self) {
        let n = effective_levels(self.w, self.h, self.cfg.levels);
        if self.work.len() == n {
            return;
        }
        self.prev_pyr.clear();
        self.cur_pyr.clear();
        self.work.clear();
        for k in 0..n {
            let lw = (self.w >> k).max(2);
            let lh = (self.h >> k).max(2);
            self.prev_pyr.push(Grid::new(lw, lh));
            self.cur_pyr.push(Grid::new(lw, lh));
            self.work.push(Work::new(lw, lh));
        }
        self.pyr_ready = false;
    }

    /// Coarse-to-fine Horn–Schunck. Leaves the full-resolution answer, in
    /// cells per frame, in `work[0].u` / `work[0].v`.
    fn solve(&mut self) {
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
    fn finish(&mut self, dt: f32) {
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
                let (fu, fv) = if mag > floor {
                    // Soft threshold rather than a hard gate: subtracting the
                    // floor keeps the field continuous, so a hand drifting
                    // through the deadband does not pop the fluid.
                    let per_frame = (mag - floor).min(MAX_STEP);
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

/// `cfg.levels`, capped so the coarsest level keeps usable gradients.
fn effective_levels(w: usize, h: usize, requested: usize) -> usize {
    let mut n = requested.clamp(1, MAX_LEVELS);
    while n > 1 {
        let k = n - 1;
        if (w >> k) >= MIN_LEVEL_DIM && (h >> k) >= MIN_LEVEL_DIM {
            break;
        }
        n -= 1;
    }
    n
}

/// Fills `pyr` (index 0 = full resolution) with a Gaussian pyramid of `src`.
fn build_pyramid(src: &Grid, pyr: &mut [Grid], work: &mut [Work]) {
    pyr[0].resample_from(src);
    pyr[0].blur(&mut work[0].tmp, PRE_BLUR);
    for k in 1..pyr.len() {
        // Blur a copy, not the level itself: the unsmoothed level is the image
        // its own solve runs on, and over-filtering it would throw away the
        // fine detail that is the whole point of having a finest level.
        let src_level = {
            let w = &mut work[k - 1];
            w.smooth.data.copy_from_slice(&pyr[k - 1].data);
            w.smooth.blur(&mut w.tmp, DOWN_BLUR);
            &w.smooth
        };
        pyr[k].resample_from(src_level);
    }
}

/// Upsamples the coarse solution into `fine` as its initial guess.
fn prolong(fine: &mut Work, coarse: &Work) {
    // The scale factor is the classic bug here: a displacement of one coarse
    // cell is `(fine - 1) / (coarse - 1)` fine cells, and `resample_from` maps
    // the lattices with exactly that ratio. Drop it and every level halves the
    // motion it inherited.
    let sx = (fine.u.w - 1) as f32 / (coarse.u.w - 1) as f32;
    let sy = (fine.u.h - 1) as f32 / (coarse.u.h - 1) as f32;
    fine.u.resample_from(&coarse.u);
    fine.u.scale(sx);
    fine.v.resample_from(&coarse.v);
    fine.v.scale(sy);
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
                    * (w.u.data[row + xm] + w.u.data[row + xp] + w.u.data[up + x] + w.u.data[down + x]);
                let vbar = 0.25
                    * (w.v.data[row + xm] + w.v.data[row + xp] + w.v.data[up + x] + w.v.data[down + x]);
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

/// Clamps a frame interval into a range where `1 / dt` is meaningful.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 1000.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use core::f32::consts::TAU;

    /// Deterministic band-limited test texture.
    ///
    /// Sampled analytically so a translated or zoomed copy is exact instead of
    /// resampled, and built from one wave per octave so that *every* pyramid
    /// level still has real gradients: a single high-frequency octave is gone
    /// by the second decimation and the coarse solve would have nothing to
    /// lock onto, which makes a "flow is nonzero" assertion vacuous.
    struct Texture {
        waves: [(f32, f32, f32, f32); 3],
        norm: f32,
    }

    impl Texture {
        fn new(seed: u64) -> Self {
            let mut rng = Rng::new(seed);
            let mut waves = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 3];
            // Wavelengths in full-resolution cells; the longest survives three
            // decimations, the shortest gives the finest level something to do.
            let spec = [(30.0f32, 0.45f32), (14.0, 0.30), (8.0, 0.25)];
            // The seed picks the overall orientation, but the three waves are
            // then spread evenly: a texture whose gradients all point the same
            // way leaves the flow component across them unobservable (the
            // aperture problem), and the ground truth would be unrecoverable
            // for reasons that have nothing to do with the solver.
            let base = rng.next_f32() * TAU;
            for (i, (slot, &(wavelength, amp))) in waves.iter_mut().zip(spec.iter()).enumerate() {
                let theta = base + i as f32 * TAU / 3.0;
                let k = TAU / wavelength;
                *slot = (k * theta.cos(), k * theta.sin(), rng.next_f32() * TAU, amp);
            }
            let norm = waves.iter().map(|w| w.3.abs()).sum::<f32>().max(1e-6);
            Self { waves, norm }
        }

        fn at(&self, x: f32, y: f32) -> f32 {
            let s: f32 = self
                .waves
                .iter()
                .map(|&(kx, ky, phase, amp)| amp * (kx * x + ky * y + phase).sin())
                .sum();
            0.5 + 0.5 * s / self.norm
        }

        fn render(&self, w: usize, h: usize, map: impl Fn(f32, f32) -> (f32, f32)) -> Vec<u8> {
            let mut out = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = map(x as f32, y as f32);
                    out[y * w + x] = (self.at(sx, sy).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            out
        }

        fn still(&self, w: usize, h: usize) -> Vec<u8> {
            self.render(w, h, |x, y| (x, y))
        }

        /// Content shifted by `(dx, dy)` cells, so the true flow is `(dx, dy)`.
        fn translated(&self, w: usize, h: usize, dx: f32, dy: f32) -> Vec<u8> {
            self.render(w, h, |x, y| (x - dx, y - dy))
        }

        /// Content magnified by `s` about the frame centre: outward flow.
        fn zoomed(&self, w: usize, h: usize, s: f32) -> Vec<u8> {
            let cx = (w - 1) as f32 * 0.5;
            let cy = (h - 1) as f32 * 0.5;
            self.render(w, h, |x, y| (cx + (x - cx) / s, cy + (y - cy) / s))
        }
    }

    const W: usize = 128;
    const H: usize = 72;
    const DT: f32 = 1.0 / 30.0;

    fn flow_of(prev: &[u8], cur: &[u8], dt: f32) -> OpticalFlow {
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        f.update(prev, dt);
        f.update(cur, dt);
        f
    }

    /// Mean flow over the interior, away from the border where the clamped
    /// warp invents data that was never in frame.
    fn mean_interior(f: &VecField, margin: usize) -> (f32, f32) {
        let mut su = 0.0f64;
        let mut sv = 0.0f64;
        let mut n = 0u32;
        for y in margin..f.h() - margin {
            for x in margin..f.w() - margin {
                let (u, v) = f.get(x, y);
                su += u as f64;
                sv += v as f64;
                n += 1;
            }
        }
        let inv = 1.0 / n.max(1) as f64;
        ((su * inv) as f32, (sv * inv) as f32)
    }

    fn assert_recovers(prev: &[u8], cur: &[u8], want: (f32, f32)) {
        let f = flow_of(prev, cur, DT);
        let (u, v) = mean_interior(f.flow(), 16);
        let (wu, wv) = (want.0 / DT, want.1 / DT);
        let scale = (wu * wu + wv * wv).sqrt();
        assert!(
            (u - wu).abs() < 0.35 * scale,
            "u {u} should be near {wu} (v {v})"
        );
        assert!(
            (v - wv).abs() < 0.35 * scale,
            "v {v} should be near {wv} (u {u})"
        );
        // Sign is the part a magnitude-only test would never catch.
        if wu.abs() > 1.0 {
            assert!(u.signum() == wu.signum(), "u sign flipped: {u} vs {wu}");
        }
        if wv.abs() > 1.0 {
            assert!(v.signum() == wv.signum(), "v sign flipped: {v} vs {wv}");
        }
    }

    #[test]
    fn recovers_translation_plus_x() {
        let t = Texture::new(1);
        assert_recovers(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), (2.0, 0.0));
    }

    #[test]
    fn recovers_translation_minus_x() {
        let t = Texture::new(2);
        assert_recovers(&t.still(W, H), &t.translated(W, H, -2.0, 0.0), (-2.0, 0.0));
    }

    #[test]
    fn recovers_translation_plus_y() {
        let t = Texture::new(3);
        assert_recovers(&t.still(W, H), &t.translated(W, H, 0.0, 2.0), (0.0, 2.0));
    }

    #[test]
    fn recovers_translation_minus_y() {
        let t = Texture::new(4);
        assert_recovers(&t.still(W, H), &t.translated(W, H, 0.0, -2.0), (0.0, -2.0));
    }

    #[test]
    fn recovers_diagonal_translation() {
        let t = Texture::new(5);
        assert_recovers(&t.still(W, H), &t.translated(W, H, 3.0, -2.0), (3.0, -2.0));
    }

    #[test]
    fn dominant_motion_and_energy_track_a_translation() {
        let t = Texture::new(6);
        let f = flow_of(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), DT);
        let (dx, dy) = f.dominant_motion();
        assert!(dx > 0.5 * 2.0 / DT, "dominant dx too small: {dx}");
        assert!(dy.abs() < 0.35 * 2.0 / DT, "spurious dominant dy: {dy}");
        // Mean square of a field whose mean is ~dx can never be below dx^2.
        assert!(
            f.motion_energy() >= dx * dx * 0.9,
            "energy {} vs dx {dx}",
            f.motion_energy()
        );
    }

    #[test]
    fn zoom_produces_outward_flow() {
        let t = Texture::new(7);
        let f = flow_of(&t.still(W, H), &t.zoomed(W, H, 1.12), DT);
        let flow = f.flow();
        let (cx, cy) = ((W - 1) as f32 * 0.5, (H - 1) as f32 * 0.5);
        let mut radial = 0.0f64;
        let mut divergence = 0.0f64;
        let mut n = 0u32;
        for y in 12..H - 12 {
            for x in 12..W - 12 {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let r = (dx * dx + dy * dy).sqrt();
                if r < 8.0 {
                    continue; // Sub-pixel motion near the centre of dilation.
                }
                let (u, v) = flow.get(x, y);
                radial += ((u * dx + v * dy) / r) as f64;
                let (dudx, _) = flow.u.gradient(x, y);
                let (_, dvdy) = flow.v.gradient(x, y);
                divergence += (dudx + dvdy) as f64;
                n += 1;
            }
        }
        let n = n.max(1) as f64;
        assert!(radial / n > 1.0, "flow is not outward: {}", radial / n);
        assert!(divergence / n > 0.0, "divergence {}", divergence / n);
    }

    #[test]
    fn duplicate_frame_is_perfectly_still() {
        let t = Texture::new(8);
        let frame = t.still(W, H);
        let f = flow_of(&frame, &frame, DT);
        assert!(f.has_history());
        assert!(
            f.flow().max_speed() < 1e-6,
            "duplicate frame moved: {}",
            f.flow().max_speed()
        );
        assert!(f.motion_energy() < 1e-6);
        assert_eq!(f.dominant_motion().0, 0.0);
    }

    #[test]
    fn first_frame_has_no_history_and_no_flow() {
        let t = Texture::new(9);
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        assert!(!f.has_history());
        f.update(&t.still(W, H), DT);
        assert!(f.has_history(), "second frame should have something to diff");
        assert_eq!(f.flow().max_speed(), 0.0);
        assert_eq!(f.motion_energy(), 0.0);
    }

    #[test]
    fn a_static_scene_never_accumulates_drift() {
        let t = Texture::new(10);
        let frame = t.still(W, H);
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        for _ in 0..200 {
            f.update(&frame, DT);
            assert!(
                f.flow().max_speed() < 1e-5,
                "drifted to {}",
                f.flow().max_speed()
            );
        }
        assert_eq!(f.dominant_motion(), (0.0, 0.0));
    }

    #[test]
    fn flow_is_per_second_not_per_frame() {
        let t = Texture::new(11);
        let (a, b) = (t.still(W, H), t.translated(W, H, 2.0, 0.0));
        let slow = flow_of(&a, &b, 1.0 / 30.0);
        let fast = flow_of(&a, &b, 1.0 / 60.0);
        let (su, _) = mean_interior(slow.flow(), 16);
        let (fu, _) = mean_interior(fast.flow(), 16);
        assert!(su > 1.0, "baseline flow too small to compare: {su}");
        let ratio = fu / su;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "halving dt should double the flow, got {ratio}x"
        );
    }

    #[test]
    fn noise_floor_soft_thresholds_the_magnitude() {
        let t = Texture::new(12);
        let (a, b) = (t.still(W, H), t.translated(W, H, 2.0, 0.0));
        let open = {
            let cfg = FlowConfig {
                noise_floor: 0.0,
                ..Default::default()
            };
            let mut f = OpticalFlow::new(W, H, cfg);
            f.update(&a, DT);
            f.update(&b, DT);
            f
        };
        let floor = 1.0; // cells/frame, i.e. half the real motion
        let gated = {
            let cfg = FlowConfig {
                noise_floor: floor,
                ..Default::default()
            };
            let mut f = OpticalFlow::new(W, H, cfg);
            f.update(&a, DT);
            f.update(&b, DT);
            f
        };
        let mut gated_cells = 0;
        for i in 0..W * H {
            let (u0, v0) = (open.flow().u.data[i], open.flow().v.data[i]);
            let raw = (u0 * u0 + v0 * v0).sqrt() * DT;
            if raw >= MAX_STEP - 1e-3 {
                continue; // Clamped, so the deadband is not the only effect.
            }
            let (u1, v1) = (gated.flow().u.data[i], gated.flow().v.data[i]);
            let got = (u1 * u1 + v1 * v1).sqrt() * DT;
            let want = (raw - floor).max(0.0);
            assert!(
                (got - want).abs() < 1e-3,
                "cell {i}: raw {raw} with floor {floor} gave {got}, want {want}"
            );
            if want == 0.0 {
                gated_cells += 1;
            }
        }
        assert!(gated_cells > 0, "floor never engaged; test proves nothing");
    }

    #[test]
    fn sensor_noise_on_a_still_camera_does_not_drive_the_fluid() {
        // The reason the noise floor exists: a static scene through a real
        // sensor jitters by a couple of LSB every frame, and that must not
        // read as motion.
        let t = Texture::new(19);
        let base = t.still(W, H);
        let mut rng = Rng::new(20);
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        let moving = flow_of(&base, &t.translated(W, H, 2.0, 0.0), DT);
        for _ in 0..20 {
            let frame: Vec<u8> = base
                .iter()
                .map(|&b| (b as f32 + rng.signed() * 2.5).clamp(0.0, 255.0) as u8)
                .collect();
            f.update(&frame, DT);
            assert!(
                f.motion_energy() < 1.0,
                "sensor noise read as motion: {} (real motion is {})",
                f.motion_energy(),
                moving.motion_energy()
            );
            assert!(f.flow().max_speed() < 8.0, "{}", f.flow().max_speed());
        }
    }

    #[test]
    fn flat_image_produces_no_nan() {
        // The degenerate case: Ix = Iy = 0 everywhere, so the Horn-Schunck
        // denominator is alpha^2 alone and a zero alpha divides by zero.
        for alpha in [0.0, 12.0, f32::NAN] {
            let cfg = FlowConfig {
                alpha,
                ..Default::default()
            };
            let mut f = OpticalFlow::new(W, H, cfg);
            f.update(&vec![128; W * H], DT);
            f.update(&vec![130; W * H], DT);
            assert!(
                f.flow().u.data.iter().all(|v| v.is_finite()),
                "non-finite flow at alpha {alpha}"
            );
            assert!(f.flow().v.data.iter().all(|v| v.is_finite()));
            assert!(f.motion_energy().is_finite());
            assert!(f.flow().max_speed() < 1e-3, "flat image invented motion");
        }
    }

    #[test]
    fn uniform_noise_is_finite_and_bounded() {
        let mut rng = Rng::new(13);
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        for _ in 0..12 {
            let frame: Vec<u8> = (0..W * H).map(|_| (rng.next_f32() * 255.0) as u8).collect();
            f.update(&frame, DT);
            assert!(f.flow().u.data.iter().all(|v| v.is_finite()));
            assert!(f.flow().v.data.iter().all(|v| v.is_finite()));
            assert!(
                f.flow().max_speed() <= MAX_FLOW + 1e-3,
                "unbounded flow: {}",
                f.flow().max_speed()
            );
            assert!(f.motion_energy().is_finite());
        }
    }

    #[test]
    fn extreme_dt_stays_finite_and_clamped() {
        let t = Texture::new(14);
        let (a, b) = (t.still(W, H), t.translated(W, H, 3.0, 1.0));
        for dt in [f32::NAN, f32::INFINITY, 0.0, -1.0, 1e-9, 1e9] {
            let f = flow_of(&a, &b, dt);
            assert!(
                f.flow().max_speed() <= MAX_FLOW + 1e-3,
                "dt {dt} produced {}",
                f.flow().max_speed()
            );
            assert!(f.flow().u.data.iter().all(|v| v.is_finite()), "dt {dt}");
            assert!(f.motion_energy().is_finite(), "dt {dt}");
            // A negative or zero dt must never invert the direction of time.
            assert!(f.dominant_motion().0 >= 0.0, "dt {dt} flipped the sign");
        }
    }

    #[test]
    fn reset_clears_history_and_flow() {
        let t = Texture::new(15);
        let mut f = flow_of(&t.still(W, H), &t.translated(W, H, 2.0, 0.0), DT);
        assert!(f.flow().max_speed() > 0.0);
        f.reset();
        assert!(!f.has_history());
        assert_eq!(f.flow().max_speed(), 0.0);
        assert_eq!(f.motion_energy(), 0.0);
        assert_eq!(f.dominant_motion(), (0.0, 0.0));
        // The first frame after a reset is history again, not motion.
        f.update(&t.still(W, H), DT);
        assert!(f.has_history());
        assert_eq!(f.flow().max_speed(), 0.0);
    }

    #[test]
    fn level_count_is_capped_by_resolution() {
        assert_eq!(effective_levels(128, 72, 3), 3);
        // 72 >> 5 is 2 samples tall, so the 6th level is dropped.
        assert_eq!(effective_levels(128, 72, 99), 5);
        assert_eq!(effective_levels(1024, 1024, 99), MAX_LEVELS);
        assert_eq!(effective_levels(8, 8, 4), 2);
        assert_eq!(effective_levels(4, 4, 5), 1);
        assert_eq!(effective_levels(2, 2, 3), 1);
        assert_eq!(effective_levels(128, 72, 0), 1);
    }

    #[test]
    fn single_level_still_recovers_small_motion() {
        // Without a pyramid, Horn-Schunck only sees motion inside its own
        // aperture, so this checks the flat (non-multi-scale) path works too.
        let t = Texture::new(16);
        let cfg = FlowConfig {
            levels: 1,
            iters: 24,
            ..Default::default()
        };
        let mut f = OpticalFlow::new(W, H, cfg);
        f.update(&t.still(W, H), DT);
        f.update(&t.translated(W, H, 1.0, 0.0), DT);
        let (u, v) = mean_interior(f.flow(), 16);
        assert_eq!(f.levels(), 1);
        assert!(u > 0.4 / DT, "single level lost the motion: {u}");
        assert!(v.abs() < 0.35 / DT, "spurious v: {v}");
    }

    #[test]
    fn reconfiguring_mid_stream_stays_correct() {
        // Changing the level count invalidates the cached pyramid; if that
        // cache is not dropped the next frame differences mismatched images.
        let t = Texture::new(17);
        let mut f = OpticalFlow::new(W, H, FlowConfig::default());
        f.update(&t.still(W, H), DT);
        f.update(&t.translated(W, H, 2.0, 0.0), DT);
        f.set_config(FlowConfig {
            levels: 2,
            ..Default::default()
        });
        assert_eq!(f.levels(), 2);
        f.update(&t.translated(W, H, 4.0, 0.0), DT);
        let (u, v) = mean_interior(f.flow(), 16);
        assert!(u > 0.5 * 2.0 / DT, "lost motion after reconfigure: {u}");
        assert!(v.abs() < 0.35 * 2.0 / DT, "spurious v: {v}");
    }

    #[test]
    fn degenerate_tiny_grid_does_not_panic() {
        // Not reachable from the browser, but the solver must not assume it
        // has room for a pyramid.
        let mut f = OpticalFlow::new(0, 0, FlowConfig::default());
        assert_eq!((f.width(), f.height()), (2, 2));
        assert_eq!(f.levels(), 1);
        f.update(&[0, 0, 0, 0], DT);
        f.update(&[255, 0, 0, 255], DT);
        assert!(f.flow().max_speed().is_finite());
        assert!(f.motion_energy().is_finite());
    }

    #[test]
    fn a_moving_patch_localises_the_flow() {
        // Global translation is the easy case; a real hand moves a small part
        // of the frame, and the flow must stay near it instead of smearing
        // over the whole field.
        let t = Texture::new(18);
        let still = t.still(W, H);
        let mut moved = still.clone();
        let shifted = t.translated(W, H, 3.0, 0.0);
        for y in 20..52 {
            for x in 80..120 {
                moved[y * W + x] = shifted[y * W + x];
            }
        }
        let f = flow_of(&still, &moved, DT);
        let flow = f.flow();
        let mut inside = 0.0f64;
        let mut outside = 0.0f64;
        for y in 0..H {
            for x in 0..W {
                let (u, v) = flow.get(x, y);
                let mag = (u * u + v * v).sqrt() as f64;
                if (24..48).contains(&y) && (88..112).contains(&x) {
                    inside += mag;
                } else if x < 48 {
                    outside += mag;
                }
            }
        }
        inside /= (24 * 24) as f64;
        outside /= (48 * H) as f64;
        assert!(
            inside > 4.0 * outside.max(1e-6),
            "flow smeared: {inside} inside vs {outside} outside"
        );
    }
}
