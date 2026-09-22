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

mod pyramid;
mod solve;
#[cfg(test)]
mod tests;
mod work;

use pyramid::build_pyramid;
use work::Work;

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

/// Clamps a frame interval into a range where `1 / dt` is meaningful.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 1000.0, 1.0)
}
