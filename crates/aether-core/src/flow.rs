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

/// Dense motion estimator over successive camera luma planes.
pub struct OpticalFlow {
    w: usize,
    h: usize,
    cfg: FlowConfig,
    prev: Grid,
    cur: Grid,
    flow: VecField,
    scratch: Grid,
    has_history: bool,
    motion_energy: f32,
    dominant: (f32, f32),
}

impl OpticalFlow {
    pub fn new(w: usize, h: usize, cfg: FlowConfig) -> Self {
        Self {
            w,
            h,
            cfg,
            prev: Grid::new(w, h),
            cur: Grid::new(w, h),
            flow: VecField::new(w, h),
            scratch: Grid::new(w, h),
            has_history: false,
            motion_energy: 0.0,
            dominant: (0.0, 0.0),
        }
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
    }

    /// Flow field in grid cells per second.
    #[inline]
    pub fn flow(&self) -> &VecField {
        &self.flow
    }

    /// Mean squared flow magnitude — the "how much is happening" scalar the
    /// visuals use to scale global intensity.
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

    pub fn reset(&mut self) {
        self.prev.zero();
        self.cur.zero();
        self.flow.zero();
        self.has_history = false;
        self.motion_energy = 0.0;
        self.dominant = (0.0, 0.0);
    }

    /// Feeds a new `w * h` luma plane and recomputes the flow.
    ///
    /// AGENT-IMPLEMENT: the placeholder only rotates the frame history and
    /// reports zero motion.
    pub fn update(&mut self, luma: &[u8], dt: f32) {
        assert_eq!(luma.len(), self.w * self.h, "luma plane size mismatch");
        let _ = dt;
        core::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.fill_from_u8(luma);
        let _ = &mut self.scratch;
        if !self.has_history {
            self.has_history = true;
            return;
        }
        self.flow.zero();
        self.motion_energy = 0.0;
        self.dominant = (0.0, 0.0);
    }
}
