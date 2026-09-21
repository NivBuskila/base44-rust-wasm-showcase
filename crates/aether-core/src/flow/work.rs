//! Per-level solver scratch, allocated once and reused.

use crate::field::Grid;

/// Per-level scratch. Allocated once and reused: rebuilding a dozen `Grid`s
/// every camera frame would churn the wasm heap at 30 Hz for no reason.
pub(super) struct Work {
    /// Current flow estimate at this level, in cells per frame.
    pub(super) u: Grid,
    pub(super) v: Grid,
    /// Jacobi ping-pong targets.
    pub(super) un: Grid,
    pub(super) vn: Grid,
    /// Spatial derivatives, averaged between `prev` and the warped `cur`.
    pub(super) ix: Grid,
    pub(super) iy: Grid,
    /// Temporal residual, linearised about the initial guess.
    pub(super) it: Grid,
    /// `cur` warped by the initial guess.
    pub(super) warp: Grid,
    /// Blurred copy of this level, the source of the next coarser one.
    pub(super) smooth: Grid,
    /// Blur scratch.
    pub(super) tmp: Grid,
}

impl Work {
    pub(super) fn new(w: usize, h: usize) -> Self {
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
