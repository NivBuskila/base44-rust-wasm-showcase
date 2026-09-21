//! The input side of the boundary: the buffers JS writes into and the four
//! `push_*` calls that consume them.
//!
//! Luma and segmentation masks are large and arrive every camera frame, so they
//! are written straight into engine-owned `Vec`s through raw pointers and
//! consumed in place — `push_luma` and `push_mask` take the buffer out with
//! `mem::take` only to satisfy the borrow checker while a subsystem holds `&mut
//! self`, and put it straight back. Landmarks are small enough to arrive as
//! slices.
//!
//! Every one of these also refreshes the perception clock, which is what keeps
//! [`Engine::step`](super::Engine::step) out of ambient mode.

use super::{sane_dt, Engine, MASK_IN_CAPACITY};

impl Engine {
    // ------------------------------------------------------- input buffers

    /// Writable `FLOW_W * FLOW_H` luma plane.
    #[inline]
    pub fn luma_mut(&mut self) -> &mut [u8] {
        &mut self.luma
    }

    #[inline]
    pub fn luma_ptr(&mut self) -> *mut u8 {
        self.luma.as_mut_ptr()
    }

    #[inline]
    pub fn luma_len(&self) -> usize {
        self.luma.len()
    }

    /// Writable segmentation mask staging buffer, [`MASK_IN_CAPACITY`] floats.
    #[inline]
    pub fn mask_mut(&mut self) -> &mut [f32] {
        &mut self.mask_in
    }

    #[inline]
    pub fn mask_ptr(&mut self) -> *mut f32 {
        self.mask_in.as_mut_ptr()
    }

    #[inline]
    pub fn mask_capacity(&self) -> usize {
        MASK_IN_CAPACITY
    }

    // --------------------------------------------------------------- inputs

    /// Consumes the luma buffer and recomputes optical flow.
    pub fn push_luma(&mut self, dt: f32) {
        let dt = sane_dt(dt);
        // Borrow split: `update` needs &self.luma while holding &mut self.flow.
        let luma = core::mem::take(&mut self.luma);
        self.flow.update(&luma, dt);
        self.luma = luma;
    }

    /// Consumes `w * h` floats from the mask buffer as the body silhouette.
    pub fn push_mask(&mut self, w: usize, h: usize, mirror: bool, dt: f32) {
        if w < 2 || h < 2 || w * h > MASK_IN_CAPACITY {
            return;
        }
        let dt = sane_dt(dt);
        let mask = core::mem::take(&mut self.mask_in);
        self.body.update_f32(&mask[..w * h], w, h, mirror, dt);
        self.mask_in = mask;
        self.since_perception = 0.0;
        self.obstacle_dirty = true;
    }

    /// Feeds packed hand landmarks. See [`crate::config`] for the layout.
    pub fn push_hands(&mut self, packed: &[f32], dt: f32) {
        let dt = sane_dt(dt);
        self.tracker.update_hands(packed, dt);
        self.tracker.update_two_hand(dt);
        if self.tracker.hands().iter().any(|h| h.present) {
            self.since_perception = 0.0;
        }
    }

    /// Feeds packed pose landmarks.
    pub fn push_pose(&mut self, packed: &[f32], dt: f32) {
        let dt = sane_dt(dt);
        self.tracker.update_pose(packed, dt);
        if self.tracker.pose().present {
            self.since_perception = 0.0;
        }
    }

    /// Clears all perception state, e.g. when the camera is switched.
    pub fn clear_perception(&mut self) {
        self.tracker.reset();
        self.body.reset();
        self.flow.reset();
        self.spell_state.reset();
        self.since_perception = f32::MAX / 4.0;
    }
}
