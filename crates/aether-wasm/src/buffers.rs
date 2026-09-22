//! The zero-copy buffer handshake: the writable perception buffers, the
//! per-frame `push_*` calls that consume them, and the pointers JS renders
//! from. Pointer invalidation rules are in the crate doc.

use wasm_bindgen::prelude::*;

use crate::AetherEngine;

#[wasm_bindgen]
impl AetherEngine {
    // ------------------------------------------------------- input buffers

    /// Pointer to the writable luma plane (`luma_len()` bytes).
    pub fn luma_ptr(&mut self) -> usize {
        self.inner.luma_ptr() as usize
    }

    pub fn luma_len(&self) -> usize {
        self.inner.luma_len()
    }

    /// Pointer to the writable segmentation-mask staging buffer
    /// (`mask_capacity()` floats).
    pub fn mask_ptr(&mut self) -> usize {
        self.inner.mask_ptr() as usize
    }

    pub fn mask_capacity(&self) -> usize {
        self.inner.mask_capacity()
    }

    // --------------------------------------------------------------- inputs

    /// Recomputes optical flow from whatever is in the luma buffer.
    pub fn push_luma(&mut self, dt: f32) {
        self.inner.push_luma(dt);
    }

    /// Consumes `w * h` floats from the mask buffer as the body silhouette.
    pub fn push_mask(&mut self, w: usize, h: usize, mirror: bool, dt: f32) {
        self.inner.push_mask(w, h, mirror, dt);
    }

    /// Packed hand landmarks; layout documented in `aether_core::config`.
    pub fn push_hands(&mut self, packed: &[f32], dt: f32) {
        self.inner.push_hands(packed, dt);
    }

    /// Packed pose landmarks.
    pub fn push_pose(&mut self, packed: &[f32], dt: f32) {
        self.inner.push_pose(packed, dt);
    }

    /// Drops all tracking state, e.g. after switching camera.
    pub fn clear_perception(&mut self) {
        self.inner.clear_perception();
    }

    // ----------------------------------------------------------------- step

    pub fn step(&mut self, dt: f32) {
        self.inner.step(dt);
    }

    // --------------------------------------------------------------- output

    /// Pointer to the RGBA8 dye texture, `fluid_w * fluid_h * 4` bytes.
    pub fn dye_ptr(&self) -> usize {
        self.inner.dye_ptr() as usize
    }

    pub fn dye_len(&self) -> usize {
        self.inner.dye_rgba().len()
    }

    /// Pointer to the interleaved particle buffer: `x, y, heat, life`,
    /// `particle_count() * 4` floats, positions normalised to `[0, 1]`.
    pub fn particle_ptr(&self) -> usize {
        self.inner.particle_ptr() as usize
    }

    pub fn particle_count(&self) -> usize {
        self.inner.particle_count()
    }

    /// Refreshes and returns a pointer to the RGBA8 debug texture
    /// (red = obstacle, green/blue = signed optical flow).
    pub fn debug_ptr(&mut self) -> usize {
        let _ = self.inner.debug_rgba();
        self.inner.debug_ptr() as usize
    }
}
