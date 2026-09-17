//! WASM bindings for `aether-core`.
//!
//! Deliberately thin: it owns no logic, only the pointer handshake that lets
//! JS read and write engine buffers without copying. Every per-frame byte
//! (camera luma in, dye texture out, particle buffer out) moves through
//! [`AetherEngine::luma_ptr`] and friends as a view over WASM linear memory.
//!
//! ## Pointer invalidation
//!
//! A pointer stays valid until something reallocates or WASM memory grows.
//! In practice that means: re-read every pointer after `set_particle_count`
//! and `reset`. The engine allocates nothing else after construction, so
//! steady-state frames never invalidate a view.

use aether_core::engine::Engine;
use wasm_bindgen::prelude::*;

// `initThreadPool(n)` on the JS side: spawns `n` Web Workers over the shared
// memory and hands them to rayon. Only present in the `--threads` build.
#[cfg(all(target_arch = "wasm32", feature = "parallel"))]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Worker threads the engine spreads its hot loops over. `1` in the
/// single-threaded build, or before `initThreadPool` has resolved.
#[wasm_bindgen]
pub fn thread_count() -> usize {
    aether_core::par::threads()
}

/// Whether this module was compiled with the multithreaded engine.
#[wasm_bindgen]
pub fn is_parallel_build() -> bool {
    cfg!(feature = "parallel")
}

/// Number of floats in the packed stats array from [`AetherEngine::stats`].
pub const STATS_LEN: usize = 16;

/// Gesture-driven fluid engine, exposed to the browser.
#[wasm_bindgen]
pub struct AetherEngine {
    inner: Engine,
}

#[wasm_bindgen]
impl AetherEngine {
    /// Creates an engine. `seed` makes the particle system reproducible.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> AetherEngine {
        AetherEngine {
            inner: Engine::new(seed.abs() as u64),
        }
    }

    /// `[fluid_w, fluid_h, flow_w, flow_h, max_particles, particle_stride]`.
    ///
    /// The JS side asserts this against its own mirrored constants at startup,
    /// so a divergence fails loudly instead of rendering garbage.
    pub fn layout(&self) -> Vec<u32> {
        self.inner.layout().to_vec()
    }

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

    // ----------------------------------------------------------------- stats

    /// Packed stats, [`STATS_LEN`] floats. Indices are mirrored in
    /// `web/src/constants.ts` as `STAT`.
    ///
    /// `0` fluid energy, `1` max speed, `2` residual divergence,
    /// `3` motion energy, `4` flow dx, `5` flow dy, `6` particles alive,
    /// `7` mask coverage, `8` mask present, `9` hands present,
    /// `10` pose present, `11` time scale, `12` bursts, `13` NaN repairs,
    /// `14` frame, `15` ambient mode.
    pub fn stats(&self) -> Vec<f32> {
        let s = self.inner.stats();
        vec![
            s.fluid_energy,
            s.fluid_max_speed,
            s.fluid_divergence,
            s.motion_energy,
            s.flow_dx,
            s.flow_dy,
            s.particles_alive as f32,
            s.mask_coverage,
            if s.mask_present { 1.0 } else { 0.0 },
            s.hands_present as f32,
            if s.pose_present { 1.0 } else { 0.0 },
            s.time_scale,
            s.bursts as f32,
            s.nan_repairs as f32,
            s.frame as f32,
            if s.ambient { 1.0 } else { 0.0 },
        ]
    }

    /// Latched spell name for hand slot 0 or 1.
    pub fn spell_name(&self, slot: usize) -> String {
        self.inner.spell_name(slot).to_string()
    }

    // ---------------------------------------------------------------- config

    /// Sets a named tunable; returns false if the key is unknown.
    pub fn set_param(&mut self, key: &str, value: f32) -> bool {
        self.inner.set_param(key, value)
    }

    pub fn set_particle_count(&mut self, n: usize) {
        self.inner.set_particle_count(n);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Routes Rust panics to `console.error` instead of an opaque
/// `unreachable executed` trap.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_array_length_matches_the_constant() {
        let e = AetherEngine::new(1.0);
        assert_eq!(e.stats().len(), STATS_LEN);
    }

    #[test]
    fn layout_is_six_entries() {
        let e = AetherEngine::new(1.0);
        assert_eq!(e.layout().len(), 6);
    }

    #[test]
    fn buffer_lengths_agree_with_the_layout() {
        let e = AetherEngine::new(2.0);
        let layout = e.layout();
        let (w, h) = (layout[0] as usize, layout[1] as usize);
        assert_eq!(e.dye_len(), w * h * 4);
        assert_eq!(e.luma_len(), layout[2] as usize * layout[3] as usize);
    }

    #[test]
    fn negative_seed_does_not_panic() {
        let mut e = AetherEngine::new(-42.5);
        e.step(1.0 / 60.0);
        assert_eq!(e.stats()[13], 0.0, "NaN repairs should be zero");
    }

    #[test]
    fn spell_name_is_defined_for_both_slots_and_beyond() {
        let e = AetherEngine::new(3.0);
        assert!(!e.spell_name(0).is_empty());
        assert!(!e.spell_name(1).is_empty());
        assert_eq!(e.spell_name(99), "idle");
    }
}
