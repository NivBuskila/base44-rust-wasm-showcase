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

mod buffers;
mod gpu;
mod reports;

/// Number of floats in the packed stats array from [`AetherEngine::stats`].
pub const STATS_LEN: usize = 16;

/// Gesture-driven fluid engine, exposed to the browser.
#[wasm_bindgen]
pub struct AetherEngine {
    inner: Engine,
    stats_buffer: [f32; STATS_LEN],
}

#[wasm_bindgen]
impl AetherEngine {
    /// Creates an engine. `seed` makes the particle system reproducible.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> AetherEngine {
        AetherEngine {
            inner: Engine::new(seed.abs() as u64),
            stats_buffer: [0.0; STATS_LEN],
        }
    }

    /// `[fluid_w, fluid_h, flow_w, flow_h, max_particles, particle_stride]`.
    ///
    /// The JS side asserts this against its own mirrored constants at startup,
    /// so a divergence fails loudly instead of rendering garbage.
    pub fn layout(&self) -> Vec<u32> {
        self.inner.layout().to_vec()
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
    fn stats_pointer_reuses_storage_and_tracks_each_step() {
        let mut e = AetherEngine::new(1.0);
        let ptr = e.stats_ptr();
        assert_eq!(ptr, e.stats_ptr());
        assert_eq!(e.stats_buffer.to_vec(), e.stats());
        e.step(1.0 / 60.0);
        assert_eq!(ptr, e.stats_ptr());
        assert_eq!(e.stats_buffer.to_vec(), e.stats());
        assert_eq!(e.stats_buffer[14], 1.0);
    }

    #[test]
    fn the_combo_book_is_parseable_and_progress_is_idle_at_rest() {
        let e = AetherEngine::new(3.0);
        let book = e.combo_book();
        assert!(!book.is_empty());
        for entry in book.split(';') {
            let (name, steps) = entry.split_once(':').expect("no name:steps separator");
            assert!(!name.is_empty(), "combo with no name");
            assert!(
                steps.split(',').count() >= 2,
                "a one-step combo is a gesture"
            );
        }
        let p = e.combo_progress();
        assert_eq!(p.len(), 4);
        assert_eq!(p[0], -1.0, "a fresh engine reports a sequence in progress");
        assert_eq!(p[3], -1.0);
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
