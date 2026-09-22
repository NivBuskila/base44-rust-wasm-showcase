//! The WebGPU offload surface: handing the particle pool over, the op-log
//! protocol the browser replays, and the fluid/obstacle planes the compute
//! pass samples.

use wasm_bindgen::prelude::*;

use crate::AetherEngine;

#[wasm_bindgen]
impl AetherEngine {
    // ----------------------------------------------------------- gpu offload

    /// Hands the particle pool to a WebGPU compute shader (or takes it back).
    /// While on, `particle_ptr()` is stale and `particle_ops_*` carry the
    /// spells' per-frame mutations for the browser to replay.
    pub fn set_gpu_particles(&mut self, on: bool) {
        self.inner.set_gpu_particles(on);
    }

    pub fn gpu_particles(&self) -> bool {
        self.inner.gpu_particles()
    }

    /// `[op_stride, max_ops, ...one entry per op kind]`, the kinds in
    /// `aether_core::particles::OP_KINDS` order.
    ///
    /// The whole protocol travels here, kinds included, so `assertOpLayout` in
    /// `web/src/constants.ts` can reject *any* drift at boot instead of only a
    /// changed stride — a renumbered kind used to replay as a different op.
    pub fn particle_op_layout(&self) -> Vec<u32> {
        let mut v = vec![
            aether_core::particles::OP_STRIDE as u32,
            aether_core::particles::MAX_OPS as u32,
        ];
        v.extend(
            aether_core::particles::OP_KINDS
                .iter()
                .map(|(_, kind)| *kind as u32),
        );
        v
    }

    /// Pointer to this frame's op records, `particle_ops_len() * op_stride` floats.
    pub fn particle_ops_ptr(&self) -> usize {
        self.inner.particle_ops_ptr() as usize
    }

    pub fn particle_ops_len(&self) -> usize {
        self.inner.particle_ops_len()
    }

    /// `[sim_dt, life, drag, spawn_rate, damping, curl_influence, gravity]`.
    pub fn particle_step_params(&self) -> Vec<f32> {
        self.inner.particle_step_params().to_vec()
    }

    /// Feeds the GPU's live particle count back into `stats()`.
    pub fn set_particles_alive(&mut self, alive: usize) {
        self.inner.set_particles_alive(alive);
    }

    /// Fluid velocity planes, each `fluid_w * fluid_h` floats.
    pub fn velocity_u_ptr(&self) -> usize {
        self.inner.velocity_u_ptr() as usize
    }

    pub fn velocity_v_ptr(&self) -> usize {
        self.inner.velocity_v_ptr() as usize
    }

    /// Obstacle field, `obstacle_info()[0] * [1]` floats; `[2]` is its max.
    pub fn obstacle_ptr(&self) -> usize {
        self.inner.obstacle_ptr() as usize
    }

    pub fn obstacle_info(&self) -> Vec<f32> {
        self.inner.obstacle_info().to_vec()
    }
}
