//! Handing the particle pool to the browser's GPU.
//!
//! The fluid, the flow, the spells and every bit of gameplay stay in Rust; only
//! the pool's per-particle integration leaves. What crosses the boundary each
//! frame: the velocity field and the obstacle (out), the op log of what the
//! spells *would* have done to the particles (out), and the live count (in).
//! The op log's protocol lives in [`crate::particles::offload`], which is its
//! single source of truth for both this side and the WGSL replay.

use super::Engine;

impl Engine {
    // ----------------------------------------------------------- gpu offload

    /// Moves the particle simulation to the browser's GPU (or back).
    ///
    /// The fluid, the flow, the spells and every bit of gameplay stay here;
    /// only the pool's per-particle integration leaves. What crosses the
    /// boundary each frame: the velocity field and obstacle (out), the op log
    /// (out) and the live count (in).
    pub fn set_gpu_particles(&mut self, on: bool) {
        self.particles.set_offloaded(on);
    }

    #[inline]
    pub fn gpu_particles(&self) -> bool {
        self.particles.offloaded()
    }

    /// The particle op log for the frame `step` just produced.
    #[inline]
    pub fn particle_ops(&self) -> &[f32] {
        self.particles.ops().records()
    }

    #[inline]
    pub fn particle_ops_ptr(&self) -> *const f32 {
        self.particles.ops().ptr()
    }

    #[inline]
    pub fn particle_ops_len(&self) -> usize {
        self.particles.ops().len()
    }

    /// Live count as measured on the GPU; ignored unless offloaded.
    pub fn set_particles_alive(&mut self, alive: usize) {
        self.particles.set_alive_hint(alive);
        self.stats.particles_alive = self.particles.alive() as u32;
    }

    /// Everything a GPU particle step needs beyond the fields:
    /// `[sim_dt, life, drag, spawn_rate, damping, curl_influence, gravity]`.
    pub fn particle_step_params(&self) -> [f32; 7] {
        [
            self.last_sim_dt,
            self.params.particle_life,
            self.params.particle_drag,
            self.params.spawn_rate,
            self.particle_cfg.damping,
            self.particle_cfg.curl_influence,
            self.particle_cfg.gravity,
        ]
    }

    /// Fluid velocity, `u` then `v`, each `FLUID_CELLS` floats in grid cells per second.
    #[inline]
    pub fn velocity_u_ptr(&self) -> *const f32 {
        self.fluid.velocity().u.data.as_ptr()
    }

    #[inline]
    pub fn velocity_v_ptr(&self) -> *const f32 {
        self.fluid.velocity().v.data.as_ptr()
    }

    /// Obstacle field the particles collide with; `[w, h, max_abs]` describes it.
    #[inline]
    pub fn obstacle_ptr(&self) -> *const f32 {
        self.fluid.obstacle().data.as_ptr()
    }

    pub fn obstacle_info(&self) -> [f32; 3] {
        let o = self.fluid.obstacle();
        [o.w as f32, o.h as f32, o.max_abs()]
    }
}
