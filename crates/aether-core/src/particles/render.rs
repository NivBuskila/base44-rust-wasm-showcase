//! The interleaved render buffer the renderers read.
//!
//! Split out of `particles/mod.rs`: the parent keeps the pool, its tuning
//! constants and the small accessors, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl Particles {
    /// Packs `x, y, heat, life` per particle, with position normalised to
    /// `[0, 1]` against the grid the simulation runs on and `life` normalised
    /// against that particle's own lifetime.
    pub fn build_render_buffer(&mut self, grid_w: usize, grid_h: usize) {
        // The GPU pool draws straight from its own storage buffer.
        if self.offloaded {
            return;
        }
        let sx = 1.0 / (grid_w - 1) as f32;
        let sy = 1.0 / (grid_h - 1) as f32;
        let n = self.active;
        let lane = par::chunk_len(n, LANE_MIN);
        let (x, y, heat, life, max_life) = (
            &self.x[..n],
            &self.y[..n],
            &self.heat[..n],
            &self.life[..n],
            &self.max_life[..n],
        );
        par::chunks_mut(
            &mut self.render[..n * PARTICLE_STRIDE],
            lane * PARTICLE_STRIDE,
            |c, out| {
                let base = c * lane;
                for (k, px) in out.chunks_exact_mut(PARTICLE_STRIDE).enumerate() {
                    let i = base + k;
                    let norm_life = if max_life[i] > 0.0 {
                        (life[i] / max_life[i]).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    px[0] = x[i] * sx;
                    px[1] = y[i] * sy;
                    px[2] = heat[i];
                    px[3] = norm_life;
                }
            },
        );
    }

    /// The interleaved buffer, truncated to the active particle count.
    #[inline]
    pub fn render_buffer(&self) -> &[f32] {
        &self.render[..self.active * PARTICLE_STRIDE]
    }

    /// Raw pointer to the render buffer, for zero-copy access from JS.
    #[inline]
    pub fn render_ptr(&self) -> *const f32 {
        self.render.as_ptr()
    }
}
