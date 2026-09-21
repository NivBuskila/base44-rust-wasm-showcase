//! The output side of the boundary: the dye texture, the particle buffer and
//! the debug overlay, plus the tone-mapping curve that produces the dye.
//!
//! Tone mapping is `1 - exp(-x)` rather than a hard clamp, because dye
//! accumulates without bound where gestures dwell and a clamp turns those
//! regions into flat white blobs. Read straight it would cost three
//! transcendental calls per cell per frame — more than the entire particle
//! system — so it comes from a table; [`TONEMAP_LUT`](super::TONEMAP_LUT) says
//! why that is visually free, and the tests pin the table against the exact
//! curve to within one 8-bit step.

use super::{Engine, TONEMAP_LUT, TONEMAP_MAX};
use crate::config::{FLOW_H, FLOW_W, FLUID_CELLS, FLUID_H, FLUID_W};

impl Engine {
    // ---------------------------------------------------------------- output

    /// Tone-maps the float dye field into the RGBA8 texture JS uploads.
    ///
    /// `1 - exp(-x)` rather than a hard clamp: dye accumulates without bound
    /// where gestures dwell, and a clamp turns those regions into flat white
    /// blobs while this keeps the internal structure visible. The curve comes
    /// from a table — see [`TONEMAP_LUT`] for why.
    pub(super) fn encode_dye(&mut self) {
        let dye = self.fluid.dye();
        let lut = &self.tonemap_lut;
        let (r_ch, g_ch, b_ch) = (&dye[0].data, &dye[1].data, &dye[2].data);

        for i in 0..FLUID_CELLS {
            let r = tonemap_lut(lut, r_ch[i]);
            let g = tonemap_lut(lut, g_ch[i]);
            let b = tonemap_lut(lut, b_ch[i]);
            let o = i * 4;
            self.dye_rgba[o] = (r * 255.0) as u8;
            self.dye_rgba[o + 1] = (g * 255.0) as u8;
            self.dye_rgba[o + 2] = (b * 255.0) as u8;
            // Alpha carries brightness so the compositor can blend the dye
            // over the camera feed without a second texture fetch.
            self.dye_rgba[o + 3] = (r.max(g).max(b) * 255.0) as u8;
        }
    }

    #[inline]
    pub fn dye_rgba(&self) -> &[u8] {
        &self.dye_rgba
    }

    #[inline]
    pub fn dye_ptr(&self) -> *const u8 {
        self.dye_rgba.as_ptr()
    }

    #[inline]
    pub fn particle_buffer(&self) -> &[f32] {
        self.particles.render_buffer()
    }

    #[inline]
    pub fn particle_ptr(&self) -> *const f32 {
        self.particles.render_ptr()
    }

    #[inline]
    pub fn particle_count(&self) -> usize {
        self.particles.active()
    }

    /// Encodes the obstacle field and flow into an RGBA debug texture:
    /// red = obstacle, green/blue = signed flow. Drives the debug overlay.
    pub fn debug_rgba(&mut self) -> &[u8] {
        let obstacle = self.fluid.obstacle();
        let flow = self.flow.flow();
        let sx = (FLOW_W - 1) as f32 / (FLUID_W - 1) as f32;
        let sy = (FLOW_H - 1) as f32 / (FLUID_H - 1) as f32;
        for y in 0..FLUID_H {
            let fy = y as f32 * sy;
            for x in 0..FLUID_W {
                let i = y * FLUID_W + x;
                let (u, v) = flow.sample(x as f32 * sx, fy);
                let o = i * 4;
                self.debug_rgba[o] = (obstacle.data[i].clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 1] = ((u * 0.02 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 2] = ((v * 0.02 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 3] = 255;
            }
        }
        &self.debug_rgba
    }

    #[inline]
    pub fn debug_ptr(&self) -> *const u8 {
        self.debug_rgba.as_ptr()
    }
}

/// The exact tone-mapping curve. Only the table is used in anger; this is kept
/// as the reference the table is asserted against.
#[cfg(test)]
#[inline]
pub(super) fn tonemap(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    1.0 - (-x).exp()
}

/// Builds the `1 - exp(-x)` table, with one guard entry past the end so the
/// interpolation in [`tonemap_lut`] never needs a bounds branch.
pub(super) fn build_tonemap_lut() -> Vec<f32> {
    let mut lut = Vec::with_capacity(TONEMAP_LUT + 2);
    for i in 0..=TONEMAP_LUT + 1 {
        let x = (i.min(TONEMAP_LUT) as f32 / TONEMAP_LUT as f32) * TONEMAP_MAX;
        lut.push(1.0 - (-x).exp());
    }
    lut
}

/// Linear interpolation into the tone-mapping table.
#[inline]
pub(super) fn tonemap_lut(lut: &[f32], x: f32) -> f32 {
    // Non-finite dye should be impossible, but a NaN here would write a
    // garbage byte into the texture and flash a stray pixel, so it is cheaper
    // to test than to debug.
    if x.is_nan() || x <= 0.0 {
        return 0.0;
    }
    if x >= TONEMAP_MAX {
        return lut[TONEMAP_LUT];
    }
    let t = x * (TONEMAP_LUT as f32 / TONEMAP_MAX);
    let i = t as usize;
    let frac = t - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * frac
}
