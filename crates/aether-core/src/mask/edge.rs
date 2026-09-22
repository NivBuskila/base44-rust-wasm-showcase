//! Boundary velocity: what makes a moving silhouette push fluid instead of
//! merely blocking it.

use super::*;

impl BodyMask {
    /// Rebuilds the boundary velocity from the temporal delta of `smooth`.
    ///
    /// The mask is a level set carried by the body, so it obeys
    /// `dm/dt + w . grad(m) = 0`: the boundary moves along `-sign(dm/dt) * n`
    /// with `n = grad(m) / |grad(m)|`. That gets the sign right at both ends of
    /// a silhouette — at the leading edge the mask is advancing into cells
    /// where `grad(m)` points backwards, at the trailing edge it is receding
    /// out of cells where `grad(m)` points forwards, and both produce a push
    /// along the direction of travel.
    ///
    /// The magnitude is the temporal rate `delta / dt` scaled by `push_gain`,
    /// not the level-set's own `|dm/dt| / |grad(m)|`: that quotient is exact for
    /// a rigid translation but divides by a gradient that vanishes two cells
    /// away from the edge, where the numerator is only *nearly* zero.
    pub(super) fn rebuild_edge(&mut self, dt: f32) {
        let gain = if self.cfg.push_gain.is_finite() {
            self.cfg.push_gain
        } else {
            0.0
        };
        let rate_scale = gain / dt;

        for y in 0..self.h {
            let row = y * self.w;
            for x in 0..self.w {
                let i = row + x;
                let delta = self.smooth.data[i] - self.prev.data[i];
                let mut u = 0.0;
                let mut v = 0.0;
                // The overwhelming majority of cells are flat interior or empty
                // space, so this early-out is what keeps the pass cheap.
                if delta.abs() > DELTA_EPS {
                    let (gx, gy) = self.smooth.gradient(x, y);
                    let gmag = length(gx, gy);
                    let band = smoothstep(EDGE_GRAD_MIN, EDGE_GRAD_FULL, gmag);
                    if band > 0.0 {
                        // `band > 0` implies `gmag > EDGE_GRAD_MIN`, so the
                        // normalisation below cannot divide by zero.
                        let speed =
                            (-delta * rate_scale * band).clamp(-MAX_EDGE_SPEED, MAX_EDGE_SPEED);
                        u = speed * gx / gmag;
                        v = speed * gy / gmag;
                    }
                }
                self.edge.u.data[i] = u;
                self.edge.v.data[i] = v;
            }
        }
    }
}
