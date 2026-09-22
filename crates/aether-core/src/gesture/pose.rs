//! Body-pose ingest: `update_pose`.
//!
//! Split out of `gesture/mod.rs` the way `flow/` and `mask/` are: the parent
//! keeps the struct and the tuning constants, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl GestureTracker {
    /// Ingests a packed pose buffer.
    ///
    /// Landmarks whose visibility is low are *frozen* rather than filtered:
    /// MediaPipe still emits a position for an occluded joint, and letting a
    /// hallucinated hip drag the torso centre halfway across the frame is worse
    /// than carrying a slightly stale one.
    pub fn update_pose(&mut self, packed: &[f32], dt: f32) {
        let dt = clamp_dt(dt);
        // Bound first: a NaN flag must read as absent, which `flag <= 0.5`
        // would get backwards.
        let flagged = packed.first().copied().unwrap_or(0.0) > 0.5;
        if packed.len() < POSE_STRIDE || !flagged {
            self.pose_missing = self.pose_missing.saturating_add(1);
            if self.pose_missing >= self.cfg.release_frames.max(1) {
                self.pose.present = false;
                self.pose.velocity = [0.0, 0.0];
                // The filter state goes with it, exactly as it does for a
                // released hand: keeping `prev_center` across the gap makes the
                // next sighting differentiate a teleport, and the torso centre
                // then slides across the frame at a metre a second for a body
                // that has not moved at all.
                self.pose_initialised = false;
                self.pose_seen = [false; POSE_LANDMARKS];
                self.pose.visibility = [0.0; POSE_LANDMARKS];
            }
            return;
        }
        self.pose_missing = 0;
        let first = !self.pose_initialised;
        let alpha_p = ema_alpha(self.cfg.position_half_life * POSE_HALF_LIFE_SCALE, dt);
        let alpha_v = ema_alpha(self.cfg.velocity_half_life, dt);
        self.pose.present = true;

        for l in 0..POSE_LANDMARKS {
            let o = 1 + l * 4;
            let (x, y, z, v) = (packed[o], packed[o + 1], packed[o + 2], packed[o + 3]);
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let v = finite_or(v, 0.0).clamp(0.0, 1.0);
            // Seeded rather than filtered up from zero. The centroid gate below
            // reads the *filtered* visibility, so easing it in from 0 would
            // leave every joint below `MIN_VISIBILITY` for the first few frames
            // and the torso centre parked at its default until the filter
            // caught up — a hundred milliseconds of wrong answer every time the
            // pose is acquired.
            if self.pose_seen[l] {
                self.pose.visibility[l] += (v - self.pose.visibility[l]) * alpha_p;
            } else {
                self.pose.visibility[l] = v;
            }
            if v < MIN_VISIBILITY {
                continue;
            }
            let p = [x, y, finite_or(z, 0.0)];
            if self.pose_seen[l] {
                for (f, t) in self.pose.landmarks[l].iter_mut().zip(p.iter()) {
                    *f += (*t - *f) * alpha_p;
                }
            } else {
                self.pose.landmarks[l] = p;
                self.pose_seen[l] = true;
            }
        }

        // Visibility-weighted so a marginal joint contributes proportionally
        // instead of flipping the centre when it crosses the threshold.
        let mut sum = [0.0f32, 0.0];
        let mut weight = 0.0f32;
        for &j in &[
            PL_LEFT_SHOULDER,
            PL_RIGHT_SHOULDER,
            PL_LEFT_HIP,
            PL_RIGHT_HIP,
        ] {
            if !self.pose_seen[j] || self.pose.visibility[j] < MIN_VISIBILITY {
                continue;
            }
            let w = self.pose.visibility[j];
            sum[0] += self.pose.landmarks[j][0] * w;
            sum[1] += self.pose.landmarks[j][1] * w;
            weight += w;
        }
        if weight > 1e-4 {
            let center = [sum[0] / weight, sum[1] / weight];
            if first {
                self.pose.center = center;
                self.pose.velocity = [0.0, 0.0];
            } else {
                let inv = 1.0 / dt;
                let raw_v = [
                    (center[0] - self.prev_center[0]) * inv,
                    (center[1] - self.prev_center[1]) * inv,
                ];
                self.pose.center = center;
                self.pose.velocity[0] += (raw_v[0] - self.pose.velocity[0]) * alpha_v;
                self.pose.velocity[1] += (raw_v[1] - self.pose.velocity[1]) * alpha_v;
            }
            self.prev_center = self.pose.center;
            self.pose_initialised = true;
        }

        if self.pose_seen[PL_LEFT_SHOULDER] && self.pose_seen[PL_RIGHT_SHOULDER] {
            let l = self.pose.landmarks[PL_LEFT_SHOULDER];
            let r = self.pose.landmarks[PL_RIGHT_SHOULDER];
            let span = distance([l[0], l[1]], [r[0], r[1]]);
            if span > 1e-4 {
                self.pose.span = span;
            }
        }
    }
}
