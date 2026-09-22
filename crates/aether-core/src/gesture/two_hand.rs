//! Two-handed derivation: `update_two_hand`.
//!
//! Split out of `gesture/mod.rs` the way `flow/` and `mask/` are: the parent
//! keeps the struct and the tuning constants, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl GestureTracker {
    /// Recomputes [`TwoHandState`] from the current hand states.
    ///
    /// The angle is differentiated after unwrapping across the +/-pi seam.
    /// Differentiating the wrapped angle instead puts a ~2*pi/dt spike in
    /// `angular_velocity` once per revolution, and that spike drives the time
    /// warp, so the simulation would visibly lurch every time the user's hands
    /// passed vertical.
    pub fn update_two_hand(&mut self, dt: f32) {
        let dt = clamp_dt(dt);
        let both = self.hands[0].present && self.hands[1].present;
        self.two.both_present = both;
        if !both {
            self.two.distance_velocity = 0.0;
            self.two.angular_velocity = 0.0;
            self.two_initialised = false;
            return;
        }

        let a = self.hands[0].palm;
        let b = self.hands[1].palm;
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dist = (dx * dx + dy * dy).sqrt();
        self.two.distance = dist;
        self.two.midpoint = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];

        // atan2 of a degenerate pair is meaningless; hold the last angle rather
        // than letting it snap to 0 and register as a huge rotation.
        let angle = if dist > MIN_SCALE {
            dy.atan2(dx)
        } else {
            self.two.angle
        };
        self.two.angle = angle;

        if !self.two_initialised {
            self.last_angle = angle;
            self.last_distance = dist;
            self.two.distance_velocity = 0.0;
            self.two.angular_velocity = 0.0;
            self.two_initialised = true;
            return;
        }

        let step = wrap_pi(angle - self.last_angle);
        self.last_angle = angle;
        let alpha = ema_alpha(self.cfg.velocity_half_life, dt);
        let inv = 1.0 / dt;
        self.two.angular_velocity += (step * inv - self.two.angular_velocity) * alpha;
        self.two.distance_velocity +=
            ((dist - self.last_distance) * inv - self.two.distance_velocity) * alpha;
        self.last_distance = dist;
    }
}
