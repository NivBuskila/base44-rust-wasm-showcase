//! Hand ingest: `update_hands` and the missing-frame path behind it.
//!
//! Split out of `gesture/mod.rs` the way `flow/` and `mask/` are: the parent
//! keeps the struct and the tuning constants, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

impl GestureTracker {
    /// Ingests a packed hand buffer. See the module docs for the layout.
    ///
    /// A hand appears the instant it is reported — waiting would make the app
    /// feel slow — but only disappears after `release_frames` consecutive
    /// frames without it, because MediaPipe drops a hand for a frame or two
    /// whenever it turns edge-on to the camera.
    pub fn update_hands(&mut self, packed: &[f32], dt: f32) {
        let dt = clamp_dt(dt);
        let cfg = self.cfg;

        for slot in 0..HANDS {
            let base = slot * HAND_STRIDE;
            // A truncated buffer is a protocol violation, not an absent hand,
            // but the only safe reading of it is "no data this frame": that
            // path already coasts and then releases.
            let Some(raw) = read_hand(packed, base) else {
                self.drop_hand_frame(slot, dt);
                continue;
            };

            let first = !self.initialised[slot];
            let raw_palm = centroid(&raw, &PALM_JOINTS);

            // One-euro: the speed that sets the cutoff is itself filtered,
            // otherwise landmark noise inflates it and the filter unblocks the
            // very jitter it exists to remove.
            let alpha_v = ema_alpha(cfg.velocity_half_life, dt);
            let raw_speed = if first {
                0.0
            } else {
                distance(raw_palm, self.raw_palm[slot]) / dt
            };
            self.raw_palm[slot] = raw_palm;
            self.speed_est[slot] = if first {
                0.0
            } else {
                self.speed_est[slot] + (raw_speed - self.speed_est[slot]) * alpha_v
            };
            let alpha_p = adaptive_alpha(cfg.position_half_life, self.speed_est[slot], dt);

            self.missing[slot] = 0;
            let hand = &mut self.hands[slot];
            hand.present = true;
            hand.right = packed[base + 1] > 0.5;
            hand.canned = CannedGesture::from_id(packed[base + 2] as i32);
            hand.canned_score = finite_or(packed[base + 3], 0.0).clamp(0.0, 1.0);

            if first {
                hand.landmarks = raw;
            } else {
                for (filtered, target) in hand.landmarks.iter_mut().zip(raw.iter()) {
                    for (f, t) in filtered.iter_mut().zip(target.iter()) {
                        *f += (*t - *f) * alpha_p;
                    }
                }
            }

            let lm = &hand.landmarks;
            hand.palm = centroid(lm, &PALM_JOINTS);
            hand.index_tip = [lm[LM_INDEX_TIP][0], lm[LM_INDEX_TIP][1]];
            hand.thumb_tip = [lm[LM_THUMB_TIP][0], lm[LM_THUMB_TIP][1]];
            // Wrist to middle knuckle: the one span on a hand that is almost
            // invariant to pose, which is what makes it usable as a size proxy.
            let wrist = [lm[LM_WRIST][0], lm[LM_WRIST][1]];
            hand.scale =
                distance(wrist, [lm[LM_MIDDLE_MCP][0], lm[LM_MIDDLE_MCP][1]]).max(MIN_SCALE);

            // Everything geometric is expressed in hand scales. Raw normalised
            // distances would make every gesture depend on how far the user is
            // standing from the camera, so a pinch that works at arm's length
            // would be impossible to reach from across the room.
            if hand.scale >= MIN_CREDIBLE_SCALE {
                let pinch_ratio = distance(hand.thumb_tip, hand.index_tip) / hand.scale;
                hand.pinch = smoothstep(PINCH_FAR, PINCH_NEAR, pinch_ratio);

                let mut spread = 0.0;
                for &tip in &OPEN_TIPS {
                    spread += distance(wrist, [lm[tip][0], lm[tip][1]]);
                }
                spread /= OPEN_TIPS.len() as f32;
                hand.openness = smoothstep(OPEN_CLOSED, OPEN_EXTENDED, spread / hand.scale);
            } else {
                hand.pinch = 0.0;
                hand.openness = 0.0;
            }

            // Differentiating raw landmarks is unusable — MediaPipe's per-frame
            // noise is the same order as the motion between frames at 30 Hz —
            // so this differentiates the filtered palm and smooths the result.
            if first {
                hand.velocity = [0.0, 0.0];
            } else {
                let inv = 1.0 / dt;
                let raw_v = [
                    (hand.palm[0] - self.prev_palm[slot][0]) * inv,
                    (hand.palm[1] - self.prev_palm[slot][1]) * inv,
                ];
                hand.velocity[0] += (raw_v[0] - hand.velocity[0]) * alpha_v;
                hand.velocity[1] += (raw_v[1] - hand.velocity[1]) * alpha_v;
            }
            self.prev_palm[slot] = hand.palm;
            self.initialised[slot] = true;

            let candidate = candidate_spell(hand, &cfg);
            if candidate == self.candidate[slot].0 {
                self.candidate[slot].1 = self.candidate[slot].1.saturating_add(1);
            } else {
                self.candidate[slot] = (candidate, 1);
            }
            if self.candidate[slot].1 >= cfg.commit_frames.max(1) && hand.spell != candidate {
                hand.spell = candidate;
                hand.spell_age = 0.0;
            } else {
                hand.spell_age += dt;
            }
        }
    }

    /// Advances one hand through a frame in which it was not reported.
    fn drop_hand_frame(&mut self, slot: usize, dt: f32) {
        self.missing[slot] = self.missing[slot].saturating_add(1);
        if self.missing[slot] >= self.cfg.release_frames.max(1) {
            // Full release. `initialised` goes with it so the next sighting
            // snaps to the new position instead of sliding in from wherever the
            // hand was last seen, which otherwise sweeps a force across the
            // screen every time the user drops a hand and raises the other.
            self.hands[slot] = HandState::default();
            self.candidate[slot] = (Spell::Idle, 0);
            self.initialised[slot] = false;
            self.speed_est[slot] = 0.0;
            return;
        }
        if self.hands[slot].present {
            let keep = decay(COAST_DECAY, dt);
            self.hands[slot].velocity[0] *= keep;
            self.hands[slot].velocity[1] *= keep;
            self.hands[slot].spell_age += dt;
            self.speed_est[slot] *= keep;
        }
    }
}
