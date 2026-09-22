//! Fixtures: a tracker, the frame feeder and the pose buffer builder.

pub(super) use crate::gesture::*;

pub(super) use crate::config::POSE_STRIDE;
pub(super) use crate::gesture::synth::{self, Hand};
pub(super) const DT: f32 = 1.0 / 30.0;

pub(super) fn tracker() -> GestureTracker {
    GestureTracker::new(GestureConfig::default())
}

/// Feeds one hand for `frames` frames and returns the tracker.
pub(super) fn feed(t: &mut GestureTracker, hand: &Hand, frames: usize, dt: f32) {
    let mut buf = synth::buffer();
    for _ in 0..frames {
        synth::write(&mut buf, 0, hand);
        t.update_hands(&buf, dt);
        t.update_two_hand(dt);
    }
}

/// A pose buffer with `present` set and only the listed joints visible.
pub(super) fn pose_buffer(joints: &[(usize, f32, f32, f32)]) -> Vec<f32> {
    let mut buf = vec![0.0f32; POSE_STRIDE];
    buf[0] = 1.0;
    for &(j, x, y, v) in joints {
        let o = 1 + j * 4;
        buf[o] = x;
        buf[o + 1] = y;
        buf[o + 2] = 0.0;
        buf[o + 3] = v;
    }
    buf
}
