//! Synthetic landmark packing, shared by this module's tests and
//! [`crate::spells`]'s.
//!
//! The joint layout is a transcription of the scripted perception source in
//! `web/tests/gesture.spec.ts`, so the host tests and the browser end-to-end
//! suite assert against the *same* geometry instead of two independent
//! idealisations of a hand — a threshold that passes here therefore also passes
//! there.

use crate::config::{HAND_BUFFER, HAND_STRIDE};

pub const NONE: i32 = 0;
pub const CLOSED_FIST: i32 = 1;
pub const OPEN_PALM: i32 = 2;
pub const POINTING_UP: i32 = 3;
pub const THUMB_UP: i32 = 5;
pub const VICTORY: i32 = 6;

/// Finger chains in MediaPipe order, palm joint first, tip last.
const CHAINS: [[usize; 4]; 5] = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
];
/// Chain bases in hand-local units: origin at the middle MCP, +y toward the
/// wrist, one unit = one hand scale.
const BASES: [[f32; 2]; 5] = [
    [-0.62, 0.5],
    [-0.45, 0.0],
    [0.0, 0.0],
    [0.25, 0.06],
    [0.5, 0.16],
];
const EXTENDED: [[f32; 2]; 5] = [
    [-1.15, 0.32],
    [-0.5, -1.3],
    [0.0, -1.42],
    [0.26, -1.3],
    [0.52, -1.08],
];
const CURLED: [[f32; 2]; 5] = [
    [-0.5, 0.42],
    [-0.36, 0.34],
    [-0.04, 0.3],
    [0.2, 0.32],
    [0.42, 0.36],
];

/// One synthetic hand. `scale` is the wrist-to-middle-MCP distance in
/// normalised units, i.e. the knob that simulates distance from the camera.
#[derive(Clone, Copy, Debug)]
pub struct Hand {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub gesture: i32,
    pub score: f32,
    pub pinched: bool,
    pub right: bool,
}

impl Default for Hand {
    fn default() -> Self {
        Self {
            x: 0.5,
            y: 0.5,
            scale: 0.13,
            gesture: NONE,
            score: 0.92,
            pinched: false,
            right: true,
        }
    }
}

impl Hand {
    pub fn at(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Self::default()
        }
    }

    pub fn gesture(mut self, id: i32) -> Self {
        self.gesture = id;
        self
    }

    pub fn score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    pub fn pinched(mut self) -> Self {
        self.pinched = true;
        self
    }

    pub fn scaled(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// Which fingers a canned label implies are extended.
fn extension(gesture: i32) -> [bool; 5] {
    match gesture {
        OPEN_PALM => [true; 5],
        CLOSED_FIST => [false; 5],
        POINTING_UP => [false, true, false, false, false],
        VICTORY => [false, true, true, false, false],
        THUMB_UP => [true, false, false, false, false],
        _ => [true, true, true, false, false],
    }
}

pub fn buffer() -> Vec<f32> {
    vec![0.0; HAND_BUFFER]
}

/// Zeroes a slot, i.e. reports the hand as absent.
pub fn clear(buf: &mut [f32], slot: usize) {
    let base = slot * HAND_STRIDE;
    buf[base..base + HAND_STRIDE].fill(0.0);
}

pub fn write(buf: &mut [f32], slot: usize, hand: &Hand) {
    let base = slot * HAND_STRIDE;
    buf[base] = 1.0;
    buf[base + 1] = if hand.right { 1.0 } else { 0.0 };
    buf[base + 2] = hand.gesture as f32;
    buf[base + 3] = hand.score;

    let mut put = |id: usize, lx: f32, ly: f32| {
        let o = base + 4 + id * 3;
        buf[o] = hand.x + lx * hand.scale;
        buf[o + 1] = hand.y + ly * hand.scale;
        buf[o + 2] = 0.0;
    };

    // The wrist sits exactly one hand scale from the middle MCP, which is
    // what makes `hand.scale` the measured scale.
    put(0, 0.0, 1.0);

    let ext = extension(hand.gesture);
    for (c, chain) in CHAINS.iter().enumerate() {
        let [bx, by] = BASES[c];
        let [tx, ty] = if ext[c] { EXTENDED[c] } else { CURLED[c] };
        for (j, &id) in chain.iter().enumerate() {
            let t = j as f32 / (chain.len() - 1) as f32;
            put(id, bx + (tx - bx) * t, by + (ty - by) * t);
        }
    }

    if hand.pinched {
        put(4, -0.5, -0.62);
        put(8, -0.44, -0.72);
    }
}
