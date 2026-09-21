//! Turns raw landmarks into a stable, smoothed gesture state.
//!
//! MediaPipe's per-frame output is jittery and its canned classifier flickers
//! between labels on adjacent frames. Mapping it straight onto forces makes the
//! visuals twitch, so everything here is temporally filtered:
//!
//! - landmark positions pass through a one-euro / EMA filter,
//! - velocities are finite differences of the *filtered* positions,
//! - a spell must hold for `commit_frames` consecutive frames before it
//!   latches, and must be absent for `release_frames` before it clears.
//!
//! ## Contract
//!
//! Packed input layouts are defined in [`crate::config`]:
//! hands are `HANDS * HAND_STRIDE` floats, `present, handedness, gesture_id,
//! gesture_score` then 21 * `(x, y, z)`; pose is `POSE_STRIDE` floats,
//! `present` then 33 * `(x, y, z, visibility)`. All coordinates arrive
//! normalised to `[0, 1]` with y down, already mirrored to match the flipped
//! camera view the user sees.

use crate::config::{
    HANDS, HAND_LANDMARKS, HAND_STRIDE, LM_INDEX_MCP, LM_INDEX_TIP, LM_MIDDLE_MCP, LM_MIDDLE_TIP,
    LM_PINKY_MCP, LM_PINKY_TIP, LM_RING_TIP, LM_THUMB_TIP, LM_WRIST, PL_LEFT_HIP, PL_LEFT_SHOULDER,
    PL_RIGHT_HIP, PL_RIGHT_SHOULDER, POSE_LANDMARKS, POSE_STRIDE,
};
use crate::math::{decay, smoothstep};

/// Ring-finger MCP. Deliberately not in [`crate::config`] — nothing outside the
/// palm centroid needs it, and that centroid is noticeably steadier with all
/// four knuckles than with the three that are exported.
const LM_RING_MCP: usize = 13;

/// Joints averaged into [`HandState::palm`]. The wrist alone is what MediaPipe
/// tracks most confidently, but it sits at the edge of the hand, so a wrist-only
/// "palm" makes every force feel like it comes out of the user's forearm.
const PALM_JOINTS: [usize; 5] = [
    LM_WRIST,
    LM_INDEX_MCP,
    LM_MIDDLE_MCP,
    LM_RING_MCP,
    LM_PINKY_MCP,
];

/// Fingertips used for [`HandState::openness`]. The thumb is excluded: it folds
/// across the palm rather than along it, so its tip-to-wrist distance barely
/// changes between a fist and an open hand and only adds noise.
const OPEN_TIPS: [usize; 4] = [LM_INDEX_TIP, LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP];

/// Floor on the hand-size proxy. Every scale-normalised feature divides by it,
/// and MediaPipe can emit a fully degenerate (all-zero) hand.
const MIN_SCALE: f32 = 1e-3;

/// Smallest wrist-to-knuckle span, in normalised units, for the scale-normalised
/// features to mean anything.
///
/// A hand can arrive `present` with every landmark collapsed onto one point.
/// That is still a tracked hand — presence is the perception layer's call, not
/// this module's — but dividing by its size is not: a zero-span hand has its
/// thumb and index tips touching by construction, so it would read as a perfect
/// pinch and conjure a vortex out of nothing.
const MIN_CREDIBLE_SCALE: f32 = 0.005;

/// Thumb-to-index distance, in hand scales, at which the pinch reads 0 and 1.
/// Measured on real hands: a relaxed open hand sits near 0.9, fingers touching
/// near 0.12. The window starts well below the open value so an idle hand never
/// drifts into a vortex.
const PINCH_FAR: f32 = 0.55;
const PINCH_NEAR: f32 = 0.18;

/// Mean fingertip-to-wrist distance, in hand scales, mapping to openness 0 and 1.
const OPEN_CLOSED: f32 = 1.0;
const OPEN_EXTENDED: f32 = 2.1;

/// Pinch strength that proposes [`Spell::Vortex`].
const PINCH_COMMIT: f32 = 0.6;

/// Distance from the wrist to the thumb/index midpoint, in hand scales, below
/// which a small thumb-to-index gap is read as a *fist* rather than a pinch.
///
/// Thumb-to-index distance alone cannot tell the two apart: on a closed fist
/// the two tips end up ~0.16 scales apart, which is a perfect pinch by that
/// measure. What differs is where those tips are — a pinch holds them out in
/// front of the knuckles (~1.7 scales from the wrist), a fist folds them back
/// onto the palm (~0.75). Without this gate the geometric path reads every
/// unlabelled fist as a vortex, and [`OPEN_FALLBACK_ATTRACT`] below is
/// unreachable, because any hand closed enough to hit it is already "pinching".
const PINCH_REACH: f32 = 1.2;

/// Openness bounds for the model-free fallback, used only when the canned label
/// is missing or unconfident. Wider than the label thresholds on purpose: this
/// path should fire on unambiguous poses and stay out of the way otherwise.
const OPEN_FALLBACK_REPEL: f32 = 0.8;
const OPEN_FALLBACK_ATTRACT: f32 = 0.12;

/// One-euro adaptivity: how much measured speed shortens the position filter's
/// half-life, in (normalised units per second)^-1.
///
/// A single fixed EMA coefficient forces a bad choice — heavy enough to kill
/// MediaPipe's jitter when the hand is still is heavy enough to lag visibly
/// when it moves fast, and a lagging hand feels like the fluid is stuck to
/// something behind the user's hand. Scaling the cutoff with speed removes the
/// choice: at rest the filter is slow and the hand is rock steady, at 1 screen
/// width per second the half-life is a quarter of its base value.
const ADAPT_BETA: f32 = 3.0;

/// Floor on the adapted half-life, so a violent frame cannot disable filtering.
const MIN_HALF_LIFE: f32 = 0.004;

/// Half-life multiplier for pose landmarks. The torso is a low-bandwidth signal
/// and the segmentation centre it feeds is used for slow, wide forces, so the
/// jitter-versus-lag tradeoff that justifies the adaptive hand filter does not
/// apply: more smoothing is simply better.
const POSE_HALF_LIFE_SCALE: f32 = 2.0;

/// Visibility below which a pose landmark is treated as not measured at all.
const MIN_VISIBILITY: f32 = 0.5;

/// Rate at which a coasting hand's velocity bleeds off, per second.
///
/// While a hand is missing but not yet released, its landmarks are stale. Held
/// velocity would keep shoving the fluid from a frozen position for the whole
/// release window, which reads as the fluid being dragged by a ghost.
const COAST_DECAY: f32 = 8.0;

/// Timestep bounds. `update_*` are public and must not be made to integrate a
/// backgrounded tab's ten-second gap, nor to divide a finite difference by zero.
const MIN_DT: f32 = 1.0 / 2000.0;
const MAX_DT: f32 = 0.5;

/// Owns the filtered gesture state across frames.
pub struct GestureTracker {
    cfg: GestureConfig,
    hands: [HandState; HANDS],
    pose: PoseState,
    two: TwoHandState,
    /// Frames each hand has been missing.
    missing: [u32; HANDS],
    /// Frames the current spell candidate has been winning.
    candidate: [(Spell, u32); HANDS],
    initialised: [bool; HANDS],
    pose_initialised: bool,
    two_initialised: bool,
    /// Previous *raw* palm position, for the one-euro speed estimate. Taken
    /// before filtering on purpose: the adaptive cutoff has to react to the
    /// input's bandwidth, and the filtered signal is exactly what has had that
    /// bandwidth removed.
    raw_palm: [[f32; 2]; HANDS],
    /// Smoothed raw palm speed driving the adaptive cutoff.
    speed_est: [f32; HANDS],
    /// Previous *filtered* palm, for [`HandState::velocity`].
    prev_palm: [[f32; 2]; HANDS],
    /// Which pose landmarks have ever been seen above [`MIN_VISIBILITY`], so an
    /// invisible joint stays at the origin instead of being filtered from it.
    pose_seen: [bool; POSE_LANDMARKS],
    prev_center: [f32; 2],
    pose_missing: u32,
    /// Previous palm-pair angle, wrapped. The *step* between frames is
    /// unwrapped before differentiating, which is what keeps the time warp from
    /// spiking once per revolution.
    last_angle: f32,
    last_distance: f32,
}

impl GestureTracker {
    pub fn new(cfg: GestureConfig) -> Self {
        Self {
            cfg,
            hands: [HandState::default(), HandState::default()],
            pose: PoseState::default(),
            two: TwoHandState::default(),
            missing: [u32::MAX; HANDS],
            candidate: [(Spell::Idle, 0); HANDS],
            initialised: [false; HANDS],
            pose_initialised: false,
            two_initialised: false,
            raw_palm: [[0.5, 0.5]; HANDS],
            speed_est: [0.0; HANDS],
            prev_palm: [[0.5, 0.5]; HANDS],
            pose_seen: [false; POSE_LANDMARKS],
            prev_center: [0.5, 0.5],
            pose_missing: u32::MAX,
            last_angle: 0.0,
            last_distance: 0.0,
        }
    }

    #[inline]
    pub fn config(&self) -> &GestureConfig {
        &self.cfg
    }

    #[inline]
    pub fn hands(&self) -> &[HandState; HANDS] {
        &self.hands
    }

    #[inline]
    pub fn pose(&self) -> &PoseState {
        &self.pose
    }

    #[inline]
    pub fn two_hand(&self) -> &TwoHandState {
        &self.two
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }
}

mod hands;
mod internals;
mod kinds;
mod pose;
mod state;
#[cfg(test)]
pub(crate) mod synth;
#[cfg(test)]
mod tests;
mod two_hand;

pub use internals::ema_alpha;
use internals::*;
pub use kinds::{CannedGesture, Spell};
pub use state::{GestureConfig, HandState, PoseState, TwoHandState};
