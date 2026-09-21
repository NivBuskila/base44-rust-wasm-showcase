//! The filtered per-frame state the tracker publishes, plus its tuning config.
//!
//! Split out of `gesture/mod.rs` the way `flow/` and `mask/` are: the parent
//! keeps the struct and the tuning constants, and `use super::*` brings both in
//! rather than an import list that grows with every constant.

use super::*;

/// Filtered state of one hand. All positions are normalised `[0, 1]`.
#[derive(Clone, Debug)]
pub struct HandState {
    pub present: bool,
    /// True for a right hand, as labelled by MediaPipe's handedness head.
    pub right: bool,
    /// Filtered landmarks.
    pub landmarks: [[f32; 3]; HAND_LANDMARKS],
    /// Palm centre (mean of wrist and the finger MCP joints).
    pub palm: [f32; 2],
    pub index_tip: [f32; 2],
    pub thumb_tip: [f32; 2],
    /// Palm velocity in normalised units per second.
    pub velocity: [f32; 2],
    /// 1.0 = thumb and index touching, 0.0 = far apart.
    pub pinch: f32,
    /// 0.0 = fist, 1.0 = fingers fully extended.
    pub openness: f32,
    /// Hand size proxy (wrist to middle MCP), used to normalise distances so
    /// the gestures behave the same near and far from the camera.
    pub scale: f32,
    pub canned: CannedGesture,
    pub canned_score: f32,
    /// The latched spell after temporal commit.
    pub spell: Spell,
    /// How long the current spell has been held, in seconds.
    pub spell_age: f32,
}

impl Default for HandState {
    fn default() -> Self {
        Self {
            present: false,
            right: false,
            landmarks: [[0.0; 3]; HAND_LANDMARKS],
            palm: [0.5, 0.5],
            index_tip: [0.5, 0.5],
            thumb_tip: [0.5, 0.5],
            velocity: [0.0, 0.0],
            pinch: 0.0,
            openness: 0.0,
            scale: 0.1,
            canned: CannedGesture::None,
            canned_score: 0.0,
            spell: Spell::Idle,
            spell_age: 0.0,
        }
    }
}

/// Filtered body pose.
#[derive(Clone, Debug)]
pub struct PoseState {
    pub present: bool,
    pub landmarks: [[f32; 3]; POSE_LANDMARKS],
    pub visibility: [f32; POSE_LANDMARKS],
    /// Torso centre (mean of shoulders and hips).
    pub center: [f32; 2],
    /// Shoulder width; a rough distance-to-camera proxy.
    pub span: f32,
    /// Torso centre velocity, normalised units per second.
    pub velocity: [f32; 2],
}

impl Default for PoseState {
    fn default() -> Self {
        Self {
            present: false,
            landmarks: [[0.0; 3]; POSE_LANDMARKS],
            visibility: [0.0; POSE_LANDMARKS],
            center: [0.5, 0.5],
            span: 0.2,
            velocity: [0.0, 0.0],
        }
    }
}

/// Derived two-handed state — the "conductor" gestures.
#[derive(Clone, Copy, Debug, Default)]
pub struct TwoHandState {
    pub both_present: bool,
    /// Normalised distance between the palms.
    pub distance: f32,
    /// Rate of change of `distance`, per second.
    pub distance_velocity: f32,
    pub midpoint: [f32; 2],
    /// Angle of the line between palms, radians.
    pub angle: f32,
    /// Rate of change of `angle`, radians per second. Drives the time warp.
    pub angular_velocity: f32,
}

/// Temporal filter settings.
#[derive(Clone, Copy, Debug)]
pub struct GestureConfig {
    /// EMA smoothing half-life for landmark positions, seconds.
    pub position_half_life: f32,
    /// EMA half-life for velocities, seconds.
    pub velocity_half_life: f32,
    /// Consecutive frames a candidate spell must win before latching.
    pub commit_frames: u32,
    /// Consecutive frames without a hand before it is declared gone.
    pub release_frames: u32,
    /// Minimum canned-gesture score to consider the label at all.
    pub min_score: f32,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            position_half_life: 0.045,
            velocity_half_life: 0.09,
            commit_frames: 3,
            release_frames: 6,
            min_score: 0.5,
        }
    }
}
