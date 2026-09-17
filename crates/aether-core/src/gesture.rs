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

use crate::config::{HANDS, HAND_LANDMARKS, POSE_LANDMARKS};

/// MediaPipe's canned gesture labels, in classifier index order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CannedGesture {
    #[default]
    None,
    ClosedFist,
    OpenPalm,
    PointingUp,
    ThumbDown,
    ThumbUp,
    Victory,
    ILoveYou,
}

impl CannedGesture {
    /// Maps the classifier index the JS side sends. Unknown -> `None`.
    pub fn from_id(id: i32) -> Self {
        match id {
            1 => Self::ClosedFist,
            2 => Self::OpenPalm,
            3 => Self::PointingUp,
            4 => Self::ThumbDown,
            5 => Self::ThumbUp,
            6 => Self::Victory,
            7 => Self::ILoveYou,
            _ => Self::None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ClosedFist => "closed_fist",
            Self::OpenPalm => "open_palm",
            Self::PointingUp => "pointing_up",
            Self::ThumbDown => "thumb_down",
            Self::ThumbUp => "thumb_up",
            Self::Victory => "victory",
            Self::ILoveYou => "i_love_you",
        }
    }
}

/// What the engine actually does with a hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Spell {
    /// Hand present but not commanding anything; still pushes the fluid.
    #[default]
    Idle,
    /// Pinched fingers spin a vortex around the pinch point.
    Vortex,
    /// Open palm pushes fluid and particles away.
    Repel,
    /// Closed fist pulls everything into a gravity well.
    Attract,
    /// Pointing finger paints a hot dye trail.
    Ignite,
    /// Victory sign chills the fluid: heavy damping, frost palette.
    Freeze,
    /// Thumb up bursts particles outward.
    Shatter,
}

impl Spell {
    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Vortex => "vortex",
            Self::Repel => "repel",
            Self::Attract => "attract",
            Self::Ignite => "ignite",
            Self::Freeze => "freeze",
            Self::Shatter => "shatter",
        }
    }

    /// Hue (in turns) the spell tints its dye with.
    pub fn hue(self) -> f32 {
        match self {
            Self::Idle => 0.55,
            Self::Vortex => 0.74,
            Self::Repel => 0.45,
            Self::Attract => 0.87,
            Self::Ignite => 0.05,
            Self::Freeze => 0.55,
            Self::Shatter => 0.14,
        }
    }
}

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

    /// Ingests a packed hand buffer. See the module docs for the layout.
    ///
    /// AGENT-IMPLEMENT: the placeholder copies landmarks through unfiltered and
    /// never latches a spell beyond `Idle`.
    pub fn update_hands(&mut self, packed: &[f32], dt: f32) {
        use crate::config::HAND_STRIDE;
        let _ = dt;
        for slot in 0..HANDS {
            let base = slot * HAND_STRIDE;
            if base + HAND_STRIDE > packed.len() {
                break;
            }
            let hand = &mut self.hands[slot];
            hand.present = packed[base] > 0.5;
            if !hand.present {
                self.missing[slot] = self.missing[slot].saturating_add(1);
                continue;
            }
            self.missing[slot] = 0;
            self.initialised[slot] = true;
            hand.right = packed[base + 1] > 0.5;
            hand.canned = CannedGesture::from_id(packed[base + 2] as i32);
            hand.canned_score = packed[base + 3];
            for l in 0..HAND_LANDMARKS {
                let o = base + 4 + l * 3;
                hand.landmarks[l] = [packed[o], packed[o + 1], packed[o + 2]];
            }
            hand.palm = [
                hand.landmarks[crate::config::LM_WRIST][0],
                hand.landmarks[crate::config::LM_WRIST][1],
            ];
            hand.spell = Spell::Idle;
        }
        let _ = &mut self.candidate;
    }

    /// Ingests a packed pose buffer.
    ///
    /// AGENT-IMPLEMENT: the placeholder copies landmarks through unfiltered.
    pub fn update_pose(&mut self, packed: &[f32], dt: f32) {
        let _ = dt;
        if packed.is_empty() {
            return;
        }
        self.pose.present = packed[0] > 0.5;
        if !self.pose.present {
            return;
        }
        self.pose_initialised = true;
        for l in 0..POSE_LANDMARKS {
            let o = 1 + l * 4;
            if o + 3 >= packed.len() {
                break;
            }
            self.pose.landmarks[l] = [packed[o], packed[o + 1], packed[o + 2]];
            self.pose.visibility[l] = packed[o + 3];
        }
    }

    /// Recomputes [`TwoHandState`] from the current hand states.
    ///
    /// AGENT-IMPLEMENT.
    pub fn update_two_hand(&mut self, dt: f32) {
        let _ = dt;
        self.two.both_present = self.hands[0].present && self.hands[1].present;
        let _ = &mut self.two_initialised;
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }
}
