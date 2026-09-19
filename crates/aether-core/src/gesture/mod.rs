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
    /// A held fist opened: an expanding ring shockwave. Reported by the spell
    /// layer for the moment after the release; never latched by the tracker.
    Release,
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
            Self::Release => "release",
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
            Self::Release => 0.62,
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

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }
}

mod internals;
#[cfg(test)]
pub(crate) mod synth;
#[cfg(test)]
mod tests;

pub use internals::ema_alpha;
use internals::*;
