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

/// Seconds an open palm must have been repelling before holding it still turns
/// it into a sculpt. A palm that is thrown open pushes first; only a deliberate
/// hold gathers.
const SCULPT_DWELL: f32 = 0.5;
/// Palm speed, normalised units per second, below which a held palm is "still"
/// and the sculpt may begin, and above which a sculpting hand is read as a
/// sweep and goes back to repelling. The gap between them is the hysteresis
/// that keeps a slightly trembling hand from flickering between the two.
const SCULPT_ENTER_SPEED: f32 = 0.10;
const SCULPT_EXIT_SPEED: f32 = 0.40;

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
    /// Open palm held still: particles gather onto the hand's own skeleton.
    Sculpt,
    /// Both hands pointing: a beam strung between the two index tips.
    Beam,
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
            Self::Sculpt => "sculpt",
            Self::Beam => "beam",
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
            Self::Sculpt => 0.36,
            Self::Beam => 0.97,
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

// ---------------------------------------------------------------- internals

/// EMA coefficient for a half-life: `alpha = 1 - 2^(-dt / half_life)`.
///
/// Written through [`decay`] so it composes exactly across frame rates — ten
/// 10 ms steps and one 100 ms step leave the same fraction of the error, which
/// is the whole reason the config is expressed as a half-life rather than as a
/// per-frame coefficient.
#[inline]
pub fn ema_alpha(half_life: f32, dt: f32) -> f32 {
    if !half_life.is_finite() || half_life <= 0.0 {
        return 1.0;
    }
    1.0 - decay(core::f32::consts::LN_2 / half_life, dt)
}

/// [`ema_alpha`] with the one-euro speed-dependent cutoff applied.
#[inline]
fn adaptive_alpha(half_life: f32, speed: f32, dt: f32) -> f32 {
    let speed = finite_or(speed, 0.0).abs();
    let adapted = (half_life / (1.0 + ADAPT_BETA * speed)).max(MIN_HALF_LIFE);
    ema_alpha(adapted, dt)
}

#[inline]
fn clamp_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(MIN_DT, MAX_DT)
}

#[inline]
fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

#[inline]
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    crate::math::length(a[0] - b[0], a[1] - b[1])
}

/// Folds an angle into `[-pi, pi]`.
#[inline]
fn wrap_pi(a: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    if !a.is_finite() {
        return 0.0;
    }
    (a + PI).rem_euclid(TAU) - PI
}

fn centroid(lm: &[[f32; 3]; HAND_LANDMARKS], joints: &[usize]) -> [f32; 2] {
    let mut sum = [0.0f32, 0.0];
    for &j in joints {
        sum[0] += lm[j][0];
        sum[1] += lm[j][1];
    }
    let n = joints.len().max(1) as f32;
    [sum[0] / n, sum[1] / n]
}

/// Extracts one hand's landmarks, or `None` if the slot is absent, truncated or
/// non-finite.
///
/// A single NaN coordinate rejects the whole hand: it would otherwise enter the
/// EMA state and poison every later frame, and one hand's worth of NaN in the
/// fluid blanks the screen.
fn read_hand(packed: &[f32], base: usize) -> Option<[[f32; 3]; HAND_LANDMARKS]> {
    if base + HAND_STRIDE > packed.len() {
        return None;
    }
    // A NaN presence flag must read as absent, which `flag <= 0.5` would get
    // backwards.
    let flagged = packed[base] > 0.5;
    if !flagged {
        return None;
    }
    let mut out = [[0.0f32; 3]; HAND_LANDMARKS];
    for (l, dst) in out.iter_mut().enumerate() {
        let o = base + 4 + l * 3;
        let (x, y) = (packed[o], packed[o + 1]);
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        *dst = [x, y, finite_or(packed[o + 2], 0.0)];
    }
    Some(out)
}

/// The spell this frame's geometry and label argue for, before hysteresis.
///
/// A confident canned label wins outright; the geometric paths exist because
/// the classifier is frequently unconfident and occasionally unavailable, and a
/// hand that produces no spell at all feels broken.
fn candidate_spell(hand: &HandState, cfg: &GestureConfig) -> Spell {
    let base = base_candidate(hand, cfg);
    if base != Spell::Repel {
        return base;
    }
    // An open palm is two spells depending on what it is doing: sweeping
    // repels, holding still sculpts. Decided against the *latched* spell so the
    // transition has memory in both directions.
    let speed = crate::math::length(hand.velocity[0], hand.velocity[1]);
    match hand.spell {
        Spell::Sculpt if speed < SCULPT_EXIT_SPEED => Spell::Sculpt,
        Spell::Repel if hand.spell_age >= SCULPT_DWELL && speed < SCULPT_ENTER_SPEED => {
            Spell::Sculpt
        }
        _ => Spell::Repel,
    }
}

/// [`candidate_spell`] before the open-palm split.
fn base_candidate(hand: &HandState, cfg: &GestureConfig) -> Spell {
    if hand.canned_score >= cfg.min_score {
        if let Some(spell) = spell_for_label(hand.canned) {
            return spell;
        }
    }
    // Below that the features are not measurements, so the geometric fallbacks
    // must not vote — `openness` of 0 would otherwise read as a fist.
    if hand.scale < MIN_CREDIBLE_SCALE {
        return Spell::Idle;
    }
    if hand.pinch >= PINCH_COMMIT && pinch_reach(hand) >= PINCH_REACH {
        return Spell::Vortex;
    }
    if hand.openness >= OPEN_FALLBACK_REPEL {
        Spell::Repel
    } else if hand.openness <= OPEN_FALLBACK_ATTRACT {
        Spell::Attract
    } else {
        Spell::Idle
    }
}

/// How far the thumb/index midpoint sits from the wrist, in hand scales.
/// See [`PINCH_REACH`]: this is what separates a pinch from a fist.
#[inline]
fn pinch_reach(hand: &HandState) -> f32 {
    let wrist = [hand.landmarks[LM_WRIST][0], hand.landmarks[LM_WRIST][1]];
    let mid = [
        (hand.index_tip[0] + hand.thumb_tip[0]) * 0.5,
        (hand.index_tip[1] + hand.thumb_tip[1]) * 0.5,
    ];
    distance(wrist, mid) / hand.scale.max(MIN_SCALE)
}

/// Canned label to spell. `None` for labels Aether does not bind, which fall
/// through to the geometric paths.
fn spell_for_label(canned: CannedGesture) -> Option<Spell> {
    match canned {
        CannedGesture::ClosedFist => Some(Spell::Attract),
        CannedGesture::OpenPalm => Some(Spell::Repel),
        CannedGesture::PointingUp => Some(Spell::Ignite),
        CannedGesture::Victory => Some(Spell::Freeze),
        CannedGesture::ThumbUp => Some(Spell::Shatter),
        CannedGesture::None | CannedGesture::ThumbDown | CannedGesture::ILoveYou => None,
    }
}

/// Synthetic landmark packing, shared by this module's tests and
/// [`crate::spells`]'s.
///
/// The joint layout is a transcription of the scripted perception source in
/// `web/tests/gesture.spec.ts`, so the host tests and the browser end-to-end
/// suite assert against the *same* geometry instead of two independent
/// idealisations of a hand — a threshold that passes here therefore also passes
/// there.
#[cfg(test)]
pub(crate) mod synth {
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
}

#[cfg(test)]
mod tests {
    use super::synth::{self, Hand};
    use super::*;
    use crate::config::POSE_STRIDE;

    const DT: f32 = 1.0 / 30.0;

    fn tracker() -> GestureTracker {
        GestureTracker::new(GestureConfig::default())
    }

    /// Feeds one hand for `frames` frames and returns the tracker.
    fn feed(t: &mut GestureTracker, hand: &Hand, frames: usize, dt: f32) {
        let mut buf = synth::buffer();
        for _ in 0..frames {
            synth::write(&mut buf, 0, hand);
            t.update_hands(&buf, dt);
            t.update_two_hand(dt);
        }
    }

    /// A pose buffer with `present` set and only the listed joints visible.
    fn pose_buffer(joints: &[(usize, f32, f32, f32)]) -> Vec<f32> {
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

    #[test]
    fn ema_alpha_composes_across_frame_rates() {
        // The claim the whole filter rests on: the surviving fraction of an
        // error after 100 ms must not depend on how many steps it took to get
        // there. `1 - rate * dt` fails this; a half-life does not.
        let one = 1.0 - ema_alpha(0.045, 0.1);
        let mut many = 1.0f32;
        for _ in 0..10 {
            many *= 1.0 - ema_alpha(0.045, 0.01);
        }
        assert!((one - many).abs() < 1e-6, "{one} vs {many}");
        // And a half-life means what it says.
        assert!((ema_alpha(0.05, 0.05) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ema_alpha_and_dt_survive_degenerate_input() {
        assert_eq!(ema_alpha(0.0, 0.016), 1.0, "zero half-life must snap");
        assert_eq!(ema_alpha(f32::NAN, 0.016), 1.0);
        assert!(ema_alpha(0.05, f32::INFINITY).is_finite());
        assert_eq!(clamp_dt(f32::NAN), 1.0 / 60.0);
        assert_eq!(clamp_dt(-5.0), MIN_DT);
        assert_eq!(clamp_dt(1e9), MAX_DT);
        assert!(adaptive_alpha(0.045, f32::NAN, DT).is_finite());
    }

    #[test]
    fn landmark_filter_is_framerate_independent() {
        // Two trackers see the same step input, one at 100 Hz and one at 10 Hz.
        // After the same 100 ms of simulated time they must agree.
        let cfg = GestureConfig::default();
        let (mut fast, mut slow) = (GestureTracker::new(cfg), GestureTracker::new(cfg));
        let start = Hand::at(0.40, 0.5);
        let moved = Hand::at(0.42, 0.5);
        feed(&mut fast, &start, 1, 0.01);
        feed(&mut slow, &start, 1, 0.1);
        feed(&mut fast, &moved, 10, 0.01);
        feed(&mut slow, &moved, 1, 0.1);

        let (a, b) = (fast.hands()[0].palm[0], slow.hands()[0].palm[0]);
        let step = 0.02;
        // 2% of the step. Not zero: the one-euro cutoff reads the step as a
        // ten-times-higher speed at the shorter dt, so the two paths converge
        // at slightly different rates on the first frame after a discontinuity.
        assert!(
            (a - b).abs() < 0.02 * step,
            "100 Hz and 10 Hz disagree: {a} vs {b}"
        );
        // And prove filtering actually happened rather than both snapping.
        let target = slow.hands()[0].palm[0];
        assert!(target > 0.40, "filter did not move toward the target");
    }

    #[test]
    fn landmark_filter_rejects_jitter() {
        // Alternating noise is the signal MediaPipe actually produces when a
        // hand is held still; the filter exists to remove it.
        let mut t = tracker();
        let mut buf = synth::buffer();
        let mut raw_span = 0.0f32;
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for frame in 0..60 {
            let jitter = if frame % 2 == 0 { 0.004 } else { -0.004 };
            let hand = Hand::at(0.5 + jitter, 0.5);
            synth::write(&mut buf, 0, &hand);
            t.update_hands(&buf, DT);
            if frame >= 40 {
                raw_span = 0.008;
                lo = lo.min(t.hands()[0].palm[0]);
                hi = hi.max(t.hands()[0].palm[0]);
            }
        }
        let filtered_span = hi - lo;
        assert!(
            filtered_span * 1.8 < raw_span,
            "jitter barely attenuated: {filtered_span} vs {raw_span}"
        );
    }

    #[test]
    fn pinch_is_scale_invariant() {
        // The same pose near the camera and far from it. If pinch used raw
        // normalised distance this would differ by the scale ratio, and the
        // gesture would only work with the hand right up against the lens.
        let mut near = tracker();
        let mut far = tracker();
        feed(
            &mut near,
            &Hand::at(0.5, 0.5).pinched().scaled(0.16),
            20,
            DT,
        );
        feed(
            &mut far,
            &Hand::at(0.5, 0.5).pinched().scaled(0.045),
            20,
            DT,
        );
        let (a, b) = (near.hands()[0].pinch, far.hands()[0].pinch);
        assert!(
            (a - b).abs() < 0.02,
            "pinch not scale invariant: {a} vs {b}"
        );
        assert!(a > 0.9, "a touching pinch should read ~1, got {a}");

        // Openness too, and with the same hand at both distances.
        let mut near = tracker();
        let mut far = tracker();
        feed(
            &mut near,
            &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM).scaled(0.16),
            20,
            DT,
        );
        feed(
            &mut far,
            &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM).scaled(0.045),
            20,
            DT,
        );
        let (a, b) = (near.hands()[0].openness, far.hands()[0].openness);
        assert!((a - b).abs() < 0.02, "openness not scale invariant");
    }

    #[test]
    fn pinch_and_openness_separate_the_poses() {
        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM),
            20,
            DT,
        );
        assert!(t.hands()[0].pinch < 0.05, "open hand read as pinched");
        assert!(t.hands()[0].openness > 0.95, "open hand read as closed");

        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST),
            20,
            DT,
        );
        assert!(t.hands()[0].openness < 0.05, "fist read as open");

        let mut t = tracker();
        feed(&mut t, &Hand::at(0.5, 0.5).pinched(), 20, DT);
        assert!(t.hands()[0].pinch > 0.9, "touching fingers read as open");
    }

    #[test]
    fn a_spell_needs_commit_frames_to_latch() {
        let cfg = GestureConfig::default();
        let mut t = GestureTracker::new(cfg);
        let hand = Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM);
        for frame in 1..cfg.commit_frames {
            feed(&mut t, &hand, 1, DT);
            assert_eq!(
                t.hands()[0].spell,
                Spell::Idle,
                "latched after only {frame} frames"
            );
        }
        feed(&mut t, &hand, 1, DT);
        assert_eq!(t.hands()[0].spell, Spell::Repel);
        assert!(t.hands()[0].spell_age < 1e-6, "age should restart on latch");
        feed(&mut t, &hand, 3, DT);
        assert!(t.hands()[0].spell_age > 2.0 * DT, "age should accumulate");
    }

    #[test]
    fn a_one_frame_flicker_does_not_dislodge_the_latched_spell() {
        let mut t = tracker();
        let fist = Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST);
        feed(&mut t, &fist, 10, DT);
        assert_eq!(t.hands()[0].spell, Spell::Attract);

        // MediaPipe really does emit one frame of the wrong label mid-hold.
        feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 1, DT);
        assert_eq!(
            t.hands()[0].spell,
            Spell::Attract,
            "one frame of Open_Palm dislodged the latch"
        );
        feed(&mut t, &fist, 1, DT);
        assert_eq!(t.hands()[0].spell, Spell::Attract);

        // Held, though, it must win.
        feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 4, DT);
        assert_eq!(t.hands()[0].spell, Spell::Repel);
    }

    #[test]
    fn canned_labels_map_to_their_spells() {
        for (gesture, want) in [
            (synth::OPEN_PALM, Spell::Repel),
            (synth::CLOSED_FIST, Spell::Attract),
            (synth::POINTING_UP, Spell::Ignite),
            (synth::VICTORY, Spell::Freeze),
            (synth::THUMB_UP, Spell::Shatter),
        ] {
            let mut t = tracker();
            feed(&mut t, &Hand::at(0.5, 0.5).gesture(gesture), 8, DT);
            assert_eq!(t.hands()[0].spell, want, "gesture {gesture} mismapped");
        }
    }

    #[test]
    fn a_pinch_beats_a_weak_label_but_not_a_confident_one() {
        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched(),
            8,
            DT,
        );
        assert_eq!(t.hands()[0].spell, Spell::Vortex);

        // A fist has the thumb and index tips close together too, so a pure
        // distance test would read it as a pinch; a confident label must win.
        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(0.95),
            8,
            DT,
        );
        assert_eq!(t.hands()[0].spell, Spell::Attract);
    }

    #[test]
    fn an_unlabelled_fist_is_a_gravity_well_not_a_vortex() {
        // A fist's thumb and index tips end up ~0.16 hand scales apart, i.e. a
        // perfect pinch by thumb-to-index distance alone. Whenever the canned
        // label is unconfident — which is exactly what the geometric fallback
        // exists for, and what happens for a few frames every time a hand
        // closes — a distance-only test spins a vortex out of a closed fist.
        for score in [0.49, 0.0] {
            let mut t = tracker();
            feed(
                &mut t,
                &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(score),
                10,
                DT,
            );
            let hand = &t.hands()[0];
            assert!(hand.pinch > 0.9, "the fist geometry should look pinched");
            assert_eq!(
                hand.spell,
                Spell::Attract,
                "an unlabelled fist (score {score}) should be a gravity well"
            );
        }

        // The pinch path must still win when the fingers are actually held out
        // in front of the knuckles, label or no label.
        let mut t = tracker();
        feed(&mut t, &Hand::at(0.5, 0.5).pinched().score(0.0), 10, DT);
        assert_eq!(t.hands()[0].spell, Spell::Vortex);

        // The two poses are separated by where the pinch sits, not how tight
        // it is: both read pinch ~1, and only the reach differs.
        let mut fist = tracker();
        feed(
            &mut fist,
            &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST).score(0.0),
            10,
            DT,
        );
        let a = pinch_reach(&fist.hands()[0]);
        let b = pinch_reach(&t.hands()[0]);
        assert!(a < PINCH_REACH, "fist reach {a} should be short");
        assert!(b > PINCH_REACH, "pinch reach {b} should be long");
    }

    #[test]
    fn a_closing_hand_does_not_flash_a_vortex_on_the_way_to_a_fist() {
        // MediaPipe's score collapses while the fingers are mid-curl. The
        // commit window is only 3 frames, so a spurious candidate that holds
        // for a third of a second really does latch.
        let mut t = tracker();
        let mut seen = Vec::new();
        for frame in 0..20 {
            let (g, score) = if frame < 5 {
                (synth::OPEN_PALM, 0.9)
            } else if frame < 14 {
                (synth::CLOSED_FIST, 0.2)
            } else {
                (synth::CLOSED_FIST, 0.9)
            };
            feed(&mut t, &Hand::at(0.5, 0.5).gesture(g).score(score), 1, DT);
            seen.push(t.hands()[0].spell);
        }
        assert!(
            !seen.contains(&Spell::Vortex),
            "a vortex flashed while the hand was closing: {seen:?}"
        );
        assert_eq!(seen[19], Spell::Attract);
    }

    #[test]
    fn a_brief_dropout_keeps_the_hand_present() {
        let cfg = GestureConfig::default();
        let mut t = GestureTracker::new(cfg);
        feed(&mut t, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), 8, DT);
        assert!(t.hands()[0].present);

        let empty = synth::buffer();
        for frame in 1..cfg.release_frames {
            t.update_hands(&empty, DT);
            assert!(
                t.hands()[0].present,
                "released after only {frame} missing frames"
            );
            assert_eq!(t.hands()[0].spell, Spell::Repel, "spell dropped too early");
        }
        t.update_hands(&empty, DT);
        assert!(!t.hands()[0].present, "hand should be gone by now");
        assert_eq!(t.hands()[0].spell, Spell::Idle);
        assert_eq!(t.hands()[0].velocity, [0.0, 0.0]);
    }

    #[test]
    fn velocity_tracks_a_known_sweep() {
        // 0.012 normalised units per frame at 30 Hz = 0.36 per second.
        let mut t = tracker();
        let mut buf = synth::buffer();
        let mut x = 0.2;
        for _ in 0..40 {
            synth::write(&mut buf, 0, &Hand::at(x, 0.5));
            t.update_hands(&buf, DT);
            x += 0.012;
        }
        let v = t.hands()[0].velocity;
        assert!(
            (v[0] - 0.36).abs() < 0.05,
            "expected ~0.36 units/s, got {}",
            v[0]
        );
        assert!(v[1].abs() < 0.02, "no vertical motion was applied");

        // A still hand must settle back to zero, or it would push the fluid
        // forever after one swipe.
        feed(&mut t, &Hand::at(x, 0.5), 40, DT);
        assert!(t.hands()[0].velocity[0].abs() < 0.02);
    }

    #[test]
    fn angular_velocity_is_continuous_through_the_pi_seam() {
        // Two palms rotating about the frame centre at a constant rate. The
        // pair's angle crosses +/-pi once per revolution; differentiating the
        // wrapped angle would spike by 2*pi/dt exactly there, and that spike
        // drives the time warp.
        let mut t = tracker();
        let mut buf = synth::buffer();
        let omega = 3.0f32;
        let radius = 0.12;
        let mut worst = 0.0f32;
        let frames = 160;
        for frame in 0..frames {
            let theta = omega * frame as f32 * DT;
            let (c, s) = (theta.cos(), theta.sin());
            synth::write(&mut buf, 0, &Hand::at(0.5 + radius * c, 0.5 + radius * s));
            synth::write(&mut buf, 1, &Hand::at(0.5 - radius * c, 0.5 - radius * s));
            t.update_hands(&buf, DT);
            t.update_two_hand(DT);
            if frame > 30 {
                let av = t.two_hand().angular_velocity;
                assert!(av.is_finite(), "non-finite angular velocity");
                assert!(
                    (av - omega).abs() < 0.5 * omega,
                    "spike at frame {frame}: {av} rad/s vs {omega}"
                );
                worst = worst.max((av - omega).abs());
            }
        }
        assert!(t.two_hand().both_present);
        assert!(worst < 0.5 * omega, "worst deviation {worst}");
        // A rigid rotation changes no distance.
        assert!(t.two_hand().distance_velocity.abs() < 0.2);
        assert!(t.two_hand().angle.abs() <= core::f32::consts::PI + 1e-5);
    }

    #[test]
    fn two_hand_distance_and_midpoint_track_the_palms() {
        let mut t = tracker();
        let mut buf = synth::buffer();
        // Hands separating by 0.01 each per frame, i.e. 0.6 units/s of spread.
        let mut half = 0.1;
        for _ in 0..40 {
            synth::write(&mut buf, 0, &Hand::at(0.5 - half, 0.5));
            synth::write(&mut buf, 1, &Hand::at(0.5 + half, 0.5));
            t.update_hands(&buf, DT);
            t.update_two_hand(DT);
            half += 0.01;
        }
        let two = *t.two_hand();
        assert!(two.both_present);
        assert!((two.distance - 2.0 * half).abs() < 0.05, "{}", two.distance);
        assert!(
            (two.distance_velocity - 0.6).abs() < 0.12,
            "separation rate {}",
            two.distance_velocity
        );
        assert!((two.midpoint[0] - 0.5).abs() < 0.02);

        // Losing one hand must zero the rates rather than freeze them.
        synth::clear(&mut buf, 1);
        for _ in 0..GestureConfig::default().release_frames {
            t.update_hands(&buf, DT);
            t.update_two_hand(DT);
        }
        assert!(!t.two_hand().both_present);
        assert_eq!(t.two_hand().angular_velocity, 0.0);
        assert_eq!(t.two_hand().distance_velocity, 0.0);
    }

    #[test]
    fn low_visibility_pose_landmarks_do_not_move_the_center() {
        let mut t = tracker();
        let torso = |hip: (f32, f32, f32)| {
            pose_buffer(&[
                (PL_LEFT_SHOULDER, 0.45, 0.35, 0.95),
                (PL_RIGHT_SHOULDER, 0.55, 0.35, 0.95),
                (PL_LEFT_HIP, 0.45, 0.70, 0.95),
                (PL_RIGHT_HIP, hip.0, hip.1, hip.2),
            ])
        };
        // The right hip is never credible, so its position must never count.
        let ghost = torso((0.55, 0.70, 0.05));
        for _ in 0..30 {
            t.update_pose(&ghost, DT);
        }
        let before = t.pose().center;
        assert!(t.pose().present);
        assert!((t.pose().span - 0.1).abs() < 0.01, "span {}", t.pose().span);

        // Now hallucinate it into the corner of the frame.
        let yanked = torso((0.98, 0.02, 0.05));
        for _ in 0..30 {
            t.update_pose(&yanked, DT);
        }
        let after = t.pose().center;
        assert!(
            (after[0] - before[0]).abs() < 1e-4 && (after[1] - before[1]).abs() < 1e-4,
            "an invisible joint moved the centre: {before:?} -> {after:?}"
        );

        // The same joint *is* credible: now it must count, or the gate would be
        // ignoring visibility rather than respecting it.
        let seen = torso((0.98, 0.02, 0.95));
        for _ in 0..40 {
            t.update_pose(&seen, DT);
        }
        assert!(
            (t.pose().center[0] - before[0]).abs() > 0.05,
            "a visible joint should move the centre"
        );
    }

    #[test]
    fn pose_center_and_velocity_track_the_torso() {
        let mut t = tracker();
        let mut x = 0.3;
        for _ in 0..60 {
            let buf = pose_buffer(&[
                (PL_LEFT_SHOULDER, x - 0.05, 0.35, 0.9),
                (PL_RIGHT_SHOULDER, x + 0.05, 0.35, 0.9),
                (PL_LEFT_HIP, x - 0.05, 0.65, 0.9),
                (PL_RIGHT_HIP, x + 0.05, 0.65, 0.9),
            ]);
            t.update_pose(&buf, DT);
            x += 0.004;
        }
        let pose = t.pose();
        assert!(
            (pose.center[0] - (x - 0.004)).abs() < 0.03,
            "{:?}",
            pose.center
        );
        assert!((pose.center[1] - 0.5).abs() < 0.02);
        // 0.004 per frame at 30 Hz.
        assert!(
            (pose.velocity[0] - 0.12).abs() < 0.03,
            "velocity {:?}",
            pose.velocity
        );
        assert!((pose.span - 0.1).abs() < 0.005);
    }

    #[test]
    fn pose_absence_releases_only_after_the_hysteresis_window() {
        let cfg = GestureConfig::default();
        let mut t = GestureTracker::new(cfg);
        let buf = pose_buffer(&[
            (PL_LEFT_SHOULDER, 0.45, 0.35, 0.9),
            (PL_RIGHT_SHOULDER, 0.55, 0.35, 0.9),
        ]);
        for _ in 0..5 {
            t.update_pose(&buf, DT);
        }
        assert!(t.pose().present);
        for _ in 1..cfg.release_frames {
            t.update_pose(&[], DT);
            assert!(t.pose().present, "pose released on a single dropped frame");
        }
        t.update_pose(&[], DT);
        assert!(!t.pose().present);
    }

    #[test]
    fn the_torso_centre_is_live_on_the_first_pose_frame() {
        // The centroid gate reads filtered visibility, so a filter that eases
        // in from zero leaves the centre at its [0.5, 0.5] default for the
        // first few frames — and 0.5 is not where this torso is.
        let mut t = tracker();
        t.update_pose(
            &pose_buffer(&[
                (PL_LEFT_SHOULDER, 0.25, 0.30, 1.0),
                (PL_RIGHT_SHOULDER, 0.35, 0.30, 1.0),
                (PL_LEFT_HIP, 0.25, 0.70, 1.0),
                (PL_RIGHT_HIP, 0.35, 0.70, 1.0),
            ]),
            DT,
        );
        let c = t.pose().center;
        assert!(
            (c[0] - 0.3).abs() < 1e-4 && (c[1] - 0.5).abs() < 1e-4,
            "first-frame centre is still the default: {c:?}"
        );
        assert_eq!(t.pose().velocity, [0.0, 0.0]);
    }

    #[test]
    fn a_released_pose_reappears_without_a_velocity_spike() {
        let mut t = tracker();
        let torso = |x: f32| {
            pose_buffer(&[
                (PL_LEFT_SHOULDER, x - 0.05, 0.35, 0.95),
                (PL_RIGHT_SHOULDER, x + 0.05, 0.35, 0.95),
                (PL_LEFT_HIP, x - 0.05, 0.70, 0.95),
                (PL_RIGHT_HIP, x + 0.05, 0.70, 0.95),
            ])
        };
        for _ in 0..40 {
            t.update_pose(&torso(0.3), DT);
        }
        for _ in 0..60 {
            t.update_pose(&[], DT);
        }
        assert!(!t.pose().present);

        // The body is standing still somewhere else when tracking resumes.
        // Carrying `prev_center` across the gap would differentiate the jump.
        t.update_pose(&torso(0.7), DT);
        let c = t.pose().center;
        assert!(
            (c[0] - 0.7).abs() < 1e-3,
            "re-acquired centre slid in from the old position: {c:?}"
        );
        assert_eq!(
            t.pose().velocity,
            [0.0, 0.0],
            "a stationary body reported motion on re-acquisition"
        );
        for _ in 0..10 {
            t.update_pose(&torso(0.7), DT);
            assert!(
                t.pose().velocity[0].abs() < 0.02,
                "velocity spike after re-acquisition: {:?}",
                t.pose().velocity
            );
        }
    }

    #[test]
    fn garbage_input_never_panics_or_poisons_the_state() {
        let mut t = tracker();
        // Establish a good hand first, so there is state to poison.
        feed(&mut t, &Hand::at(0.4, 0.6).gesture(synth::OPEN_PALM), 8, DT);
        let good_palm = t.hands()[0].palm;

        let mut buf = synth::buffer();
        synth::write(&mut buf, 0, &Hand::at(0.4, 0.6));
        for v in buf.iter_mut().skip(4).take(10) {
            *v = f32::NAN;
        }
        t.update_hands(&buf, DT);
        assert_eq!(
            t.hands()[0].palm,
            good_palm,
            "NaN landmarks entered the filter"
        );

        // Present, but every landmark collapsed to the origin.
        let mut zeroed = synth::buffer();
        zeroed[0] = 1.0;
        zeroed[3] = 0.99;
        t.update_hands(&zeroed, DT);
        assert!(t.hands()[0].palm.iter().all(|v| v.is_finite()));

        // Short buffers, absurd timesteps, infinities.
        t.update_hands(&[], f32::NAN);
        t.update_hands(&[1.0, 0.0], 1e9);
        t.update_hands(&buf, -1.0);
        t.update_two_hand(f32::NAN);
        t.update_pose(&[], 0.0);
        t.update_pose(&[1.0], DT);
        let mut pose = pose_buffer(&[(PL_LEFT_SHOULDER, f32::NAN, f32::INFINITY, 0.9)]);
        pose[1 + PL_RIGHT_HIP * 4 + 3] = f32::NAN;
        t.update_pose(&pose, 1e-9);
        t.update_two_hand(1e9);

        for hand in t.hands() {
            assert!(hand.palm.iter().all(|v| v.is_finite()));
            assert!(hand.velocity.iter().all(|v| v.is_finite()));
            assert!(hand.pinch.is_finite() && hand.openness.is_finite());
            assert!(hand.scale.is_finite() && hand.scale > 0.0);
            assert!(hand.spell_age.is_finite());
        }
        assert!(t.pose().center.iter().all(|v| v.is_finite()));
        assert!(t.pose().span.is_finite());
        assert!(t.two_hand().angular_velocity.is_finite());
        assert!(t.two_hand().distance.is_finite());
    }

    #[test]
    fn reset_forgets_everything() {
        let mut t = tracker();
        feed(
            &mut t,
            &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM),
            10,
            DT,
        );
        assert_eq!(t.hands()[0].spell, Spell::Repel);
        t.reset();
        assert!(!t.hands()[0].present);
        assert_eq!(t.hands()[0].spell, Spell::Idle);
        assert!(!t.pose().present);
        assert!(!t.two_hand().both_present);
    }

    #[test]
    fn wrap_pi_folds_into_the_principal_branch() {
        use core::f32::consts::{PI, TAU};
        assert!((wrap_pi(0.25) - 0.25).abs() < 1e-6);
        assert!((wrap_pi(0.25 + TAU) - 0.25).abs() < 1e-5);
        assert!((wrap_pi(-0.25 - TAU) + 0.25).abs() < 1e-5);
        // The seam itself: a step from just under +pi to just over must read as
        // a small positive step, not as -2*pi.
        let step = wrap_pi((-PI + 0.05) - (PI - 0.05));
        assert!((step - 0.1).abs() < 1e-4, "seam step {step}");
        assert_eq!(wrap_pi(f32::NAN), 0.0);
        for k in -20..20 {
            assert!(wrap_pi(k as f32 * 1.7).abs() <= PI + 1e-5);
        }
    }
}
