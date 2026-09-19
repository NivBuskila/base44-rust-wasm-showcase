//! The finger gun: index extended, the other three fingers curled, thumb up as
//! the hammer. Dropping the thumb pulls the trigger and fires a shot from the
//! fingertip along the line the finger points.
//!
//! This is a *pose plus a trigger*, unlike a [`crate::bolt`] throw (pure motion)
//! or a [`crate::combo`] (a sequence of latched spells). The pose is read
//! straight from the landmarks rather than from MediaPipe's canned labels,
//! which have no finger-gun class — the closest, `Pointing_Up`, is also what the
//! hand becomes once the thumb drops, so it could only ever see the shot, never
//! the aim. Every distance is in hand scales so the gun works at any range.

use crate::config::{
    HANDS, LM_INDEX_MCP, LM_INDEX_TIP, LM_MIDDLE_TIP, LM_PINKY_TIP, LM_RING_TIP, LM_THUMB_TIP,
    LM_WRIST,
};
use crate::gesture::{GestureTracker, HandState};

/// Index fingertip at least this far from the wrist reads as extended, and the
/// other three tips must all stay under `CURLED`. The absolute numbers are
/// deliberately loose — a hand aimed *at* the camera foreshortens, so a real
/// gun pose measures far shorter than the 2.3 hand scales a flat straight
/// finger does. What actually separates the pose is `LONGER`: the index must
/// out-reach the longest curled finger by that ratio, which holds at any angle.
const EXTENDED: f32 = 1.45;
const CURLED: f32 = 1.6;
const LONGER: f32 = 1.3;

/// A finger aimed straight *at the lens* foreshortens until the tip sits almost
/// on top of the wrist, so it can never clear `EXTENDED`. Anything above this
/// floor that still out-reaches the curled fingers by `LONGER` is a gun — and a
/// reach below `EXTENDED` is that gun pointing at the viewer, which fires with
/// no screen direction (`dir == [0, 0]`) exactly like a bolt pushed at the lens.
const AT_LENS: f32 = 0.72;

/// A foreshortened finger is a much weaker signal than an extended one — a
/// half-curled hand measures the same — so aiming at the viewer additionally
/// demands this much clearance over the curled fingers, well past `LONGER`.
const AT_LENS_LONGER: f32 = 1.4;

/// How much closer to the lens the fingertip must sit than the wrist, in hand
/// scales, to read as aimed at the viewer. This is the *reliable* test: the
/// screen-reach floors above can only ever infer depth from foreshortening,
/// which a half-curled hand mimics exactly, whereas MediaPipe's per-landmark
/// `z` states it outright. A finger pointed at the camera clears this even when
/// its on-screen reach still looks like an ordinary extended finger, which is
/// why aiming at the lens used to be nearly impossible to hit.
const AT_LENS_DEPTH: f32 = 0.5;

/// The hammer is the thumb tip's distance from the index knuckle, in hand
/// scales — but it is read *relatively*, against the widest spread this hand has
/// shown while holding the pose, not against absolute numbers. Absolute bars
/// could never work at every orientation at once: a hand turned towards the lens
/// projects its raised thumb almost on top of the knuckle, so a bar that armed a
/// side-on gun never armed a gun aimed at the camera, and a bar low enough for
/// the latter fired on a side-on hand's resting thumb.
///
/// `ARMED` is how close to that peak counts as the hammer being up, `DROP` how
/// far below it counts as the trigger being pulled; the gap between them is the
/// hysteresis that stops a wobbling thumb from double-tapping. `MIN_PEAK` is the
/// smallest spread still believable as a raised thumb, so a hand that never
/// raised one cannot fire from noise alone.
const ARMED: f32 = 0.88;
const DROP: f32 = 0.72;
const MIN_PEAK: f32 = 0.34;

/// The peak decays this much per second while the pose is held, so a spread
/// measured at one orientation cannot keep the gun un-fireable after the hand
/// turns and projects smaller.
const PEAK_DECAY: f32 = 0.35;

/// Shortest knuckle-to-tip segment, in hand scales, still trusted for aim.
const MIN_AIM: f32 = 0.35;

/// Below this size the landmarks are too coarse to tell fingers apart.
const MIN_SCALE: f32 = 0.02;

/// One shot fired this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    /// The index fingertip, normalised `[0, 1]`.
    pub from: [f32; 2],
    /// Unit direction the finger points.
    pub dir: [f32; 2],
}

/// Per-hand trigger state.
#[derive(Clone, Debug, Default)]
pub struct GunTracker {
    /// True once the gun pose has been seen with the thumb up; the thumb then
    /// dropping is the shot. Cleared by the shot, re-set by raising the thumb.
    cocked: [bool; HANDS],
    /// Widest thumb spread seen while this hand held the pose, in hand scales.
    peak: [f32; HANDS],
}

impl GunTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// True while the hand holds the finger-gun pose, thumb up or down.
    pub fn aiming(&self, tracker: &GestureTracker, slot: usize) -> bool {
        tracker
            .hands()
            .get(slot)
            .is_some_and(|hand| hand.present && pose(hand).is_some())
    }

    /// The aim ray of a hand holding the gun pose, for the sight the renderer
    /// draws. `dir == [0, 0]` means the gun points at the lens, which has no
    /// screen bearing to draw along.
    pub fn aim(&self, tracker: &GestureTracker, slot: usize) -> Option<Shot> {
        let hand = tracker.hands().get(slot)?;
        if !hand.present {
            return None;
        }
        let p = pose(hand)?;
        Some(Shot {
            from: p.tip,
            dir: p.dir,
        })
    }

    /// Returns whichever hands fired this step.
    pub fn update(&mut self, tracker: &GestureTracker, dt: f32) -> [Option<Shot>; HANDS] {
        let mut fired: [Option<Shot>; HANDS] = [None; HANDS];
        let decay = 1.0 - (PEAK_DECAY * dt.max(0.0)).min(0.5);
        for (slot, hand) in tracker.hands().iter().enumerate().take(HANDS) {
            let Some(p) = hand.present.then(|| pose(hand)).flatten() else {
                self.cocked[slot] = false;
                self.peak[slot] = 0.0;
                continue;
            };
            let peak = (self.peak[slot] * decay).max(p.thumb);
            self.peak[slot] = peak;
            if peak < MIN_PEAK {
                continue;
            }
            if p.thumb >= peak * ARMED {
                self.cocked[slot] = true;
            } else if p.thumb <= peak * DROP && self.cocked[slot] {
                self.cocked[slot] = false;
                // The pull only re-arms from a fresh raise, so the peak restarts
                // from where the thumb is now.
                self.peak[slot] = p.thumb;
                fired[slot] = Some(Shot {
                    from: p.tip,
                    dir: p.dir,
                });
            }
        }
        fired
    }
}

/// The gun pose as read from one hand, or `None` if the hand is not making it.
struct Pose {
    tip: [f32; 2],
    dir: [f32; 2],
    /// Thumb tip to index knuckle, in hand scales.
    thumb: f32,
}

fn pose(hand: &HandState) -> Option<Pose> {
    let scale = hand.scale;
    if !scale.is_finite() || scale < MIN_SCALE {
        return None;
    }
    let lm = |i: usize| [hand.landmarks[i][0], hand.landmarks[i][1]];
    let wrist = lm(LM_WRIST);
    let reach = |i: usize| dist(wrist, lm(i)) / scale;

    let index = reach(LM_INDEX_TIP);
    let longest_curled = [LM_MIDDLE_TIP, LM_RING_TIP, LM_PINKY_TIP]
        .iter()
        .map(|&i| reach(i))
        .fold(0.0f32, f32::max);
    let extended = index >= AT_LENS && index >= longest_curled * LONGER;
    let curled = longest_curled <= CURLED;
    if !extended || !curled {
        return None;
    }
    // Fingertip depth relative to the wrist, in hand scales; positive means the
    // tip is nearer the lens.
    let depth = (hand.landmarks[LM_WRIST][2] - hand.landmarks[LM_INDEX_TIP][2]) / scale;
    let at_lens_by_depth = depth.is_finite() && depth >= AT_LENS_DEPTH;
    if at_lens_by_depth || index < EXTENDED {
        // Pointed at the viewer: there is no screen bearing to shoot along.
        // Foreshortening alone is a weak signal, so without the depth reading
        // the pose still has to out-reach the curled fingers by a wide margin.
        if !at_lens_by_depth && index < longest_curled * AT_LENS_LONGER {
            return None;
        }
        let thumb = dist(lm(LM_THUMB_TIP), lm(LM_INDEX_MCP)) / scale;
        if !thumb.is_finite() {
            return None;
        }
        return Some(Pose {
            tip: lm(LM_INDEX_TIP),
            dir: [0.0, 0.0],
            thumb,
        });
    }

    let knuckle = lm(LM_INDEX_MCP);
    let tip = lm(LM_INDEX_TIP);
    // Aim along the finger, but the knuckle-to-tip segment collapses when the
    // finger is aimed towards the lens, and a near-zero segment takes its
    // direction from landmark noise — which is how a shot came out backwards.
    // The wrist is far enough from the tip that its bearing can never flip, so
    // it serves both as the fallback for a collapsed finger and as the sanity
    // check: a finger bearing that disagrees with it by more than a right angle
    // is noise, not aim.
    let (wx, wy) = (tip[0] - wrist[0], tip[1] - wrist[1]);
    let (mut dx, mut dy) = (tip[0] - knuckle[0], tip[1] - knuckle[1]);
    if (dx * dx + dy * dy).sqrt() < MIN_AIM * scale || dx * wx + dy * wy <= 0.0 {
        dx = wx;
        dy = wy;
    }
    let len = (dx * dx + dy * dy).sqrt();
    if !len.is_finite() || len <= 0.0 {
        return None;
    }
    let thumb = dist(lm(LM_THUMB_TIP), knuckle) / scale;
    if !thumb.is_finite() {
        return None;
    }
    Some(Pose {
        tip,
        dir: [dx / len, dy / len],
        thumb,
    })
}

#[inline]
fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gesture::{synth, GestureConfig};

    const DT: f32 = 1.0 / 60.0;

    struct Rig {
        gestures: GestureTracker,
        gun: GunTracker,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                gestures: GestureTracker::new(GestureConfig::default()),
                gun: GunTracker::new(),
            }
        }

        fn frames(&mut self, gesture: i32, n: usize) -> Option<Shot> {
            let mut fired = None;
            for _ in 0..n {
                let mut buf = synth::buffer();
                synth::write(&mut buf, 0, &synth::Hand::at(0.5, 0.5).gesture(gesture));
                self.gestures.update_hands(&buf, DT);
                fired = fired.or(self.gun.update(&self.gestures, DT)[0]);
            }
            fired
        }
    }

    #[test]
    fn dropping_the_thumb_on_a_finger_gun_fires_along_the_finger() {
        let mut rig = Rig::new();
        assert!(rig.frames(synth::FINGER_GUN, 20).is_none(), "aiming fired");
        assert!(rig.gun.aiming(&rig.gestures, 0));
        let shot = rig
            .frames(synth::POINTING_UP, 20)
            .expect("the trigger pull did not fire");
        // The synthetic index finger points straight up the screen.
        assert!(
            shot.dir[1] < -0.9,
            "shot did not follow the finger: {:?}",
            shot.dir
        );
        assert!(shot.from[1] < 0.5, "shot did not leave from the fingertip");
    }

    #[test]
    fn the_gun_fires_once_per_trigger_pull() {
        let mut rig = Rig::new();
        rig.frames(synth::FINGER_GUN, 20);
        assert!(rig.frames(synth::POINTING_UP, 20).is_some());
        assert!(
            rig.frames(synth::POINTING_UP, 40).is_none(),
            "holding the thumb down kept firing"
        );
        rig.frames(synth::FINGER_GUN, 20);
        assert!(
            rig.frames(synth::POINTING_UP, 20).is_some(),
            "did not re-cock"
        );
    }

    #[test]
    fn pointing_without_ever_cocking_is_not_a_shot() {
        let mut rig = Rig::new();
        assert!(rig.frames(synth::POINTING_UP, 40).is_none());
        // An open hand or a fist is not a gun at all, so closing them cannot
        // fire either.
        rig.frames(synth::OPEN_PALM, 20);
        assert!(!rig.gun.aiming(&rig.gestures, 0));
        assert!(rig.frames(synth::CLOSED_FIST, 20).is_none());
    }
}
