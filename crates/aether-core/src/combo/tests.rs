use super::*;
use crate::gesture::{synth, GestureConfig};

const DT: f32 = 1.0 / 60.0;

/// A live pair of trackers, driven the way the engine drives them: the
/// gesture tracker keeps its history across gesture changes, which is what
/// makes the latching delay part of the test rather than something the test
/// steps around.
struct Caster {
    gestures: GestureTracker,
    combos: ComboTracker,
}

impl Caster {
    fn new() -> Self {
        Self {
            gestures: GestureTracker::new(GestureConfig::default()),
            combos: ComboTracker::new(),
        }
    }

    /// Performs `gesture` for `seconds`, returning the first combo that
    /// fired during it.
    fn perform(&mut self, gesture: i32, seconds: f32) -> Option<ComboHit> {
        let mut buf = synth::buffer();
        let hand = synth::Hand::at(0.5, 0.5).gesture(gesture);
        synth::write(&mut buf, 0, &hand);
        let mut hit: Option<ComboHit> = None;
        for _ in 0..(seconds / DT).ceil() as usize {
            self.gestures.update_hands(&buf, DT);
            self.gestures.update_two_hand(DT);
            for h in self.combos.update(&self.gestures, DT).into_iter().flatten() {
                hit = hit.or(Some(h));
            }
        }
        hit
    }
}

#[test]
fn a_full_sequence_fires_once_and_needs_recasting() {
    let mut c = Caster::new();
    assert!(c.perform(synth::CLOSED_FIST, 0.6).is_none());
    let hit = c.perform(synth::OPEN_PALM, 0.6).expect("nova did not fire");
    assert_eq!(hit.effect, ComboEffect::Nova);
    // Staying on the last gesture must not machine-gun the effect.
    assert!(
        c.perform(synth::OPEN_PALM, 1.0).is_none(),
        "combo fired again without a fresh cast"
    );
}

#[test]
fn the_charged_sequence_needs_its_hold() {
    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.5);
    c.perform(synth::VICTORY, 0.6);
    // The final fist has to be *held*; a twitch is not a cast. It still has
    // to latch first, so this window is short but not zero.
    assert!(c.perform(synth::CLOSED_FIST, 0.2).is_none());
    let hit = c
        .perform(synth::CLOSED_FIST, 1.2)
        .expect("supernova did not fire");
    assert_eq!(hit.effect, ComboEffect::Supernova);
}

#[test]
fn a_brief_wrong_gesture_does_not_break_a_sequence() {
    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.6);
    // A few frames of something else: a hand turning edge-on, not intent.
    c.perform(synth::THUMB_UP, 0.08);
    assert!(
        c.perform(synth::OPEN_PALM, 0.6).is_some(),
        "noise cancelled a valid sequence"
    );
}

#[test]
fn a_held_wrong_gesture_cancels_and_so_does_a_long_pause() {
    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.6);
    c.perform(synth::THUMB_UP, 1.0);
    assert!(
        c.perform(synth::OPEN_PALM, 0.6).is_none(),
        "a held wrong spell did not cancel"
    );

    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.6);
    c.perform(synth::NONE, STEP_WINDOW + 0.5);
    assert!(
        c.perform(synth::OPEN_PALM, 0.6).is_none(),
        "the timeout did not cancel"
    );
}

#[test]
fn progress_reports_the_furthest_candidate_and_the_last_hit() {
    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.6);
    let p = c.combos.progress();
    assert_ne!(p.combo, usize::MAX, "nothing reported as in progress");
    assert_eq!(p.matched, 1);

    c.perform(synth::OPEN_PALM, 0.6);
    assert_eq!(
        c.combos.progress().fired,
        0,
        "the fired combo is not being shown"
    );
}

#[test]
fn an_absent_hand_and_a_wild_dt_are_both_survivable() {
    let mut c = Caster::new();
    c.perform(synth::CLOSED_FIST, 0.6);
    let empty = synth::buffer();
    for _ in 0..30 {
        c.gestures.update_hands(&empty, DT);
        c.combos.update(&c.gestures, f32::NAN);
    }
    assert!(
        c.perform(synth::OPEN_PALM, 0.6).is_none(),
        "a sequence survived the hand disappearing"
    );
}
