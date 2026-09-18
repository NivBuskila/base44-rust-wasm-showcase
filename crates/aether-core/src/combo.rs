//! Gesture *sequences*: spells cast by performing several gestures in order.
//!
//! A single gesture is a classifier result. A sequence is a state machine over
//! time, and that is the point of this module: recognising "fist, then open
//! palm" requires knowing when a gesture began, surviving the frame or two
//! where MediaPipe drops or mislabels a hand, and knowing when to give up.
//!
//! ## Design
//!
//! A combo is **data**, not code: [`ComboDef`] is a name, an ordered list of
//! [`ComboStep`]s and an effect tag. Adding a combo is adding a row to
//! [`COMBOS`], which is what lets the HUD's spellbook and the recogniser stay in
//! sync without either knowing about the other.
//!
//! ## Why it tolerates noise
//!
//! The recogniser advances on the *latched* spell from [`crate::gesture`], which
//! is already temporally committed, and it only resets when
//!
//! - nothing happened for [`STEP_WINDOW`] seconds (the caster gave up), or
//! - a committed spell that is neither the current step nor the next one has
//!   persisted for [`WRONG_GRACE`] seconds (the caster did something else).
//!
//! A wrong spell for a handful of frames — the tell of a hand turning edge-on —
//! is deliberately not enough to break a sequence. The opposite bias is the
//! failure mode that matters: a combo that shatters on noise reads as broken
//! recognition even when every individual gesture was classified correctly.
//!
//! Effects are never applied here. [`crate::spells`] owns every write to the
//! fluid and the particles; this module only decides *that* something fired.

use crate::config::HANDS;
use crate::gesture::{GestureTracker, Spell};

/// Seconds without progress before a partial sequence is abandoned.
pub const STEP_WINDOW: f32 = 2.2;
/// Seconds a wrong committed spell must persist before it cancels a sequence.
const WRONG_GRACE: f32 = 0.35;
/// Seconds the HUD keeps celebrating a completed combo.
pub const SHOW_TIME: f32 = 1.1;
/// Upper bound on [`COMBOS`], so the per-hand lane state stays a fixed array
/// instead of a heap allocation on a path that runs every frame.
const MAX_COMBOS: usize = 4;

/// One element of a sequence: a spell, and how long it must be held.
///
/// `hold` of 0 means "as soon as it commits"; a positive hold on a final step is
/// what makes a charged combo feel like winding up rather than a twitch.
#[derive(Clone, Copy, Debug)]
pub struct ComboStep {
    pub spell: Spell,
    pub hold: f32,
}

impl ComboStep {
    const fn new(spell: Spell, hold: f32) -> Self {
        Self { spell, hold }
    }
}

/// What a completed sequence does, resolved by [`crate::spells`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComboEffect {
    /// A bright expanding ring: the easy, always-lands combo.
    Nova,
    /// A wide swirling storm around the hand.
    Tempest,
    /// The charged one: a huge shell, a hard outward kick and a heat flash.
    Supernova,
}

/// A castable sequence.
#[derive(Clone, Copy, Debug)]
pub struct ComboDef {
    /// Stable identifier, also the HUD label.
    pub name: &'static str,
    pub steps: &'static [ComboStep],
    pub effect: ComboEffect,
}

/// The spellbook. Three entries on purpose: two steps are learnable by watching
/// someone else do it, four are not, and a caster who cannot remember the
/// sequence never reaches the part where the recognition is impressive.
pub static COMBOS: &[ComboDef] = &[
    ComboDef {
        name: "nova",
        steps: &[
            ComboStep::new(Spell::Attract, 0.25),
            ComboStep::new(Spell::Repel, 0.15),
        ],
        effect: ComboEffect::Nova,
    },
    ComboDef {
        name: "tempest",
        steps: &[
            ComboStep::new(Spell::Repel, 0.25),
            ComboStep::new(Spell::Vortex, 0.35),
        ],
        effect: ComboEffect::Tempest,
    },
    ComboDef {
        name: "supernova",
        steps: &[
            ComboStep::new(Spell::Attract, 0.2),
            ComboStep::new(Spell::Freeze, 0.3),
            ComboStep::new(Spell::Attract, 0.8),
        ],
        effect: ComboEffect::Supernova,
    },
];

/// A sequence that completed this step.
#[derive(Clone, Copy, Debug)]
pub struct ComboHit {
    /// Index into [`COMBOS`].
    pub combo: usize,
    pub effect: ComboEffect,
    /// Hand slot that cast it.
    pub slot: usize,
}

/// What the HUD draws: how far the best candidate has got.
#[derive(Clone, Copy, Debug)]
pub struct ComboProgress {
    /// Index into [`COMBOS`], or `usize::MAX` when nothing is in progress.
    pub combo: usize,
    /// Steps already matched.
    pub matched: usize,
    /// 0..1 through the step being held right now.
    pub charge: f32,
    /// Combo that fired within the last [`SHOW_TIME`] seconds, or `usize::MAX`.
    pub fired: usize,
}

impl Default for ComboProgress {
    fn default() -> Self {
        Self {
            combo: usize::MAX,
            matched: 0,
            charge: 0.0,
            fired: usize::MAX,
        }
    }
}

/// Progress of one hand through one combo.
///
/// Every combo advances in parallel rather than the recogniser committing to
/// one: `nova` and `supernova` share their first step, and picking a lane on
/// step one would make the longer combo impossible to cast.
#[derive(Clone, Copy, Debug, Default)]
struct Lane {
    matched: usize,
    /// Seconds the current step's spell has been held since it appeared.
    held: f32,
    /// Seconds since the last successful advance.
    idle: f32,
    /// Seconds a wrong committed spell has persisted.
    wrong: f32,
}

impl Lane {
    fn active(&self) -> bool {
        self.matched > 0 || self.held > 0.0
    }
}

/// Cross-frame state of the sequence recogniser.
#[derive(Clone, Debug)]
pub struct ComboTracker {
    lanes: [[Lane; MAX_COMBOS]; HANDS],
    show: f32,
    shown: usize,
}

impl Default for ComboTracker {
    fn default() -> Self {
        Self {
            lanes: [[Lane::default(); MAX_COMBOS]; HANDS],
            show: 0.0,
            shown: usize::MAX,
        }
    }
}

impl ComboTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advances every lane by `dt` and returns whatever completed.
    ///
    /// At most one hit per call: two combos completing on the same frame would
    /// stack into a single indistinguishable flash, so the first match wins and
    /// every other lane is rewound.
    pub fn update(&mut self, tracker: &GestureTracker, dt: f32) -> Option<ComboHit> {
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        if self.show > 0.0 {
            self.show -= dt;
            if self.show <= 0.0 {
                self.shown = usize::MAX;
            }
        }

        let mut hit = None;
        for (slot, hand) in tracker.hands().iter().enumerate().take(HANDS) {
            if !hand.present {
                self.lanes[slot] = [Lane::default(); MAX_COMBOS];
                continue;
            }
            let spell = hand.spell;
            for (ci, def) in COMBOS.iter().enumerate().take(MAX_COMBOS) {
                let lane = &mut self.lanes[slot][ci];
                let step = def.steps[lane.matched.min(def.steps.len() - 1)];

                if spell == step.spell {
                    lane.wrong = 0.0;
                    lane.idle = 0.0;
                    lane.held += dt;
                    if lane.held >= step.hold {
                        lane.matched += 1;
                        lane.held = 0.0;
                        if lane.matched >= def.steps.len() {
                            *lane = Lane::default();
                            if hit.is_none() {
                                hit = Some(ComboHit {
                                    combo: ci,
                                    effect: def.effect,
                                    slot,
                                });
                            }
                        }
                    }
                    continue;
                }

                // Not the spell this step wants. Holding *nothing* is only a
                // timeout; holding another committed spell is a real cancel,
                // but only once it has persisted past the grace window.
                lane.held = 0.0;
                lane.idle += dt;
                // Lingering on the step just completed is not a mistake — the
                // caster's hand cannot teleport into the next shape, and the
                // gesture tracker needs a few frames to commit the new one.
                let lingering = lane.matched > 0
                    && def
                        .steps
                        .get(lane.matched - 1)
                        .is_some_and(|s| s.spell == spell);
                if spell != Spell::Idle && spell != Spell::Release && !lingering {
                    lane.wrong += dt;
                } else {
                    lane.wrong = 0.0;
                }
                if lane.idle > STEP_WINDOW || lane.wrong > WRONG_GRACE {
                    *lane = Lane::default();
                }
            }
        }

        if let Some(h) = hit {
            self.lanes = [[Lane::default(); MAX_COMBOS]; HANDS];
            self.show = SHOW_TIME;
            self.shown = h.combo;
        }
        hit
    }

    /// How deep into a sequence one hand is, as 0..1.
    ///
    /// Non-zero only once a step has actually been matched: the first gesture of
    /// a combo is indistinguishable from casting that gesture on its own, so it
    /// must keep its full effect. Past that point the caster is winding up, and
    /// [`crate::spells`] uses this to fade the intermediate gestures' own
    /// effects out instead of firing them at full strength inside the combo.
    pub fn charging(&self, slot: usize) -> f32 {
        let mut best = 0.0f32;
        for (ci, def) in COMBOS.iter().enumerate().take(MAX_COMBOS) {
            let lane = match self.lanes.get(slot).map(|l| l[ci]) {
                Some(l) if l.matched > 0 => l,
                _ => continue,
            };
            let step = def.steps[lane.matched.min(def.steps.len() - 1)];
            let charge = if step.hold > 0.0 {
                (lane.held / step.hold).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let depth = (lane.matched as f32 - 1.0 + charge) / def.steps.len() as f32;
            best = best.max(depth.clamp(0.0, 1.0));
        }
        best
    }

    /// The furthest-along candidate across both hands, for the HUD.
    pub fn progress(&self) -> ComboProgress {
        let mut best = ComboProgress {
            fired: self.shown,
            ..Default::default()
        };
        let mut best_rank = -1.0f32;
        for slot in 0..HANDS {
            for (ci, def) in COMBOS.iter().enumerate().take(MAX_COMBOS) {
                let lane = self.lanes[slot][ci];
                if !lane.active() {
                    continue;
                }
                let step = def.steps[lane.matched.min(def.steps.len() - 1)];
                let charge = if step.hold > 0.0 {
                    (lane.held / step.hold).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let rank = lane.matched as f32 + charge;
                if rank > best_rank {
                    best_rank = rank;
                    best = ComboProgress {
                        combo: ci,
                        matched: lane.matched,
                        charge,
                        fired: self.shown,
                    };
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
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
            let mut hit = None;
            for _ in 0..(seconds / DT).ceil() as usize {
                self.gestures.update_hands(&buf, DT);
                self.gestures.update_two_hand(DT);
                if let Some(h) = self.combos.update(&self.gestures, DT) {
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
}
