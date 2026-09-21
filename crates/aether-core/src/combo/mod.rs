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

mod book;
#[cfg(test)]
mod tests;

pub use book::{ComboDef, ComboEffect, ComboStep, COMBOS};

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
    /// At most one hit *per hand*: two combos completing on the same frame for
    /// the same hand would stack into a single indistinguishable flash, so the
    /// first match wins there. Both hands may land together, which is what makes
    /// a two-handed cast possible.
    pub fn update(&mut self, tracker: &GestureTracker, dt: f32) -> [Option<ComboHit>; HANDS] {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        if self.show > 0.0 {
            self.show -= dt;
            if self.show <= 0.0 {
                self.shown = usize::MAX;
            }
        }

        let mut hits: [Option<ComboHit>; HANDS] = [None; HANDS];
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
                            if hits[slot].is_none() {
                                hits[slot] = Some(ComboHit {
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

        for h in hits.iter().flatten() {
            // Only the hand that cast is rewound. Wiping both hands' lanes made a
            // two-handed cast impossible: whichever hand completed first also
            // cancelled the other one mid-sequence.
            self.lanes[h.slot] = [Lane::default(); MAX_COMBOS];
            self.show = SHOW_TIME;
            self.shown = h.combo;
        }
        hits
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
