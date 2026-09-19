//! Two-hand *duets*: spells that exist only in the relationship between the
//! hands.
//!
//! A [`crate::combo`] is one hand performing gestures in order. A duet is both
//! hands at once, and what matters is not just which spell each holds but how
//! the pair moves: how far apart they are, whether they are closing, whether
//! both fists opened together. That relationship is already measured by
//! [`crate::gesture::TwoHandState`]; this module turns it into events and
//! ramps the way [`crate::combo`] turns latched spells into hits.
//!
//! ## Design
//!
//! Like combos, a duet is **data**: [`DuetDef`] is a name, the HUD's step
//! tokens and an effect tag, and [`DUETS`] is the spellbook the HUD renders.
//! Unlike combos the recogniser is not a generic state machine — the three
//! duets have genuinely different triggers (a held pair, a collision, a
//! charge-and-release) and a common abstraction would be larger than the three
//! special cases together.
//!
//! Every distance here is in **hand widths** (`TwoHandState::distance` over the
//! mean hand scale), so a duet feels the same near and far from the camera.
//!
//! Effects are never applied here. [`crate::spells`] owns every write to the
//! fluid and the particles; this module only decides *that* and *how hard*.

use crate::config::HANDS;
use crate::gesture::{GestureTracker, Spell};
use crate::math::smoothstep;

/// Seconds the HUD keeps celebrating a completed duet.
pub const SHOW_TIME: f32 = 1.1;

/// Tide: seconds of holding fist + palm to reach full flow, and seconds for
/// the flow to die away once the pair breaks.
const TIDE_WINDUP: f32 = 0.6;
const TIDE_RELEASE: f32 = 0.35;

/// Clap: the pair must be closing faster than this (hand widths per second)
/// and end up closer than `CLAP_DISTANCE`; `CLAP_FULL_SPEED` is a full-power
/// clap. The clap re-arms only once the hands have opened past `CLAP_REARM`,
/// so a hovering pair cannot machine-gun shockwaves.
const CLAP_SPEED: f32 = 2.5;
const CLAP_FULL_SPEED: f32 = 9.0;
const CLAP_DISTANCE: f32 = 1.4;
const CLAP_REARM: f32 = 2.6;

/// Big bang: two fists charge at full rate closer than `BANG_NEAR` hand widths
/// and not at all beyond `BANG_FAR`; a full charge takes `BANG_CHARGE_TIME`
/// seconds of squeezing. The release must be worth at least `BANG_MIN_CHARGE`,
/// and the second fist has `BANG_GRACE` seconds to open after the first.
const BANG_NEAR: f32 = 1.6;
const BANG_FAR: f32 = 4.5;
const BANG_CHARGE_TIME: f32 = 1.4;
const BANG_MIN_CHARGE: f32 = 0.15;
const BANG_GRACE: f32 = 0.4;

/// Floor on the hand scale used to normalise distances; a degenerate scale
/// would turn every distance into "very far".
const MIN_SCALE: f32 = 0.02;

/// What a duet does, resolved by [`crate::spells`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuetEffect {
    /// A continuous jet from the open palm into the fist.
    Tide,
    /// A shockwave from the point where the hands met.
    Clap,
    /// Two squeezed fists let go together: a supernova from between them.
    BigBang,
}

/// A castable duet.
#[derive(Clone, Copy, Debug)]
pub struct DuetDef {
    /// Stable identifier, also the HUD label.
    pub name: &'static str,
    /// HUD instructions. Spell names joined by `+` are both hands at once;
    /// anything else is a motion word the HUD shows verbatim.
    pub steps: &'static [&'static str],
    pub effect: DuetEffect,
}

/// The duet spellbook. Index order is what the HUD and
/// [`DuetProgress`] refer to.
pub static DUETS: &[DuetDef] = &[
    DuetDef {
        name: "tide",
        steps: &["attract+repel"],
        effect: DuetEffect::Tide,
    },
    DuetDef {
        name: "clap",
        steps: &["repel+repel", "clap"],
        effect: DuetEffect::Clap,
    },
    DuetDef {
        name: "big bang",
        steps: &["attract+attract", "squeeze", "release+release"],
        effect: DuetEffect::BigBang,
    },
];

const TIDE: usize = 0;
const CLAP: usize = 1;
const BIG_BANG: usize = 2;

/// A duet event that fired this step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DuetFire {
    pub effect: DuetEffect,
    /// 0..1: how hard the clap was, how long the fists were squeezed.
    pub power: f32,
}

/// What the recogniser hands the spell layer each step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DuetFrame {
    /// Tide strength 0..1; zero when the pair is not fist + palm.
    pub tide: f32,
    pub fire: Option<DuetFire>,
}

/// What the HUD draws.
#[derive(Clone, Copy, Debug)]
pub struct DuetProgress {
    /// Index into [`DUETS`] being wound up, or `usize::MAX`.
    pub active: usize,
    /// 0..1 through the wind-up.
    pub charge: f32,
    /// Duet that fired within the last [`SHOW_TIME`] seconds, or `usize::MAX`.
    pub fired: usize,
}

impl Default for DuetProgress {
    fn default() -> Self {
        Self {
            active: usize::MAX,
            charge: 0.0,
            fired: usize::MAX,
        }
    }
}

/// Cross-frame state of the duet recogniser.
#[derive(Clone, Debug)]
pub struct DuetTracker {
    tide: f32,
    clap_armed: bool,
    /// Big bang charge, 0..1.
    charge: f32,
    /// Seconds since the first fist opened while charged.
    pending: f32,
    show: f32,
    shown: usize,
}

impl Default for DuetTracker {
    fn default() -> Self {
        Self {
            tide: 0.0,
            clap_armed: true,
            charge: 0.0,
            pending: 0.0,
            show: 0.0,
            shown: usize::MAX,
        }
    }
}

impl DuetTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Advances by `dt` and returns this step's tide strength and event.
    pub fn update(&mut self, tracker: &GestureTracker, dt: f32) -> DuetFrame {
        let dt = if dt.is_finite() { dt.clamp(0.0, 0.25) } else { 0.0 };
        if self.show > 0.0 {
            self.show -= dt;
            if self.show <= 0.0 {
                self.shown = usize::MAX;
            }
        }

        let two = tracker.two_hand();
        let hands = tracker.hands();
        if !two.both_present || hands.len() < HANDS {
            // Losing a hand is not a release: a fist that vanishes edge-on must
            // not detonate, and a tide with one hand is just a palm.
            self.tide = 0.0;
            self.charge = 0.0;
            self.pending = 0.0;
            self.clap_armed = true;
            return DuetFrame::default();
        }

        let scale = ((hands[0].scale + hands[1].scale) * 0.5).max(MIN_SCALE);
        let dist = fin(two.distance) / scale;
        let closing = -fin(two.distance_velocity) / scale;
        let (a, b) = (hands[0].spell, hands[1].spell);
        let any_fist = a == Spell::Attract || b == Spell::Attract;
        let both_fists = a == Spell::Attract && b == Spell::Attract;

        // Tide: fist + palm, ramped so it feels like a current building rather
        // than a switch.
        let pair = (a == Spell::Attract && b == Spell::Repel)
            || (a == Spell::Repel && b == Spell::Attract);
        self.tide = if pair {
            (self.tide + dt / TIDE_WINDUP).min(1.0)
        } else {
            (self.tide - dt / TIDE_RELEASE).max(0.0)
        };

        let mut fire = None;

        // Clap: open hands meeting fast. A fist is excluded so that squeezing
        // two fists together for the big bang never reads as a clap.
        if dist > CLAP_REARM {
            self.clap_armed = true;
        }
        if self.clap_armed && !any_fist && dist < CLAP_DISTANCE && closing > CLAP_SPEED {
            self.clap_armed = false;
            fire = Some(DuetFire {
                effect: DuetEffect::Clap,
                power: smoothstep(CLAP_SPEED, CLAP_FULL_SPEED, closing),
            });
            self.show = SHOW_TIME;
            self.shown = CLAP;
        }

        // Big bang: charge while both fists squeeze, fire when both let go.
        if both_fists {
            let closeness = smoothstep(BANG_FAR, BANG_NEAR, dist);
            self.charge = (self.charge + dt * closeness / BANG_CHARGE_TIME).min(1.0);
            self.pending = 0.0;
        } else if self.charge > 0.0 {
            if !any_fist {
                if self.charge >= BANG_MIN_CHARGE && fire.is_none() {
                    fire = Some(DuetFire {
                        effect: DuetEffect::BigBang,
                        power: self.charge,
                    });
                    self.show = SHOW_TIME;
                    self.shown = BIG_BANG;
                }
                self.charge = 0.0;
                self.pending = 0.0;
            } else {
                // One fist opened, the other is still closed: wait for it, but
                // not forever, or a single opened fist an age ago still counts.
                self.pending += dt;
                if self.pending > BANG_GRACE {
                    self.charge = 0.0;
                    self.pending = 0.0;
                }
            }
        }

        DuetFrame {
            tide: self.tide,
            fire,
        }
    }

    /// The duet being wound up right now, for the HUD.
    pub fn progress(&self) -> DuetProgress {
        let (active, charge) = if self.charge > 0.0 {
            (BIG_BANG, self.charge)
        } else if self.tide > 0.0 {
            (TIDE, self.tide)
        } else {
            (usize::MAX, 0.0)
        };
        DuetProgress {
            active,
            charge,
            fired: self.shown,
        }
    }
}

#[inline]
fn fin(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gesture::{synth, GestureConfig};

    const DT: f32 = 1.0 / 60.0;
    const SCALE: f32 = 0.1;

    struct Pair {
        gestures: GestureTracker,
        duets: DuetTracker,
    }

    impl Pair {
        fn new() -> Self {
            Self {
                gestures: GestureTracker::new(GestureConfig::default()),
                duets: DuetTracker::new(),
            }
        }

        /// Holds two hands `gap` apart (normalised units) with the given
        /// gestures for `seconds`, returning the last frame and any fire.
        fn hold(
            &mut self,
            left: i32,
            right: i32,
            gap: f32,
            seconds: f32,
        ) -> (DuetFrame, Option<DuetFire>) {
            let mut fired = None;
            let mut last = DuetFrame::default();
            for _ in 0..(seconds / DT).ceil() as usize {
                last = self.frame(left, right, gap);
                fired = fired.or(last.fire);
            }
            (last, fired)
        }

        fn frame(&mut self, left: i32, right: i32, gap: f32) -> DuetFrame {
            let mut buf = synth::buffer();
            let mut l = synth::Hand::at(0.5 - gap * 0.5, 0.5)
                .gesture(left)
                .scaled(SCALE);
            l.right = false;
            let r = synth::Hand::at(0.5 + gap * 0.5, 0.5)
                .gesture(right)
                .scaled(SCALE);
            synth::write(&mut buf, 0, &l);
            synth::write(&mut buf, 1, &r);
            self.gestures.update_hands(&buf, DT);
            self.gestures.update_two_hand(DT);
            self.duets.update(&self.gestures, DT)
        }

        /// One fast closing motion from `from` to `to` over `frames` frames.
        fn swing(&mut self, gesture: i32, from: f32, to: f32, frames: usize) -> Option<DuetFire> {
            let mut fire = None;
            for i in 0..frames {
                let gap = from + (to - from) * (i as f32 + 1.0) / frames as f32;
                fire = fire.or(self.frame(gesture, gesture, gap).fire);
            }
            fire
        }
    }

    #[test]
    fn fist_and_palm_build_a_tide_that_fades_when_broken() {
        let mut p = Pair::new();
        let (frame, _) = p.hold(synth::CLOSED_FIST, synth::OPEN_PALM, 0.4, 1.0);
        assert!(frame.tide > 0.95, "tide did not build: {}", frame.tide);
        assert_eq!(p.duets.progress().active, TIDE);
        let (frame, _) = p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.4, 0.6);
        assert_eq!(frame.tide, 0.0, "tide did not fade");
    }

    #[test]
    fn squeezed_fists_charge_and_detonate_on_release() {
        let mut p = Pair::new();
        // Far apart: no charge worth anything.
        p.hold(synth::CLOSED_FIST, synth::CLOSED_FIST, 0.6, 0.5);
        assert!(p.duets.progress().charge < 0.05, "charged at arm's length");
        // Squeeze.
        p.hold(synth::CLOSED_FIST, synth::CLOSED_FIST, 0.1, 1.5);
        let charge = p.duets.progress().charge;
        assert!(charge > 0.9, "did not charge while squeezed: {charge}");
        let (_, fire) = p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.1, 0.3);
        let fire = fire.expect("big bang did not fire");
        assert_eq!(fire.effect, DuetEffect::BigBang);
        assert!(fire.power > 0.9);
        assert_eq!(p.duets.progress().fired, BIG_BANG);
        // No second detonation from staying open.
        let (_, again) = p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.1, 0.5);
        assert!(again.is_none());
    }

    #[test]
    fn a_vanishing_hand_never_detonates() {
        let mut p = Pair::new();
        p.hold(synth::CLOSED_FIST, synth::CLOSED_FIST, 0.1, 1.5);
        let buf = synth::buffer();
        let mut fire = None;
        for _ in 0..30 {
            p.gestures.update_hands(&buf, DT);
            p.gestures.update_two_hand(DT);
            fire = fire.or(p.duets.update(&p.gestures, DT).fire);
        }
        assert!(fire.is_none(), "big bang fired from a lost hand");
        assert_eq!(p.duets.progress().charge, 0.0);
    }

    #[test]
    fn a_fast_clap_fires_once_and_rearms_only_when_apart() {
        let mut p = Pair::new();
        // Let the palms latch, well apart.
        p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.6, 0.4);
        let fire = p
            .swing(synth::OPEN_PALM, 0.6, 0.1, 9)
            .expect("clap did not fire");
        assert_eq!(fire.effect, DuetEffect::Clap);
        // Hovering together is not a clap.
        let (_, again) = p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.1, 0.5);
        assert!(again.is_none(), "clap re-fired while hovering");
        // Apart again, then another clap lands.
        p.hold(synth::OPEN_PALM, synth::OPEN_PALM, 0.6, 0.4);
        assert!(
            p.swing(synth::OPEN_PALM, 0.6, 0.1, 9).is_some(),
            "clap did not re-arm"
        );
    }

    #[test]
    fn two_fists_meeting_is_a_squeeze_not_a_clap() {
        let mut p = Pair::new();
        p.hold(synth::CLOSED_FIST, synth::CLOSED_FIST, 0.6, 0.4);
        assert!(
            p.swing(synth::CLOSED_FIST, 0.6, 0.1, 9).is_none(),
            "fists clapped"
        );
    }
}
