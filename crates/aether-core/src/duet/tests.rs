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
