use super::*;
use crate::config::{FLUID_H, FLUID_W};
use crate::fluid::Fluid;
use crate::gesture::{synth, GestureConfig};
use crate::particles::Particles;

const DT: f32 = 1.0 / 60.0;
const SCALE: f32 = 0.1;

struct Rig {
    gestures: GestureTracker,
    throws: ThrowTracker,
}

impl Rig {
    fn new() -> Self {
        Self {
            gestures: GestureTracker::new(GestureConfig::default()),
            throws: ThrowTracker::new(),
        }
    }

    /// One hand at `x`, stepped through the tracker and the recogniser.
    fn frame(&mut self, gesture: i32, x: f32) -> [Option<Throw>; HANDS] {
        self.frame_scaled(gesture, x, SCALE)
    }

    fn frame_scaled(&mut self, gesture: i32, x: f32, scale: f32) -> [Option<Throw>; HANDS] {
        let mut buf = synth::buffer();
        let hand = synth::Hand::at(x, 0.5).gesture(gesture).scaled(scale);
        synth::write(&mut buf, 0, &hand);
        self.gestures.update_hands(&buf, DT);
        self.gestures.update_two_hand(DT);
        self.throws.update(&self.gestures, DT)
    }

    /// The hand grows in place from `SCALE` to `to` over `frames`: a push
    /// straight at the lens.
    fn push(&mut self, gesture: i32, to: f32, frames: usize) -> Option<Throw> {
        let mut thrown = None;
        for i in 0..frames {
            let s = SCALE + (to - SCALE) * (i as f32 + 1.0) / frames as f32;
            thrown = thrown.or(self.frame_scaled(gesture, 0.5, s)[0]);
        }
        thrown
    }

    /// Holds the hand still, so the gesture latches and the speed decays.
    fn settle(&mut self, gesture: i32, x: f32) {
        for _ in 0..40 {
            self.frame(gesture, x);
        }
    }

    /// One fast sweep from `from` to `to`, returning any throw.
    fn flick(&mut self, gesture: i32, from: f32, to: f32, frames: usize) -> Option<Throw> {
        let mut thrown = None;
        for i in 0..frames {
            let x = from + (to - from) * (i as f32 + 1.0) / frames as f32;
            thrown = thrown.or(self.frame(gesture, x)[0]);
        }
        thrown
    }
}

#[test]
fn a_flick_of_an_open_hand_throws_along_the_motion() {
    let mut rig = Rig::new();
    rig.settle(synth::OPEN_PALM, 0.25);
    let throw = rig
        .flick(synth::OPEN_PALM, 0.25, 0.75, 8)
        .expect("a fast open-hand flick did not throw");
    assert!(
        throw.dir[0] > 0.8,
        "bolt did not fly along the flick: {:?}",
        throw.dir
    );
    assert!(throw.power > 0.0);
}

#[test]
fn a_slow_hand_and_a_fist_never_throw() {
    let mut rig = Rig::new();
    rig.settle(synth::OPEN_PALM, 0.4);
    assert!(
        rig.flick(synth::OPEN_PALM, 0.4, 0.46, 40).is_none(),
        "a slow drift threw a bolt"
    );
    let mut fist = Rig::new();
    fist.settle(synth::CLOSED_FIST, 0.25);
    assert!(
        fist.flick(synth::CLOSED_FIST, 0.25, 0.75, 8).is_none(),
        "a fist threw a bolt"
    );
}

#[test]
fn one_flick_throws_once_until_the_hand_slows_down() {
    let mut rig = Rig::new();
    rig.settle(synth::OPEN_PALM, 0.2);
    assert!(rig.flick(synth::OPEN_PALM, 0.1, 0.5, 6).is_some());
    // Still sweeping the same way: one flick is one bolt. (Reversing *is* a
    // new throw — the hand has to stop to turn around.)
    assert!(
        rig.flick(synth::OPEN_PALM, 0.5, 0.9, 6).is_none(),
        "a continuous sweep machine-gunned bolts"
    );
    rig.settle(synth::OPEN_PALM, 0.2);
    assert!(
        rig.flick(synth::OPEN_PALM, 0.2, 0.8, 8).is_some(),
        "the throw never re-armed"
    );
}

#[test]
fn a_push_at_the_camera_throws_at_the_viewer() {
    let mut rig = Rig::new();
    rig.settle(synth::OPEN_PALM, 0.5);
    let throw = rig
        .push(synth::OPEN_PALM, SCALE * 1.8, 10)
        .expect("a fast push at the camera did not throw");
    assert_eq!(throw.dir, [0.0, 0.0], "a push at the lens flew sideways");
    assert!(throw.power > 0.0);
    // A slow approach is just moving closer, not a throw.
    let mut slow = Rig::new();
    slow.settle(synth::OPEN_PALM, 0.5);
    assert!(
        slow.push(synth::OPEN_PALM, SCALE * 1.3, 90).is_none(),
        "leaning in threw a bolt"
    );
}

#[test]
fn a_bolt_at_the_viewer_detonates_at_the_palm() {
    let mut fluid = Fluid::new(FLUID_W, FLUID_H);
    let mut particles = Particles::new(4096, 7);
    particles.set_active(4096);
    particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
    fire(
        &mut fluid,
        &mut particles,
        2.0,
        Throw {
            from: [0.5, 0.5],
            dir: [0.0, 0.0],
            power: 1.0,
        },
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    assert!(fluid.max_speed() > 0.0, "the bolt injected no velocity");
}

#[test]
fn a_thrown_bolt_lights_the_field_and_kicks_it() {
    let mut fluid = Fluid::new(FLUID_W, FLUID_H);
    let mut particles = Particles::new(4096, 7);
    particles.set_active(4096);
    particles.seed_uniform(FLUID_W, FLUID_H, 2.0);
    fire(
        &mut fluid,
        &mut particles,
        2.0,
        Throw {
            from: [0.3, 0.5],
            dir: [1.0, 0.0],
            power: 1.0,
        },
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    assert!(fluid.max_speed() > 0.0, "the bolt injected no velocity");
    assert!(
        fluid.dye()[2].data.iter().any(|&v| v > 0.01),
        "the bolt left no dye"
    );
}

#[test]
fn garbage_throw_data_is_harmless() {
    let mut fluid = Fluid::new(FLUID_W, FLUID_H);
    let mut particles = Particles::new(256, 3);
    particles.set_active(256);
    fire(
        &mut fluid,
        &mut particles,
        2.0,
        Throw {
            from: [f32::NAN, 0.5],
            dir: [f32::NAN, f32::NAN],
            power: f32::NAN,
        },
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    assert!(fluid.max_speed().is_finite());
}
