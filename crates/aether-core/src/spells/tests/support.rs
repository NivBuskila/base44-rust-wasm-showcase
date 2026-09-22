//! Fixtures: a tracker fed synthetic hands, the pose builders and the palm helper.

pub(super) use crate::spells::*;

pub(super) use crate::config::{FLOW_H, FLOW_W, FLUID_H, FLUID_W};
pub(super) use crate::field::Grid;
pub(super) use crate::gesture::synth::{self, Hand};
pub(super) use crate::gesture::{GestureConfig, GestureTracker};
pub(super) use crate::particles::ParticleConfig;
pub(super) struct Rig {
    pub(super) fluid: Fluid,
    pub(super) particles: Particles,
    pub(super) flow: VecField,
    pub(super) body: VecField,
    pub(super) state: SpellState,
    pub(super) params: Params,
}
impl Rig {
    /// The field drives are off by default so the gesture paths can be
    /// measured on their own; the two tests that care switch them on.
    pub(super) fn new() -> Self {
        Self::sized(FLUID_W, FLUID_H)
    }
    /// A rig on a smaller grid, for the tests that need hundreds of solver
    /// steps rather than a realistic resolution.
    pub(super) fn sized(w: usize, h: usize) -> Self {
        let params = Params {
            flow_force: 0.0,
            body_push: 0.0,
            ..Params::default()
        };
        let mut particles = Particles::new(4096, 0xA37E);
        particles.set_active(4096);
        particles.seed_uniform(w, h, params.particle_life);
        Self {
            fluid: Fluid::new(w, h),
            particles,
            flow: VecField::new(FLOW_W, FLOW_H),
            body: VecField::new(w, h),
            state: SpellState::new(),
            params,
        }
    }
    pub(super) fn run(&mut self, tracker: &GestureTracker, dt: f32) -> SpellReport {
        self.run_with(tracker, None, dt)
    }
    /// One step with a completed sequence handed in, as the engine does it.
    pub(super) fn run_with(
        &mut self,
        tracker: &GestureTracker,
        combo: Option<ComboHit>,
        dt: f32,
    ) -> SpellReport {
        apply(
            tracker,
            &self.flow,
            &self.body,
            &mut self.fluid,
            &mut self.particles,
            &mut self.state,
            &self.params,
            {
                let mut per_hand = [None; 2];
                if let Some(h) = combo {
                    per_hand[h.slot.min(1)] = Some(h);
                }
                per_hand
            },
            [0.0; 2],
            DuetFrame::default(),
            dt,
        )
    }
    pub(super) fn velocity_at(&self, x: f32, y: f32) -> (f32, f32) {
        self.fluid.velocity().sample(x, y)
    }
}
pub(super) const IDLE_HAND: Hand = Hand {
    x: 0.4,
    y: 0.5,
    scale: 0.13,
    gesture: synth::NONE,
    // Unconfident, so no label path fires and the geometry is ambiguous.
    score: 0.1,
    pinched: false,
    right: true,
};

/// Render step.
pub(super) const DT: f32 = 1.0 / 60.0;

/// Inference step, which is what the tracker is fed at.
pub(super) const HAND_DT: f32 = 1.0 / 30.0;

pub(super) fn empty_tracker() -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    t.update_hands(&synth::buffer(), HAND_DT);
    t.update_two_hand(HAND_DT);
    t
}

/// A tracker holding one still hand, held long enough to latch and to age
/// past the vortex windup.
pub(super) fn holding(hand: &Hand) -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, hand);
    for _ in 0..60 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
    }
    t
}

/// A still hand held just long enough to latch its spell.
pub(super) fn latched(hand: &Hand) -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, hand);
    for _ in 0..10 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
    }
    t
}

/// The same, but sweeping right at 0.36 normalised units per second.
pub(super) fn sweeping(hand: &Hand) -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    let mut moving = *hand;
    for _ in 0..60 {
        synth::write(&mut buf, 0, &moving);
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        moving.x += 0.012;
    }
    t
}

/// Two palms rotating about the frame centre at `omega` radians per second.
pub(super) fn rotating(omega: f32, frames: usize) -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    for frame in 0..frames {
        let theta = omega * frame as f32 * HAND_DT;
        let (c, s) = (theta.cos(), theta.sin());
        synth::write(&mut buf, 0, &Hand::at(0.5 + 0.12 * c, 0.5 + 0.12 * s));
        synth::write(&mut buf, 1, &Hand::at(0.5 - 0.12 * c, 0.5 - 0.12 * s));
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
    }
    t
}

/// Grid-space palm of slot 0.
pub(super) fn palm_of(tracker: &GestureTracker) -> [f32; 2] {
    to_grid(
        tracker.hands()[0].palm,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    )
}
