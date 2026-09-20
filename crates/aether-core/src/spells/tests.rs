use super::*;
use crate::config::{FLOW_H, FLOW_W, FLUID_H, FLUID_W};
use crate::field::Grid;
use crate::gesture::synth::{self, Hand};
use crate::gesture::{GestureConfig, GestureTracker};
use crate::particles::ParticleConfig;

/// Render step.
const DT: f32 = 1.0 / 60.0;
/// Inference step, which is what the tracker is fed at.
const HAND_DT: f32 = 1.0 / 30.0;

struct Rig {
    fluid: Fluid,
    particles: Particles,
    flow: VecField,
    body: VecField,
    state: SpellState,
    params: Params,
}

impl Rig {
    /// The field drives are off by default so the gesture paths can be
    /// measured on their own; the two tests that care switch them on.
    fn new() -> Self {
        Self::sized(FLUID_W, FLUID_H)
    }

    /// A rig on a smaller grid, for the tests that need hundreds of solver
    /// steps rather than a realistic resolution.
    fn sized(w: usize, h: usize) -> Self {
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

    fn run(&mut self, tracker: &GestureTracker, dt: f32) -> SpellReport {
        self.run_with(tracker, None, dt)
    }

    /// One step with a completed sequence handed in, as the engine does it.
    fn run_with(
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

    fn velocity_at(&self, x: f32, y: f32) -> (f32, f32) {
        self.fluid.velocity().sample(x, y)
    }
}

fn empty_tracker() -> GestureTracker {
    let mut t = GestureTracker::new(GestureConfig::default());
    t.update_hands(&synth::buffer(), HAND_DT);
    t.update_two_hand(HAND_DT);
    t
}

/// A tracker holding one still hand, held long enough to latch and to age
/// past the vortex windup.
fn holding(hand: &Hand) -> GestureTracker {
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
fn latched(hand: &Hand) -> GestureTracker {
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
fn sweeping(hand: &Hand) -> GestureTracker {
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
fn rotating(omega: f32, frames: usize) -> GestureTracker {
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
fn palm_of(tracker: &GestureTracker) -> [f32; 2] {
    to_grid(
        tracker.hands()[0].palm,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    )
}

const IDLE_HAND: Hand = Hand {
    x: 0.4,
    y: 0.5,
    scale: 0.13,
    gesture: synth::NONE,
    // Unconfident, so no label path fires and the geometry is ambiguous.
    score: 0.1,
    pinched: false,
    right: true,
};

#[test]
fn an_idle_present_hand_injects_energy_and_an_absent_one_does_not() {
    let tracker = sweeping(&IDLE_HAND);
    assert_eq!(tracker.hands()[0].spell, Spell::Idle, "expected no spell");
    assert!(
        tracker.hands()[0].velocity[0] > 0.2,
        "hand should be moving"
    );

    let mut rig = Rig::new();
    let report = rig.run(&tracker, DT);
    assert_eq!(report.spells[0], Spell::Idle);
    assert!(
        report.injected > 0.0,
        "an idle hand must still drive the fluid"
    );
    assert!(rig.fluid.energy() > 0.0);
    // It must push the way the hand is going.
    let palm = palm_of(&tracker);
    assert!(rig.velocity_at(palm[0], palm[1]).0 > 0.0);

    let mut rig = Rig::new();
    let report = rig.run(&empty_tracker(), DT);
    assert_eq!(report.injected, 0.0, "no hand, no energy");
    assert_eq!(rig.fluid.energy(), 0.0);
    assert_eq!(report.spells, [Spell::Idle, Spell::Idle]);
}

#[test]
fn every_spell_keeps_the_fluid_finite_and_under_the_ceiling() {
    let cases = [
        (Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM), Spell::Repel),
        (
            Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST),
            Spell::Attract,
        ),
        (
            Hand::at(0.5, 0.5).gesture(synth::POINTING_UP),
            Spell::Ignite,
        ),
        (Hand::at(0.5, 0.5).gesture(synth::VICTORY), Spell::Freeze),
        (Hand::at(0.5, 0.5).gesture(synth::THUMB_UP), Spell::Shatter),
        (
            Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched(),
            Spell::Vortex,
        ),
        (IDLE_HAND, Spell::Idle),
    ];
    for (hand, want) in cases {
        let tracker = sweeping(&hand);
        assert_eq!(tracker.hands()[0].spell, want, "wrong spell latched");
        let mut rig = Rig::new();
        for _ in 0..4 {
            let report = rig.run(&tracker, DT);
            assert!(report.injected.is_finite(), "{want:?} injected NaN");
            assert!(report.time_scale.is_finite());
        }
        let vel = rig.fluid.velocity();
        assert!(
            vel.u.max_abs() <= MAX_IMPULSE && vel.v.max_abs() <= MAX_IMPULSE,
            "{want:?} exceeded MAX_IMPULSE: {} / {}",
            vel.u.max_abs(),
            vel.v.max_abs()
        );
        assert_eq!(rig.fluid.sanitize(), 0, "{want:?} wrote a non-finite cell");
        for channel in rig.fluid.dye() {
            assert!(
                channel.data.iter().all(|v| v.is_finite()),
                "{want:?} wrote non-finite dye"
            );
        }
    }
}

#[test]
fn absurd_parameters_are_clamped_to_the_impulse_ceiling() {
    // hand_force saturates at 8 and dt at 0.25, which together ask for a
    // 3000 cells/s push — the clamp is the only thing between that and a
    // velocity field the projection cannot resolve.
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut rig = Rig::new();
    rig.params.hand_force = 8.0;
    rig.run(&tracker, 1e6);
    let vel = rig.fluid.velocity();
    assert!(
        (vel.u.max_abs() - MAX_IMPULSE).abs() < 1e-3,
        "clamp did not bite: {}",
        vel.u.max_abs()
    );
    assert!(vel.v.max_abs() <= MAX_IMPULSE);
    assert_eq!(rig.fluid.sanitize(), 0);
}

#[test]
fn shatter_fires_once_per_hold_and_rearms_on_release() {
    let hand = Hand::at(0.5, 0.5).gesture(synth::THUMB_UP);
    let mut tracker = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, &hand);
    let mut rig = Rig::new();

    let mut bursts = 0;
    for _ in 0..30 {
        tracker.update_hands(&buf, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(tracker.hands()[0].spell, Spell::Shatter);
    assert_eq!(bursts, 1, "holding the gesture re-fired the burst");

    // Drop the hand for long enough to be released, then show it again.
    let empty = synth::buffer();
    for _ in 0..12 {
        tracker.update_hands(&empty, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(bursts, 1, "a burst fired while no hand was present");
    for _ in 0..30 {
        tracker.update_hands(&buf, HAND_DT);
        tracker.update_two_hand(HAND_DT);
        bursts += rig.run(&tracker, DT).bursts;
    }
    assert_eq!(bursts, 2, "the burst did not re-arm after release");

    // And the burst actually did something to the pool.
    assert!(
        (0..rig.particles.active()).any(|i| rig.particles.heat_of(i) > 0.9),
        "no particle came out of the burst hot"
    );
}

#[test]
fn ignite_paints_the_whole_stroke_not_just_the_endpoints() {
    let mut rig = Rig::new();
    let left = holding(&Hand::at(0.15, 0.5).gesture(synth::POINTING_UP));
    assert_eq!(left.hands()[0].spell, Spell::Ignite);
    rig.run(&left, DT);
    let a = to_grid(
        left.hands()[0].index_tip,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );

    // The finger is now far away; one frame of dye must cover the gap.
    let right = holding(&Hand::at(0.85, 0.5).gesture(synth::POINTING_UP));
    rig.run(&right, DT);
    let b = to_grid(
        right.hands()[0].index_tip,
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    let span = length(b[0] - a[0], b[1] - a[1]);
    assert!(span > 100.0, "the test needs a long stroke, got {span}");

    let red = &rig.fluid.dye()[0];
    for t in [0.25, 0.5, 0.75] {
        let x = lerp(a[0], b[0], t);
        let y = lerp(a[1], b[1], t);
        assert!(
            red.sample(x, y) > 1e-4,
            "no dye at {t} along the stroke ({x}, {y})"
        );
    }
    // Mid-stroke brightness must be the same order as the endpoints, not a
    // faint smear from the splat tails.
    let mid = red.sample(lerp(a[0], b[0], 0.5), lerp(a[1], b[1], 0.5));
    let end = red.sample(b[0], b[1]);
    assert!(mid * 4.0 > end, "stroke is dotted: mid {mid} vs end {end}");
}

#[test]
fn vortex_winds_up_and_the_flow_is_tangential() {
    let hand = Hand::at(0.5, 0.5).gesture(synth::NONE).score(0.1).pinched();

    // Freshly latched versus held past the windup.
    let mut young = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    synth::write(&mut buf, 0, &hand);
    for _ in 0..GestureConfig::default().commit_frames {
        young.update_hands(&buf, HAND_DT);
        young.update_two_hand(HAND_DT);
    }
    assert_eq!(young.hands()[0].spell, Spell::Vortex);
    let mut fresh_rig = Rig::new();
    fresh_rig.run(&young, DT);
    let fresh = fresh_rig.fluid.energy();

    let old = holding(&hand);
    assert!(old.hands()[0].spell_age > VORTEX_WINDUP);
    let mut wound_rig = Rig::new();
    wound_rig.run(&old, DT);
    let wound = wound_rig.fluid.energy();
    assert!(
        wound > fresh * 4.0,
        "vortex did not wind up: {fresh} -> {wound}"
    );

    // Sample to the right of the pinch point: the flow there must be
    // perpendicular to the radius.
    let tip = old.hands()[0].index_tip;
    let thumb = old.hands()[0].thumb_tip;
    let center = to_grid(
        [(tip[0] + thumb[0]) * 0.5, (tip[1] + thumb[1]) * 0.5],
        (FLUID_W - 1) as f32,
        (FLUID_H - 1) as f32,
    );
    let radius = old.hands()[0].scale * (FLUID_W - 1) as f32 * PALM_RADIUS_SCALE;
    let probe = radius * VORTEX_RADIUS_SCALE * 0.4;
    let (u, v) = wound_rig.velocity_at(center[0] + probe, center[1]);
    assert!(v.abs() > 1.0, "no swirl at the probe: {v}");
    assert!(
        u.abs() < 0.35 * v.abs(),
        "flow is radial, not tangential: u {u} v {v}"
    );
    // Opposite sides of the core circulate opposite ways.
    let (_, v_left) = wound_rig.velocity_at(center[0] - probe, center[1]);
    assert!(v * v_left < 0.0, "not a circulation: {v} vs {v_left}");
}

#[test]
fn repel_pushes_out_and_attract_pulls_in_more_gently() {
    let mut out = Rig::new();
    let repel = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let palm = palm_of(&repel);
    // A particle just to the right of the palm, to check the particle path.
    out.particles
        .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
    out.run(&repel, DT);
    let (ru, _) = out.velocity_at(palm[0] + 10.0, palm[1]);
    assert!(ru > 0.0, "repel pushed inward: {ru}");
    assert!(
        out.particles.velocity(0).0 > 0.0,
        "repel did not push particles out"
    );

    let mut inward = Rig::new();
    let attract = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    inward
        .particles
        .place(0, palm[0] + 6.0, palm[1], 0.0, 0.0, 3.0);
    inward.run(&attract, DT);
    let (au, _) = inward.velocity_at(palm[0] + 10.0, palm[1]);
    assert!(au < 0.0, "attract pushed outward: {au}");
    assert!(
        inward.particles.velocity(0).0 < 0.0,
        "attract did not pull particles in"
    );
    assert!(
        au.abs() < ru.abs() * 0.8,
        "attract must be the weaker well: {au} vs {ru}"
    );

    // Both should be symmetric about the palm.
    let (lu, _) = out.velocity_at(palm[0] - 10.0, palm[1]);
    assert!((lu + ru).abs() < 0.2 * ru.abs(), "asymmetric push");
}

#[test]
fn every_combo_payload_moves_the_field_and_stays_finite() {
    for effect in [
        ComboEffect::Nova,
        ComboEffect::Tempest,
        ComboEffect::Supernova,
    ] {
        let mut rig = Rig::new();
        let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
        let hit = ComboHit {
            combo: 0,
            effect,
            slot: 0,
        };
        let report = rig.run_with(&tracker, Some(hit), DT);
        assert_eq!(report.bursts, 1, "{effect:?} did not report its burst");
        assert!(
            report.injected > 0.0 && report.injected.is_finite(),
            "{effect:?} injected {}",
            report.injected
        );
        // A payload is an event; it must not leave the solver unstable.
        for _ in 0..30 {
            rig.run(&tracker, DT);
        }
        assert_eq!(rig.fluid.sanitize(), 0, "{effect:?} destabilised the fluid");
    }
}

#[test]
fn freeze_damps_the_fluid_locally_and_leaves_the_rest_alone() {
    let mut rig = Rig::new();
    rig.fluid.velocity_mut().u.data.fill(50.0);
    let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::VICTORY));
    assert_eq!(tracker.hands()[0].spell, Spell::Freeze);
    let palm = palm_of(&tracker);
    for _ in 0..10 {
        rig.run(&tracker, DT);
    }
    let (near, _) = rig.velocity_at(palm[0], palm[1]);
    // Not a "did it go down" test: the always-on palm drag relaxes toward a
    // still hand's zero velocity all by itself, so a loose threshold here
    // passes for `Idle` too and proves nothing about the spell. Measure the
    // spell alone with the drag switched off, against its own rate.
    assert!(near < 40.0, "freeze did not damp the local flow: {near}");
    let (far, _) = rig.velocity_at(5.0, 5.0);
    assert!((far - 50.0).abs() < 1e-3, "freeze damped the whole grid");
    // And it lays down cold dye: hue 0.55 has no red in it.
    let dye = rig.fluid.dye();
    assert!(dye[2].sample(palm[0], palm[1]) > dye[0].sample(palm[0], palm[1]));

    // The falloff weight is ~1 over the middle of the disc, so the centre
    // must decay by exactly `exp(-FREEZE_RATE * t)` — and must do so at any
    // frame rate, since the weight sits inside the decay rate.
    for (hz, frames) in [(60.0f32, 30u32), (240.0, 120)] {
        let mut rig = Rig::new();
        rig.params.hand_force = 0.0;
        rig.fluid.velocity_mut().u.data.fill(50.0);
        for _ in 0..frames {
            rig.run(&tracker, 1.0 / hz);
        }
        let elapsed = frames as f32 / hz;
        let want = 50.0 * decay(FREEZE_RATE, elapsed);
        let got = rig.velocity_at(palm[0], palm[1]).0;
        assert!(
            (got - want).abs() < 0.02 * want,
            "{hz} Hz: freeze centre {got}, analytic {want}"
        );
    }

    // Nothing but the drag: an idle hand must leave the field alone once
    // the drag is off, which is what makes the comparison above meaningful.
    let mut idle = Rig::new();
    idle.params.hand_force = 0.0;
    idle.fluid.velocity_mut().u.data.fill(50.0);
    let idle_tracker = holding(&IDLE_HAND);
    assert_eq!(idle_tracker.hands()[0].spell, Spell::Idle);
    for _ in 0..30 {
        idle.run(&idle_tracker, 1.0 / 60.0);
    }
    let palm = palm_of(&idle_tracker);
    assert!(
        (idle.velocity_at(palm[0], palm[1]).0 - 50.0).abs() < 1e-3,
        "an idle hand damped the field"
    );
}

#[test]
fn an_ignite_stroke_respects_the_impulse_ceiling() {
    // The trail is a line of deliberately overlapping splats: they are
    // spaced `IGNITE_STEP` apart and each is `IGNITE_RADIUS` wide, so
    // clamping them one at a time is not enough — the sum is what lands in
    // the velocity field. Worst case is the module's own `MAX_DT` and the
    // largest `hand_force` `Params::set` allows.
    let mut worst = 0.0f32;
    for cells in [0.0f32, 1.2, 2.2, 3.4, 4.8, 9.0, 40.0, 200.0] {
        let mut rig = Rig::new();
        rig.params.hand_force = 8.0;
        let a = holding(&Hand::at(0.3, 0.5).gesture(synth::POINTING_UP));
        rig.run(&a, MAX_DT);
        let b =
            holding(&Hand::at(0.3 + cells / (FLUID_W - 1) as f32, 0.5).gesture(synth::POINTING_UP));
        rig.run(&b, MAX_DT);
        let vel = rig.fluid.velocity();
        worst = worst.max(vel.u.max_abs()).max(vel.v.max_abs());
    }
    assert!(
        worst <= MAX_IMPULSE,
        "ignite drove the field to {worst} cells/s, past the {MAX_IMPULSE} ceiling"
    );
}

#[test]
fn time_scale_follows_the_two_hand_rotation_and_stays_clamped() {
    let mut rig = Rig::new();
    let forward = rotating(2.5, 90);
    assert!(forward.two_hand().both_present);
    let mut ts = 1.0;
    for _ in 0..60 {
        ts = rig.run(&forward, DT).time_scale;
    }
    assert!(ts > 1.1, "rotation did not warp time: {ts}");
    assert!(ts <= MAX_TIME_SCALE);

    let backward = rotating(-2.5, 90);
    for _ in 0..60 {
        ts = rig.run(&backward, DT).time_scale;
    }
    assert!(ts < 0.9, "reverse rotation should slow time: {ts}");
    assert!(ts >= MIN_TIME_SCALE);

    // An absurd spin must saturate, not run away.
    let wild = rotating(60.0, 120);
    for _ in 0..240 {
        ts = rig.run(&wild, DT).time_scale;
    }
    assert!(
        (MIN_TIME_SCALE..=MAX_TIME_SCALE).contains(&ts),
        "time scale escaped its bounds: {ts}"
    );

    // And it eases back to normal once the hands are gone.
    let none = empty_tracker();
    for _ in 0..120 {
        ts = rig.run(&none, DT).time_scale;
    }
    assert!((ts - 1.0).abs() < 0.05, "time scale did not recover: {ts}");
}

#[test]
fn the_whole_field_drives_push_the_fluid() {
    // Optical flow: model-free, so it must work with no tracker at all.
    let mut rig = Rig::new();
    rig.params.flow_force = 0.45;
    rig.flow.u.data.fill(4.0);
    let report = rig.run(&empty_tracker(), DT);
    assert!(report.injected > 0.0);
    let (u, _) = rig.velocity_at(100.0, 70.0);
    assert!(u > 0.0, "flow drive pushed nothing: {u}");

    // Body edge velocity, injected only where the silhouette moves.
    let mut rig = Rig::new();
    rig.params.body_push = 1.0;
    for y in 60..80 {
        for x in 100..120 {
            rig.body.u.set(x, y, -6.0);
        }
    }
    rig.run(&empty_tracker(), DT);
    let (inside, _) = rig.velocity_at(110.0, 70.0);
    let (outside, _) = rig.velocity_at(10.0, 10.0);
    assert!(inside < 0.0, "body edge did not push: {inside}");
    assert_eq!(outside, 0.0, "body edge leaked across the grid");

    // A zero body field must cost nothing and change nothing.
    let mut rig = Rig::new();
    rig.params.body_push = 1.0;
    assert_eq!(rig.run(&empty_tracker(), DT).injected, 0.0);
}

#[test]
fn forces_are_framerate_independent() {
    // The same simulated interval, split six ways. Both paths must deposit
    // the same energy; a `1 - rate * dt` decay or an unscaled impulse fails
    // this by a factor of six.
    let tracker = sweeping(&IDLE_HAND);
    let mut coarse = Rig::new();
    coarse.run(&tracker, DT);
    let mut fine = Rig::new();
    for _ in 0..6 {
        fine.run(&tracker, DT / 6.0);
    }
    let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
    assert!(a > 0.0);
    assert!(
        (a - b).abs() < 0.005 * a,
        "frame rate changed the outcome: {a} vs {b}"
    );

    // Repel is a pure dt-scaled impulse. Probed outside the palm drag's
    // radius but inside the push's, where nothing else has touched the
    // field, the two paths must agree to floating-point noise.
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let palm = palm_of(&tracker);
    let mut coarse = Rig::new();
    coarse.run(&tracker, DT);
    let mut fine = Rig::new();
    for _ in 0..6 {
        fine.run(&tracker, DT / 6.0);
    }
    let probe = palm[0] + 50.0;
    let (a, b) = (
        coarse.velocity_at(probe, palm[1]).0,
        fine.velocity_at(probe, palm[1]).0,
    );
    assert!(a > 1.0, "the probe should see the push: {a}");
    assert!(
        (a - b).abs() < 1e-3 * a,
        "impulse not dt-scaled: {a} vs {b}"
    );

    // Over the whole field the two differ by a few percent, and that is
    // operator splitting rather than a dt bug: the fine path interleaves
    // six drag passes with six impulses, so the drag sees — and damps — a
    // field the coarse path only produces after its single drag pass.
    let (a, b) = (coarse.fluid.energy(), fine.fluid.energy());
    assert!(
        (a - b).abs() < 0.04 * a,
        "splitting error too large: {a} vs {b}"
    );
}

#[test]
fn a_held_spell_survives_a_long_stepped_run() {
    // The real failure mode this guards: a force that is individually
    // clamped but accumulates every frame until the solver diverges.
    for gesture in [
        synth::OPEN_PALM,
        synth::CLOSED_FIST,
        synth::POINTING_UP,
        synth::VICTORY,
        synth::THUMB_UP,
    ] {
        let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(gesture));
        // A sixteenth of the real grid: this test is about what accumulates
        // over 240 steps, which is resolution independent, and at full size
        // five of these runs take a minute and a half in a debug build.
        let mut rig = Rig::sized(64, 36);
        for _ in 0..240 {
            rig.run(&tracker, DT);
            rig.fluid.step(DT, &rig.params);
        }
        assert_eq!(
            rig.fluid.sanitize(),
            0,
            "gesture {gesture} produced non-finite cells"
        );
        let speed = rig.fluid.max_speed();
        assert!(
            speed.is_finite() && speed < MAX_IMPULSE * 4.0,
            "gesture {gesture} diverged: {speed} cells/s"
        );
    }
}

#[test]
fn garbage_input_never_panics() {
    let tracker = sweeping(&Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));

    // Absurd timesteps.
    let mut rig = Rig::new();
    for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 1e9] {
        let report = rig.run(&tracker, dt);
        assert!(report.time_scale.is_finite());
        assert!(report.injected.is_finite());
    }
    assert_eq!(rig.fluid.sanitize(), 0);

    // A non-finite flow field must not reach the velocity grid.
    let mut rig = Rig::new();
    rig.params.flow_force = 1.0;
    rig.params.body_push = 1.0;
    rig.flow.u.data.fill(f32::NAN);
    rig.body.v.data.fill(f32::INFINITY);
    rig.run(&tracker, DT);
    assert_eq!(rig.fluid.sanitize(), 0, "NaN flow leaked into the fluid");

    // A landmark-free "present" hand, and hands off the edge of the frame.
    let mut rig = Rig::new();
    for hand in [
        Hand::at(-4.0, 9.0).gesture(synth::OPEN_PALM),
        Hand::at(0.5, 0.5).gesture(synth::POINTING_UP).scaled(0.9),
        Hand::at(0.5, 0.5).gesture(synth::THUMB_UP).scaled(0.006),
    ] {
        let tracker = holding(&hand);
        rig.run(&tracker, DT);
    }
    assert_eq!(rig.fluid.sanitize(), 0);

    // The smallest grid a Fluid can exist on, and an empty particle pool.
    let mut tiny = Fluid::new(2, 2);
    let mut none = Particles::new(0, 1);
    let mut state = SpellState::new();
    let report = apply(
        &tracker,
        &VecField::new(2, 2),
        &VecField::new(2, 2),
        &mut tiny,
        &mut none,
        &mut state,
        &Params::default(),
        [None; 2],
        [0.0; 2],
        DuetFrame::default(),
        DT,
    );
    assert!(report.injected.is_finite());
    assert_eq!(tiny.sanitize(), 0);
}

#[test]
fn a_sweeping_open_palm_still_repels() {
    let tracker = sweeping(&Hand::at(0.3, 0.5).gesture(synth::OPEN_PALM));
    assert_eq!(tracker.hands()[0].spell, Spell::Repel);
}

#[test]
fn a_closed_fist_gathers_particles_instead_of_slinging_them_past() {
    let tracker = latched(&Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    let hand = &tracker.hands()[0];
    let (sx, sy) = ((FLUID_W - 1) as f32, (FLUID_H - 1) as f32);
    let palm = to_grid(hand.palm, sx, sy);

    let mut rig = Rig::new();
    rig.particles
        .place(0, palm[0] + 10.0, palm[1], 0.0, 0.0, 30.0);
    let obstacle = Grid::new(FLUID_W, FLUID_H);
    let cfg = ParticleConfig::default();
    for _ in 0..60 {
        rig.run(&tracker, DT);
        rig.particles
            .step(rig.fluid.velocity(), &obstacle, DT, &rig.params, &cfg);
    }
    let (px, py) = rig.particles.position(0);
    let dist = length(px - palm[0], py - palm[1]);
    let (vx, vy) = rig.particles.velocity(0);
    assert!(dist < 10.0, "particle was not gathered: {dist} cells out");
    // Held, not orbiting: the grip has taken the speed off it.
    assert!(length(vx, vy) < 60.0, "particle still slinging: {vx} {vy}");
}

#[test]
fn one_pointing_hand_is_still_an_ignite() {
    let tracker = holding(&Hand::at(0.5, 0.5).gesture(synth::POINTING_UP));
    let mut rig = Rig::new();
    let report = rig.run(&tracker, DT);
    assert_eq!(report.spells[0], Spell::Ignite);
}

#[test]
fn opening_a_held_fist_fires_a_ring_once() {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    let mut rig = Rig::new();
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    for _ in 0..45 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        rig.run(&t, HAND_DT);
    }
    assert_eq!(t.hands()[0].spell, Spell::Attract);
    let palm = palm_of(&t);

    // Still the pool first so the only fast particles afterwards are the
    // ring's own, not two seconds of attract pull.
    rig.particles
        .seed_uniform(FLUID_W, FLUID_H, rig.params.particle_life);
    // Open the hand: the tracker takes `commit_frames` to relatch.
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut bursts = 0;
    let mut saw_release = false;
    for _ in 0..12 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        let report = rig.run(&t, HAND_DT);
        bursts += report.bursts;
        saw_release |= report.spells[0] == Spell::Release;
    }
    assert_eq!(bursts, 1, "the ring must fire exactly once per open");
    assert!(saw_release, "the release should be reported to the HUD");
    // Particles were re-seeded on a ring and fly outward from the palm.
    let mut outward = 0;
    let mut total = 0;
    for i in 0..rig.particles.active() {
        let (px, py) = rig.particles.position(i);
        let (vx, vy) = rig.particles.velocity(i);
        let speed = length(vx, vy);
        if speed < 50.0 {
            continue;
        }
        total += 1;
        if (px - palm[0]) * vx + (py - palm[1]) * vy > 0.0 {
            outward += 1;
        }
    }
    assert!(
        total > 100,
        "expected a burst of fast particles, got {total}"
    );
    assert!(
        outward * 10 > total * 9,
        "burst is not outward: {outward}/{total}"
    );
}

#[test]
fn a_fist_barely_closed_does_not_fire_a_ring() {
    let mut t = GestureTracker::new(GestureConfig::default());
    let mut buf = synth::buffer();
    let mut rig = Rig::new();
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::CLOSED_FIST));
    for _ in 0..4 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        rig.run(&t, HAND_DT);
    }
    assert_eq!(t.hands()[0].spell, Spell::Attract);
    synth::write(&mut buf, 0, &Hand::at(0.5, 0.5).gesture(synth::OPEN_PALM));
    let mut bursts = 0;
    for _ in 0..12 {
        t.update_hands(&buf, HAND_DT);
        t.update_two_hand(HAND_DT);
        bursts += rig.run(&t, HAND_DT).bursts;
    }
    assert_eq!(bursts, 0, "an uncharged fist must not fire");
}
