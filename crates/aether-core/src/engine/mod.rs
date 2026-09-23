//! The orchestrator: owns every subsystem and the buffers that cross into JS.
//!
//! # Zero-copy boundary
//!
//! Per-frame data is never serialised. JS writes camera luma and segmentation
//! masks straight into engine-owned buffers through raw pointers, and reads the
//! dye texture and particle buffer back the same way. Landmarks are small
//! enough that they come in as slices.
//!
//! The one rule: a pointer is only valid until the next call that can grow a
//! buffer ([`Engine::set_particle_count`] or [`Engine::reset`]). The JS side
//! re-reads pointers after those, and after any WASM memory growth.
//!
//! # Frame protocol
//!
//! ```text
//! // once per camera frame (~30 Hz)
//! write luma into luma_ptr(); engine.push_luma(dt_since_last_luma)
//! // once per perception result (~30 Hz, may lag the camera)
//! write masks into mask_ptr(); engine.push_mask(w, h, mirror, dt)
//! engine.push_hands(&packed, dt); engine.push_pose(&packed, dt)
//! // once per animation frame (~60 Hz)
//! engine.step(dt)  -> then read dye_ptr() and particle_ptr()
//! ```

use crate::combo::{ComboProgress, ComboTracker};
use crate::config::{
    Params, DEFAULT_PARTICLES, FLOW_CELLS, FLOW_H, FLOW_W, FLUID_CELLS, FLUID_H, FLUID_W,
    HAND_BUFFER, MAX_PARTICLES, PARTICLE_STRIDE, POSE_STRIDE,
};
use crate::duet::{DuetProgress, DuetTracker};
use crate::field::{Grid, VecField};
use crate::flow::{FlowConfig, OpticalFlow};
use crate::fluid::Fluid;
use crate::gesture::{GestureConfig, GestureTracker, Spell};
use crate::mask::{BodyMask, MaskConfig};
use crate::particles::{ParticleConfig, Particles};
use crate::spells::{SpellReport, SpellState};

/// Largest segmentation mask the input buffer accepts, per axis. MediaPipe's
/// pose model emits 256x256; anything bigger must be downscaled by the caller.
pub const MASK_IN_MAX_DIM: usize = 320;
/// Capacity of the mask input buffer, in floats.
pub const MASK_IN_CAPACITY: usize = MASK_IN_MAX_DIM * MASK_IN_MAX_DIM;

/// Seconds without any perception input before the engine drives itself.
const AMBIENT_AFTER: f32 = 1.5;

/// Frames between the three grid-walking HUD statistics.
///
/// `fluid_energy` and `fluid_max_speed` each traverse the whole velocity field.
/// The HUD repaints at 10 Hz, so refreshing them at 60 Hz burns full grid
/// passes to produce numbers nobody reads. Every other stat is an O(1) read and
/// stays live every frame, because callers treat those as state rather than
/// telemetry.
const STATS_EVERY: u32 = 4;

/// Frames between defensive NaN scrubs.
///
/// The simulation should never produce a non-finite cell. If it does, one
/// escaped NaN spreads through the field within a few frames and blanks the
/// screen, so it has to be caught — but checking five grids every frame is a
/// cost paid forever against something that should never happen. Every fourth
/// frame bounds the damage to ~66 ms of corruption.
const SANITIZE_EVERY: u32 = 4;

/// Resolution of the dye tone-mapping table, and the dye value it saturates at.
///
/// `encode_dye` needs `1 - exp(-x)` three times per cell, which is 110k
/// transcendental calls per frame at this grid size — measurably more than the
/// entire particle system costs. A table over the curve's useful range, read
/// with linear interpolation, is visually indistinguishable and an order of
/// magnitude cheaper. Beyond `TONEMAP_MAX` the curve is within 0.2% of 1.0, so
/// clamping there loses nothing.
const TONEMAP_LUT: usize = 512;
const TONEMAP_MAX: f32 = 8.0;

mod frame;
mod gpu;
mod output;
mod perception;
mod stats;
#[cfg(test)]
mod tests;

pub use stats::Stats;

use output::build_tonemap_lut;

/// The complete Aether engine.
pub struct Engine {
    params: Params,
    particle_cfg: ParticleConfig,

    fluid: Fluid,
    particles: Particles,
    flow: OpticalFlow,
    body: BodyMask,
    tracker: GestureTracker,
    spell_state: SpellState,
    combos: ComboTracker,
    duets: DuetTracker,
    /// Practice mode: combos and duets are still tracked but never cast, so the
    /// tutorial can walk one pose after another without chaining them.
    practice: bool,

    // --- JS-writable input buffers ---
    luma: Vec<u8>,
    mask_in: Vec<f32>,

    // --- JS-readable output buffers ---
    dye_rgba: Vec<u8>,
    debug_rgba: Vec<u8>,

    /// Zero velocity field used when no body edge motion is available.
    idle_field: VecField,

    /// `1 - exp(-x)` sampled over `[0, TONEMAP_MAX]`, built once.
    tonemap_lut: Vec<f32>,
    /// True when a mask arrived since the last step, or the obstacle field is
    /// still fading out. Lets `step` skip re-uploading an unchanged obstacle,
    /// which is a full grid resample plus a wall-mask rebuild.
    obstacle_dirty: bool,

    time: f32,
    frame: u32,
    /// The warped `dt` the last `step` advanced the simulation by; what a GPU
    /// particle step has to integrate to stay in lockstep with the fluid.
    last_sim_dt: f32,
    /// Seconds since any hand, pose or mask arrived.
    since_perception: f32,
    stats: Stats,
    report: SpellReport,
    seed: u64,
}
impl Engine {
    pub fn new(seed: u64) -> Self {
        let mut particles = Particles::new(MAX_PARTICLES, seed);
        particles.set_active(DEFAULT_PARTICLES);
        let params = Params::default();
        particles.seed_uniform(FLUID_W, FLUID_H, params.particle_life);

        Self {
            params,
            particle_cfg: ParticleConfig::default(),
            fluid: Fluid::new(FLUID_W, FLUID_H),
            particles,
            flow: OpticalFlow::new(FLOW_W, FLOW_H, FlowConfig::default()),
            body: BodyMask::new(FLUID_W, FLUID_H, MaskConfig::default()),
            tracker: GestureTracker::new(GestureConfig::default()),
            spell_state: SpellState::new(),
            combos: ComboTracker::new(),
            duets: DuetTracker::new(),
            practice: false,
            luma: vec![0; FLOW_CELLS],
            mask_in: vec![0.0; MASK_IN_CAPACITY],
            dye_rgba: vec![0; FLUID_CELLS * 4],
            debug_rgba: vec![0; FLUID_CELLS * 4],
            idle_field: VecField::new(FLUID_W, FLUID_H),
            tonemap_lut: build_tonemap_lut(),
            obstacle_dirty: true,
            time: 0.0,
            frame: 0,
            last_sim_dt: 0.0,
            since_perception: f32::MAX / 4.0,
            stats: Stats {
                time_scale: 1.0,
                ..Default::default()
            },
            report: SpellReport {
                time_scale: 1.0,
                ..Default::default()
            },
            seed,
        }
    }

    // ---------------------------------------------------------------- layout

    /// Grid dimensions, so JS can assert its mirrored constants match.
    pub fn layout(&self) -> [u32; 6] {
        [
            FLUID_W as u32,
            FLUID_H as u32,
            FLOW_W as u32,
            FLOW_H as u32,
            MAX_PARTICLES as u32,
            PARTICLE_STRIDE as u32,
        ]
    }

    /// Latched spell for a hand slot, as a stable string for the HUD.
    pub fn spell_name(&self, slot: usize) -> &'static str {
        self.report
            .spells
            .get(slot)
            .copied()
            .unwrap_or(Spell::Idle)
            .name()
    }

    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// How far the current gesture sequence has got, for the HUD's spellbook.
    #[inline]
    pub fn combo_progress(&self) -> ComboProgress {
        self.combos.progress()
    }

    /// Turns practice mode on or off. Entering it clears whatever sequence was
    /// half-cast, so the tutorial never starts with a primed lane.
    pub fn set_practice(&mut self, on: bool) {
        if on && !self.practice {
            self.combos.reset();
            self.duets.reset();
        }
        self.practice = on;
    }

    /// Which two-hand duet is being wound up, for the HUD's spellbook.
    #[inline]
    pub fn duet_progress(&self) -> DuetProgress {
        self.duets.progress()
    }

    /// Retained renderer contract: throws no longer stage a lens rush.
    #[inline]
    pub fn rush_state(&self) -> crate::rush::RushState {
        crate::rush::RushState::IDLE
    }

    #[inline]
    pub fn tracker(&self) -> &GestureTracker {
        &self.tracker
    }

    #[inline]
    pub fn fluid(&self) -> &Fluid {
        &self.fluid
    }

    #[inline]
    pub fn body(&self) -> &BodyMask {
        &self.body
    }

    #[inline]
    pub fn flow_field(&self) -> &VecField {
        self.flow.flow()
    }

    #[inline]
    pub fn obstacle(&self) -> &Grid {
        self.fluid.obstacle()
    }

    // ---------------------------------------------------------------- config

    #[inline]
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Sets a named tunable. Returns false for an unknown key.
    pub fn set_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "curl_influence" => {
                self.particle_cfg.curl_influence = value.clamp(0.0, 20.0);
                true
            }
            "gravity" => {
                self.particle_cfg.gravity = value.clamp(-200.0, 200.0);
                true
            }
            "damping" => {
                self.particle_cfg.damping = value.clamp(0.0, 20.0);
                true
            }
            _ => self.params.set(key, value),
        }
    }

    /// Resizes the live particle pool, re-seeding the newly activated slots.
    pub fn set_particle_count(&mut self, n: usize) {
        let n = n.min(MAX_PARTICLES);
        let previous = self.particles.active();
        self.particles.set_active(n);
        if n > previous {
            self.particles
                .seed_uniform(FLUID_W, FLUID_H, self.params.particle_life);
        }
    }

    /// Full reset: clears the fluid, re-seeds particles, forgets perception.
    pub fn reset(&mut self) {
        let active = self.particles.active();
        let offloaded = self.particles.offloaded();
        self.fluid.reset();
        self.particles = Particles::new(MAX_PARTICLES, self.seed);
        self.particles.set_active(active);
        self.particles.set_offloaded(offloaded);
        self.particles
            .seed_uniform(FLUID_W, FLUID_H, self.params.particle_life);
        self.clear_perception();
        self.time = 0.0;
        self.frame = 0;
        self.dye_rgba.fill(0);
        self.obstacle_dirty = true;
        // The HUD reads a cached snapshot, so clearing the counters without
        // clearing `stats` would leave the old frame count on screen forever.
        self.stats = Stats {
            time_scale: 1.0,
            ..Default::default()
        };
        self.report = SpellReport {
            time_scale: 1.0,
            ..Default::default()
        };
        self.combos.reset();
        self.duets.reset();
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(0xA37E_5EED)
    }
}

/// Clamps a timestep into a sane range.
///
/// A tab that was backgrounded for ten seconds hands back a `dt` of 10.0; one
/// step that large blows any explicit integrator apart, so a long gap is
/// treated as a single slow frame instead.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 480.0, 1.0 / 20.0)
}

/// Expected packed buffer sizes, exported for the JS side's assertions.
pub const HAND_BUFFER_LEN: usize = HAND_BUFFER;
/// Expected pose buffer length.
pub const POSE_BUFFER_LEN: usize = POSE_STRIDE;
