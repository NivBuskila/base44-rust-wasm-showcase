//! The HUD snapshot and the frame-end pass that fills it.
//!
//! Two costs are traded off here. Everything callers treat as *state* —
//! `hands_present`, `mask_present`, the frame counter — is an O(1) read and is
//! refreshed every frame, because a stale value there reads as a bug. The three
//! fluid figures each walk the whole velocity field, so they refresh every
//! [`STATS_EVERY`](super::STATS_EVERY) frames; the HUD repaints at 10 Hz and
//! cannot tell.

use super::{Engine, STATS_EVERY};

/// A snapshot of engine state for the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub fluid_energy: f32,
    pub fluid_max_speed: f32,
    pub fluid_divergence: f32,
    pub motion_energy: f32,
    pub flow_dx: f32,
    pub flow_dy: f32,
    pub particles_alive: u32,
    pub mask_coverage: f32,
    pub mask_present: bool,
    pub hands_present: u32,
    pub pose_present: bool,
    pub time_scale: f32,
    pub bursts: u32,
    /// Non-finite cells repaired this frame. Should always be 0.
    pub nan_repairs: u32,
    pub frame: u32,
    pub ambient: bool,
}

impl Engine {
    // ----------------------------------------------------------------- stats

    pub(super) fn collect_stats(&mut self, repairs: usize, ambient: bool) {
        let (dx, dy) = self.flow.dominant_motion();
        let hands = self.tracker.hands();

        // Everything here is an O(1) read, so it stays live every frame —
        // callers treat `hands_present` and friends as state, not as telemetry.
        self.stats.motion_energy = self.flow.motion_energy();
        self.stats.flow_dx = dx;
        self.stats.flow_dy = dy;
        self.stats.particles_alive = self.particles.alive() as u32;
        self.stats.mask_coverage = self.body.coverage();
        self.stats.mask_present = self.body.present();
        self.stats.hands_present = hands.iter().filter(|h| h.present).count() as u32;
        self.stats.pose_present = self.tracker.pose().present;
        self.stats.time_scale = self.params.time_scale * self.report.time_scale;
        self.stats.bursts = self.report.bursts;
        self.stats.nan_repairs = repairs as u32;
        self.stats.frame = self.frame;
        self.stats.ambient = ambient;

        // These three each walk the whole velocity field. The HUD repaints at
        // 10 Hz, so recomputing them at 60 Hz burns grid passes to produce
        // numbers nobody reads.
        if self.frame.is_multiple_of(STATS_EVERY) {
            self.stats.fluid_energy = self.fluid.energy();
            self.stats.fluid_max_speed = self.fluid.max_speed();
            self.stats.fluid_divergence = self.fluid.last_divergence();
        }
    }

    #[inline]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
}
