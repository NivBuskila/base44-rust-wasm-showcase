//! Turns a segmentation mask into something a fluid solver can collide with.
//!
//! MediaPipe hands back a soft confidence mask at the model's own resolution.
//! Feeding that to the solver directly produces a jittering, feathered blob
//! that leaks fluid, so this module:
//!
//! 1. resamples the mask onto the simulation grid,
//! 2. temporally smooths it (the mask flickers frame to frame),
//! 3. thresholds and blurs it into a crisp binary obstacle field,
//! 4. derives the **boundary velocity** from the mask's frame-to-frame
//!    difference, so a moving body pushes fluid instead of just blocking it —
//!    that is what makes waving your arm feel like it displaces smoke.
//!
//! ## Contract
//!
//! [`BodyMask::obstacle`] is 0 or 1 per cell. [`BodyMask::edge_velocity`] is in
//! grid cells per second, non-zero only in a band around the silhouette edge.
//! [`BodyMask::decay`] must fade the obstacle away when tracking is lost, so
//! losing the body does not freeze a permanent wall into the simulation.

use crate::field::{Grid, VecField};

/// Mask processing settings.
#[derive(Clone, Copy, Debug)]
pub struct MaskConfig {
    /// Confidence above which a cell counts as body.
    pub threshold: f32,
    /// Temporal EMA half-life for the raw mask, seconds.
    pub half_life: f32,
    /// Binomial blur passes applied to the thresholded obstacle.
    pub blur_passes: usize,
    /// Scale from mask delta to boundary velocity.
    pub push_gain: f32,
    /// Seconds for the obstacle to fade out once the body is lost.
    pub fade_seconds: f32,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            threshold: 0.55,
            half_life: 0.08,
            blur_passes: 2,
            push_gain: 26.0,
            fade_seconds: 0.35,
        }
    }
}

/// Body silhouette as an obstacle field plus its boundary motion.
pub struct BodyMask {
    w: usize,
    h: usize,
    cfg: MaskConfig,
    /// Latest resampled confidence.
    raw: Grid,
    /// Temporally smoothed confidence.
    smooth: Grid,
    /// Smoothed confidence from the previous update, for the delta.
    prev: Grid,
    /// Binary collision field handed to the fluid.
    obstacle: Grid,
    /// Velocity injected at the silhouette edge.
    edge: VecField,
    scratch: Grid,
    /// Staging grid at the incoming mask's own resolution.
    incoming: Option<Grid>,
    present: bool,
    coverage: f32,
    since_update: f32,
}

impl BodyMask {
    pub fn new(w: usize, h: usize, cfg: MaskConfig) -> Self {
        Self {
            w,
            h,
            cfg,
            raw: Grid::new(w, h),
            smooth: Grid::new(w, h),
            prev: Grid::new(w, h),
            obstacle: Grid::new(w, h),
            edge: VecField::new(w, h),
            scratch: Grid::new(w, h),
            incoming: None,
            present: false,
            coverage: 0.0,
            since_update: f32::MAX,
        }
    }

    #[inline]
    pub fn config(&self) -> &MaskConfig {
        &self.cfg
    }

    pub fn set_config(&mut self, cfg: MaskConfig) {
        self.cfg = cfg;
    }

    #[inline]
    pub fn obstacle(&self) -> &Grid {
        &self.obstacle
    }

    #[inline]
    pub fn confidence(&self) -> &Grid {
        &self.smooth
    }

    #[inline]
    pub fn edge_velocity(&self) -> &VecField {
        &self.edge
    }

    /// Fraction of the frame covered by the body, `[0, 1]`.
    #[inline]
    pub fn coverage(&self) -> f32 {
        self.coverage
    }

    #[inline]
    pub fn present(&self) -> bool {
        self.present
    }

    /// Lazily allocated staging grid matching the incoming mask resolution.
    fn staging(&mut self, sw: usize, sh: usize) -> &mut Grid {
        let needs_alloc = match &self.incoming {
            Some(g) => g.w != sw || g.h != sh,
            None => true,
        };
        if needs_alloc {
            self.incoming = Some(Grid::new(sw.max(2), sh.max(2)));
        }
        self.incoming.as_mut().unwrap()
    }

    /// Ingests a float confidence mask at an arbitrary resolution.
    ///
    /// `mirror` flips horizontally, matching the mirrored preview the user sees.
    ///
    /// AGENT-IMPLEMENT: the placeholder resamples and thresholds without
    /// temporal smoothing or edge velocity.
    pub fn update_f32(&mut self, src: &[f32], sw: usize, sh: usize, mirror: bool, dt: f32) {
        if sw < 2 || sh < 2 || src.len() < sw * sh {
            return;
        }
        let _ = dt;
        {
            let staging = self.staging(sw, sh);
            for y in 0..sh {
                for x in 0..sw {
                    let sx = if mirror { sw - 1 - x } else { x };
                    staging.data[y * sw + x] = src[y * sw + sx];
                }
            }
        }
        let staged = self.incoming.take().expect("staging allocated above");
        self.raw.resample_from(&staged);
        self.incoming = Some(staged);

        self.smooth.data.copy_from_slice(&self.raw.data);
        let t = self.cfg.threshold;
        for (o, &c) in self.obstacle.data.iter_mut().zip(&self.smooth.data) {
            *o = if c >= t { 1.0 } else { 0.0 };
        }
        self.coverage = self.obstacle.sum() / (self.w * self.h) as f32;
        self.present = self.coverage > 0.005;
        self.since_update = 0.0;
        let _ = (&mut self.prev, &mut self.edge, &mut self.scratch);
    }

    /// Ingests an 8-bit mask at an arbitrary resolution.
    pub fn update_u8(&mut self, src: &[u8], sw: usize, sh: usize, mirror: bool, dt: f32) {
        let mut tmp = Vec::with_capacity(src.len());
        tmp.extend(src.iter().map(|&v| v as f32 * (1.0 / 255.0)));
        self.update_f32(&tmp, sw, sh, mirror, dt);
    }

    /// Called every frame. Fades the obstacle out when no mask has arrived
    /// recently, so a lost body does not leave a wall behind.
    ///
    /// AGENT-IMPLEMENT: the placeholder only tracks elapsed time.
    pub fn decay(&mut self, dt: f32) {
        self.since_update = self.since_update.saturating_add_time(dt);
    }

    pub fn reset(&mut self) {
        self.raw.zero();
        self.smooth.zero();
        self.prev.zero();
        self.obstacle.zero();
        self.edge.zero();
        self.present = false;
        self.coverage = 0.0;
        self.since_update = f32::MAX;
    }

    /// Seconds since the last mask arrived.
    #[inline]
    pub fn staleness(&self) -> f32 {
        self.since_update
    }
}

/// Saturating float addition so `f32::MAX + dt` stays finite.
trait SaturatingAddTime {
    fn saturating_add_time(self, dt: f32) -> f32;
}

impl SaturatingAddTime for f32 {
    #[inline]
    fn saturating_add_time(self, dt: f32) -> f32 {
        if self > 1e30 {
            self
        } else {
            self + dt
        }
    }
}
