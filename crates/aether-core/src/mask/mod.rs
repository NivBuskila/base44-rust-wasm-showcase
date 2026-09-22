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
//! [`BodyMask::obstacle`] is 0 or 1 per cell while the body is tracked, and is
//! ramped down to exactly 0 while a lost body fades out.
//! [`BodyMask::edge_velocity`] is in grid cells per second, non-zero only in a
//! band around the silhouette edge. [`BodyMask::decay`] must fade the obstacle
//! away when tracking is lost, so losing the body does not freeze a permanent
//! wall into the simulation.
//!
//! ## Invariant the engine depends on
//!
//! `obstacle()` is non-empty **only** while `present()` is true, and the call
//! to [`BodyMask::decay`] that empties it is the same one that clears
//! `present`. [`crate::engine::Engine::step`] re-uploads the obstacle to the
//! solver only while the mask is dirty or `present()` holds, so any solid cell
//! left behind after `present` drops would be frozen into the fluid until the
//! next mask arrives — the invisible-wall bug this module exists to avoid.

use crate::field::{Grid, VecField};

use crate::math::{decay, length, lerp, smoothstep};

mod edge;
mod stage;
#[cfg(test)]
mod tests;

/// Level at which the blurred obstacle is re-thresholded, i.e. the erode half
/// of the morphological close.
///
/// Deliberately below 0.5. `Grid::blur` is a separable binomial, so for a
/// rectangle the blurred field factorises into its 1D profiles: with two
/// passes the last solid cell of an edge reads 11/16 = 0.69, the first empty
/// cell 5/16 = 0.31, and a corner 0.69^2 = 0.47. Cutting at 0.5 would shave
/// every corner off the silhouette; 0.35 sits in the gap left by 1-3 passes, so
/// straight edges and corners survive unmoved while an isolated cell — 0.14
/// after two passes — dies, which is the entire point of the close.
const CLOSE_LEVEL: f32 = 0.35;

/// Hard cap on `blur_passes`. The config crosses from JS and each pass is four
/// full grid sweeps, so a garbage value would stall the frame.
const MAX_BLUR_PASSES: usize = 8;

/// Gradient magnitude (per cell) at which the edge band starts and at which it
/// reaches full strength.
///
/// This is what confines the boundary push to a band: a real silhouette crosses
/// 0 -> 1 over a handful of cells, so `|grad|` there is at least ~0.1, while the
/// flat body interior and empty space sit at 0 plus mask noise. Gating on the
/// gradient rather than dividing by it also avoids the level-set formula's
/// blow-up where the field is nearly flat.
const EDGE_GRAD_MIN: f32 = 0.015;
const EDGE_GRAD_FULL: f32 = 0.06;

/// Temporal deltas below this are mask noise, not boundary motion.
const DELTA_EPS: f32 = 1e-4;

/// Ceiling on injected boundary speed, in grid cells per second.
///
/// `delta / dt` is unbounded as `dt` shrinks, and a segmentation that jumps
/// half a body width in one frame hands back an arbitrarily large rate. 600
/// cells/s is already more than two screen widths per second — past any real
/// arm — so the clamp only ever fires on a glitch, but without it one bad frame
/// injects a CFL-breaking spike that the solver takes seconds to dissipate.
const MAX_EDGE_SPEED: f32 = 600.0;

/// Solid fraction below which the silhouette is segmentation speckle, not a
/// body. ~0.5% of the frame is a blob roughly 14 cells on a side.
const MIN_COVERAGE: f32 = 0.005;

/// Solid fraction above which the mask is treated as a failed segmentation.
///
/// A mask that claims almost the whole frame is body is never a usable body: it
/// walls the fluid off the screen, and the pressure solve spends every
/// iteration on a domain with no interior. Dropping the frame keeps the last
/// good silhouette until `decay` retires it.
const MAX_COVERAGE: f32 = 0.85;

/// Timestep floor. `update_f32` is public and divides by `dt`.
const MIN_DT: f32 = 1.0 / 480.0;
/// Timestep ceiling. A gap longer than this is a stall, not a frame.
const MAX_DT: f32 = 1.0;
/// Stand-in for a non-finite `dt`.
const NOMINAL_DT: f32 = 1.0 / 60.0;

/// Fallback for a non-finite configured threshold.
const FALLBACK_THRESHOLD: f32 = 0.5;

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
    /// Fade envelope currently baked into `obstacle` and `edge`. 1 right after
    /// an accepted update, 0 once the body has been retired.
    fade: f32,
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
            fade: 0.0,
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
    ///
    /// This tracks `obstacle()`: it is 0 whenever that field is empty, and is
    /// attenuated by the fade envelope while a lost body is being retired.
    #[inline]
    pub fn coverage(&self) -> f32 {
        self.coverage
    }

    #[inline]
    pub fn present(&self) -> bool {
        self.present
    }

    /// Called every frame. Fades the obstacle out when no mask has arrived
    /// recently, so a lost body does not leave a wall behind.
    pub fn decay(&mut self, dt: f32) {
        let step = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };
        self.since_update = self.since_update.saturating_add_time(step);
        if self.fade <= 0.0 {
            return;
        }

        let env = fade_envelope(self.since_update, self.cfg.fade_seconds);
        if env >= self.fade {
            return;
        }
        if env <= 0.0 {
            // Exactly zero, and `present` drops in the same call: the engine
            // gets one final upload of an empty field and then stops asking.
            self.obstacle.zero();
            self.edge.zero();
            self.coverage = 0.0;
            self.present = false;
            self.fade = 0.0;
            return;
        }
        // Applied as a ratio against the envelope already baked into the
        // fields, so the product telescopes to `env` however the frames are
        // diced — the fade looks the same at 30 and 144 fps.
        let k = env / self.fade;
        self.obstacle.scale(k);
        self.edge.scale(k);
        self.coverage *= k;
        self.fade = env;
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
        self.fade = 0.0;
    }

    /// Seconds since the last mask arrived.
    #[inline]
    pub fn staleness(&self) -> f32 {
        self.since_update
    }
}

/// Clamps a mask timestep into a sane range.
///
/// The engine clamps its own `dt` before calling in, but these entry points are
/// public and the boundary velocity divides by `dt`: zero, negative and NaN all
/// have to be impossible by the time they reach the grid loops.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if dt.is_finite() {
        dt.clamp(MIN_DT, MAX_DT)
    } else {
        NOMINAL_DT
    }
}

/// Fraction of the previously smoothed confidence that survives `dt`.
///
/// `rate = ln2 / half_life` makes this exactly `2^(-dt / half_life)`, so two
/// half-steps compose into one full step and the silhouette settles at the same
/// rate at any frame rate. A non-positive or non-finite half-life means "no
/// smoothing" rather than "never update", because the latter is a config typo
/// that freezes body collision with no way back.
#[inline]
fn ema_keep(half_life: f32, dt: f32) -> f32 {
    if !half_life.is_finite() || half_life <= 0.0 {
        return 0.0;
    }
    decay(core::f32::consts::LN_2 / half_life, dt).clamp(0.0, 1.0)
}

/// Obstacle strength as a function of staleness: 1 while the mask is fresh, 0
/// once it is `fade_seconds` old.
///
/// Smoothstep rather than a straight ramp because the mask arrives at 30 Hz
/// against a 60 Hz render loop, so staleness is routinely a frame or two even
/// with perfect tracking. A linear ramp would shave ~10% off the obstacle in
/// that gap and the body would visibly pulse; the cubic is flat near zero and
/// costs 0.7%.
#[inline]
fn fade_envelope(stale: f32, fade_seconds: f32) -> f32 {
    if !stale.is_finite() {
        return 0.0;
    }
    1.0 - smoothstep(0.0, fade_seconds.max(0.0), stale)
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
