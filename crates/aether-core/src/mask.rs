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

    /// Lazily allocated staging grid matching the incoming mask resolution.
    fn staging(&mut self, sw: usize, sh: usize) -> &mut Grid {
        let (sw, sh) = (sw.max(2), sh.max(2));
        if self
            .incoming
            .as_ref()
            .is_some_and(|g| g.w != sw || g.h != sh)
        {
            self.incoming = None;
        }
        self.incoming.get_or_insert_with(|| Grid::new(sw, sh))
    }

    /// Copies an incoming mask into the staging grid, mirroring and clamping.
    /// Returns false if the dimensions do not describe the slice, in which case
    /// nothing has been touched.
    ///
    /// Generic over the sample type so the 8-bit path does not have to allocate
    /// a float copy of the mask every frame.
    fn stage<T: Copy>(
        &mut self,
        src: &[T],
        sw: usize,
        sh: usize,
        mirror: bool,
        load: fn(T) -> f32,
    ) -> bool {
        let Some(n) = sw.checked_mul(sh) else {
            return false;
        };
        if sw < 2 || sh < 2 || src.len() < n {
            return false;
        }
        let staging = self.staging(sw, sh);
        for y in 0..sh {
            let row = y * sw;
            for x in 0..sw {
                let sx = if mirror { sw - 1 - x } else { x };
                // `max` then `min` maps NaN to 0 branchlessly, the trick
                // `Grid::sample` documents — and the reason this is not
                // `clamp`, which propagates NaN. A NaN confidence otherwise
                // survives the resample and poisons the obstacle field, and
                // from there the velocity field, permanently.
                #[allow(clippy::manual_clamp)]
                let v = load(src[row + sx]).max(0.0).min(1.0);
                staging.data[row + x] = v;
            }
        }
        true
    }

    /// Ingests a float confidence mask at an arbitrary resolution.
    ///
    /// `mirror` flips horizontally, matching the mirrored preview the user sees.
    /// The mask covers the whole camera frame, so it is stretched across the
    /// whole grid; the aspect ratios differ and correcting for that would crop
    /// the body instead.
    pub fn update_f32(&mut self, src: &[f32], sw: usize, sh: usize, mirror: bool, dt: f32) {
        if !self.stage(src, sw, sh, mirror, |v| v) {
            return;
        }
        self.process(sane_dt(dt));
    }

    /// Ingests an 8-bit mask at an arbitrary resolution.
    pub fn update_u8(&mut self, src: &[u8], sw: usize, sh: usize, mirror: bool, dt: f32) {
        if !self.stage(src, sw, sh, mirror, |v| v as f32 * (1.0 / 255.0)) {
            return;
        }
        self.process(sane_dt(dt));
    }

    /// The pipeline proper, from the filled staging grid onwards.
    fn process(&mut self, dt: f32) {
        // Moved out and back so `raw` can be borrowed mutably while the staging
        // grid is read.
        if let Some(staged) = self.incoming.take() {
            self.raw.resample_from(&staged);
            self.incoming = Some(staged);
        }

        // A NaN threshold makes every comparison false, which silently disables
        // body collision; a garbage HUD value should degrade, not disappear.
        let level = if self.cfg.threshold.is_finite() {
            self.cfg.threshold.clamp(0.0, 1.0)
        } else {
            FALLBACK_THRESHOLD
        };

        let cells = self.raw.len().max(1);
        let raw_solid = self.raw.data.iter().filter(|&&c| c >= level).count() as f32 / cells as f32;
        if raw_solid > MAX_COVERAGE {
            // Checked before anything is committed, so a failed segmentation
            // cannot even pollute the temporal history it would take several
            // frames to walk back out of.
            return;
        }

        // A body that has just appeared has to engage on the first frame. The
        // EMA exists to stop an *established* silhouette from boiling; ramping
        // in from zero instead means the mask cannot cross the threshold at all
        // until a half-life has passed, which reads as input lag. Snapping
        // `prev` as well keeps the arrival silent: a 0 -> 1 step is a full
        // amplitude delta, and pushing on it would slam the fluid every time
        // tracking re-acquires.
        let cold = !self.present && raw_solid >= MIN_COVERAGE;
        if cold {
            self.smooth.data.copy_from_slice(&self.raw.data);
            self.prev.data.copy_from_slice(&self.raw.data);
        } else {
            let keep = ema_keep(self.cfg.half_life, dt);
            for (s, &r) in self.smooth.data.iter_mut().zip(&self.raw.data) {
                *s = lerp(r, *s, keep);
            }
        }

        for (o, &c) in self.obstacle.data.iter_mut().zip(&self.smooth.data) {
            *o = if c >= level { 1.0 } else { 0.0 };
        }
        // Blur-then-rethreshold is a cheap morphological close. A bare
        // threshold leaves single-cell speckle all over the silhouette, and
        // speckle in the obstacle field makes the pressure solve ring.
        let passes = self.cfg.blur_passes.min(MAX_BLUR_PASSES);
        if passes > 0 {
            self.obstacle.blur(&mut self.scratch, passes);
            for o in &mut self.obstacle.data {
                *o = if *o >= CLOSE_LEVEL { 1.0 } else { 0.0 };
            }
        }

        let solid = self.obstacle.data.iter().filter(|&&o| o >= 0.5).count();
        self.coverage = solid as f32 / cells as f32;
        self.present = self.coverage >= MIN_COVERAGE;

        self.rebuild_edge(dt);
        self.prev.data.copy_from_slice(&self.smooth.data);

        if !self.present {
            // Not just "no body": the field is cleared outright. See the
            // invariant in the module docs — solid cells surviving with
            // `present == false` get frozen into the solver.
            self.obstacle.zero();
            self.edge.zero();
            self.coverage = 0.0;
        }
        self.since_update = 0.0;
        self.fade = if self.present { 1.0 } else { 0.0 };
    }

    /// Rebuilds the boundary velocity from the temporal delta of `smooth`.
    ///
    /// The mask is a level set carried by the body, so it obeys
    /// `dm/dt + w . grad(m) = 0`: the boundary moves along `-sign(dm/dt) * n`
    /// with `n = grad(m) / |grad(m)|`. That gets the sign right at both ends of
    /// a silhouette — at the leading edge the mask is advancing into cells
    /// where `grad(m)` points backwards, at the trailing edge it is receding
    /// out of cells where `grad(m)` points forwards, and both produce a push
    /// along the direction of travel.
    ///
    /// The magnitude is the temporal rate `delta / dt` scaled by `push_gain`,
    /// not the level-set's own `|dm/dt| / |grad(m)|`: that quotient is exact for
    /// a rigid translation but divides by a gradient that vanishes two cells
    /// away from the edge, where the numerator is only *nearly* zero.
    fn rebuild_edge(&mut self, dt: f32) {
        let gain = if self.cfg.push_gain.is_finite() {
            self.cfg.push_gain
        } else {
            0.0
        };
        let rate_scale = gain / dt;

        for y in 0..self.h {
            let row = y * self.w;
            for x in 0..self.w {
                let i = row + x;
                let delta = self.smooth.data[i] - self.prev.data[i];
                let mut u = 0.0;
                let mut v = 0.0;
                // The overwhelming majority of cells are flat interior or empty
                // space, so this early-out is what keeps the pass cheap.
                if delta.abs() > DELTA_EPS {
                    let (gx, gy) = self.smooth.gradient(x, y);
                    let gmag = length(gx, gy);
                    let band = smoothstep(EDGE_GRAD_MIN, EDGE_GRAD_FULL, gmag);
                    if band > 0.0 {
                        // `band > 0` implies `gmag > EDGE_GRAD_MIN`, so the
                        // normalisation below cannot divide by zero.
                        let speed =
                            (-delta * rate_scale * band).clamp(-MAX_EDGE_SPEED, MAX_EDGE_SPEED);
                        u = speed * gx / gmag;
                        v = speed * gy / gmag;
                    }
                }
                self.edge.u.data[i] = u;
                self.edge.v.data[i] = v;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Grid the tests run `BodyMask` on, deliberately **not**
    /// `config::W/H`.
    ///
    /// `BodyMask::new` takes its dimensions explicitly, so a unit test for it
    /// has no business depending on the app's current display resolution — and
    /// these tests pick their cells for a reason (clear of the walls, straddling
    /// a specific edge). Tying them to a tunable meant that changing the fluid
    /// grid for the frame budget turned every carefully chosen coordinate into
    /// either an out-of-bounds index or a different part of the frame.
    const W: usize = 256;
    const H: usize = 144;

    const CELLS: usize = W * H;
    const DT30: f32 = 1.0 / 30.0;

    /// Config with the temporal EMA disabled, so one update lands the raw mask
    /// exactly and a test's arithmetic stays analytic.
    fn sharp() -> MaskConfig {
        MaskConfig {
            half_life: 0.0,
            ..MaskConfig::default()
        }
    }

    fn body(cfg: MaskConfig) -> BodyMask {
        BodyMask::new(W, H, cfg)
    }

    /// A mask at grid resolution with a filled rectangle. Same resolution means
    /// the resample is a copy, so cell indices are directly comparable.
    ///
    /// Coordinates are clamped: these tests are written against grid fractions
    /// (see [`fx`] / [`fy`]) rather than absolute cells, and a clamp here means
    /// a future resolution change degrades a rectangle instead of panicking
    /// with an out-of-bounds index.
    fn rect_mask(x0: usize, x1: usize, y0: usize, y1: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; CELLS];
        for y in y0..y1.min(H) {
            for x in x0..x1.min(W) {
                m[y * W + x] = 1.0;
            }
        }
        m
    }

    /// Vertical grid coordinate from a fraction of the frame height, for the
    /// tests whose geometry is naturally proportional (a mask that covers most
    /// of the frame) rather than a specific cell.
    fn fy(frac: f32) -> usize {
        ((H as f32 * frac) as usize).min(H - 1)
    }

    /// The standard test silhouette: a solid block left of centre, well clear
    /// of every domain wall so border effects cannot be mistaken for the
    /// behaviour under test.
    fn body_rect() -> Vec<f32> {
        rect_mask(100, 140, 40, 100)
    }

    /// [`body_rect`] translated `cells` to the right, for boundary-motion tests.
    fn body_rect_shifted(cells: usize) -> Vec<f32> {
        rect_mask(100 + cells, 140 + cells, 40, 100)
    }

    /// A point inside [`body_rect`], safely away from its edges.
    fn body_interior() -> (usize, usize) {
        (120, 70)
    }

    fn all_finite(m: &BodyMask) -> bool {
        m.obstacle()
            .data
            .iter()
            .chain(m.confidence().data.iter())
            .chain(m.edge_velocity().u.data.iter())
            .chain(m.edge_velocity().v.data.iter())
            .all(|v| v.is_finite())
    }

    #[test]
    fn a_rectangle_becomes_the_same_rectangle_of_solid_cells() {
        let mut m = body(sharp());
        m.update_f32(&body_rect(), W, H, false, DT30);

        for y in 0..H {
            for x in 0..W {
                let inside = (100..140).contains(&x) && (40..100).contains(&y);
                let want = if inside { 1.0 } else { 0.0 };
                assert_eq!(m.obstacle().get(x, y), want, "cell {x},{y}");
            }
        }
        // The close must not erode the shape, corners included.
        let expect = (40 * 60) as f32 / CELLS as f32;
        assert!(
            (m.coverage() - expect).abs() < 1e-6,
            "coverage {} vs {expect}",
            m.coverage()
        );
        assert!(m.present());
    }

    #[test]
    fn a_smaller_mask_is_stretched_across_the_whole_grid() {
        // The mask covers the camera frame, and the frame is the screen, so a
        // block over the middle half of a square mask must land over the middle
        // half of a 16:9 grid in *both* axes.
        let (mw, mh) = (64usize, 64usize);
        let mut src = vec![0.0f32; mw * mh];
        for y in 16..48 {
            for x in 16..48 {
                src[y * mw + x] = 1.0;
            }
        }
        let mut m = body(sharp());
        m.update_f32(&src, mw, mh, false, DT30);

        // The 0.5 crossing sits at mask coordinate 15.5 and 47.5, i.e. grid
        // x = 15.5 * 255/63 = 62.7 .. 192.2 and y = 35.2 .. 107.8.
        assert_eq!(m.obstacle().get(70, 70), 1.0);
        assert_eq!(m.obstacle().get(185, 70), 1.0);
        assert_eq!(m.obstacle().get(56, 70), 0.0);
        assert_eq!(m.obstacle().get(198, 70), 0.0);
        assert_eq!(m.obstacle().get(128, 30), 0.0, "block leaked upwards");
        assert_eq!(m.obstacle().get(128, 114), 0.0, "block leaked downwards");
        // ~130 x ~72 cells of 256 x 144.
        assert!(
            (0.23..0.28).contains(&m.coverage()),
            "coverage {}",
            m.coverage()
        );
    }

    #[test]
    fn mirror_flips_the_silhouette_horizontally() {
        let (mw, mh) = (64usize, 64usize);
        let mut src = vec![0.0f32; mw * mh];
        for y in 0..mh {
            for x in 0..16 {
                src[y * mw + x] = 1.0;
            }
        }
        let mut plain = body(sharp());
        plain.update_f32(&src, mw, mh, false, DT30);
        let mut flipped = body(sharp());
        flipped.update_f32(&src, mw, mh, true, DT30);

        let mid = H / 2;
        let right = W - 11;
        assert_eq!(plain.obstacle().get(10, mid), 1.0);
        assert_eq!(plain.obstacle().get(right, mid), 0.0);
        assert_eq!(
            flipped.obstacle().get(10, mid),
            0.0,
            "mirrored mask stayed on the left"
        );
        assert_eq!(
            flipped.obstacle().get(right, mid),
            1.0,
            "mirrored mask never reached the right"
        );
        assert!((plain.coverage() - flipped.coverage()).abs() < 1e-6);
    }

    #[test]
    fn temporal_smoothing_is_framerate_independent() {
        let mask = rect_mask(80, 176, 30, 120);
        let empty = vec![0.0f32; CELLS];
        let mut coarse = body(MaskConfig::default());
        let mut fine = body(MaskConfig::default());
        coarse.update_f32(&mask, W, H, false, DT30);
        fine.update_f32(&mask, W, H, false, DT30);

        // One 100 ms decay against ten 10 ms decays of the same input.
        coarse.update_f32(&empty, W, H, false, 0.1);
        for _ in 0..10 {
            fine.update_f32(&empty, W, H, false, 0.01);
        }

        let worst = coarse
            .confidence()
            .data
            .iter()
            .zip(&fine.confidence().data)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4, "EMA differs by {worst} between frame rates");
        // 2^(-0.1 / 0.08) = 0.42; if it had not decayed the test proves nothing.
        let peak = coarse.confidence().max_abs();
        assert!((0.40..0.44).contains(&peak), "peak confidence {peak}");
    }

    #[test]
    fn single_cell_speckle_does_not_survive_the_close() {
        let speckle = [(10usize, 10usize), (200, 20), (30, 120), (240, 130)];
        let mut src = body_rect();
        for &(x, y) in &speckle {
            src[y * W + x] = 1.0;
        }
        let mut m = body(sharp());
        m.update_f32(&src, W, H, false, DT30);

        for &(x, y) in &speckle {
            assert_eq!(
                m.obstacle().get(x, y),
                0.0,
                "speckle at {x},{y} reached the solver"
            );
        }
        let (bx, by) = body_interior();
        assert_eq!(m.obstacle().get(bx, by), 1.0, "the body was eroded away");
        let expect = (40 * 60) as f32 / CELLS as f32;
        assert!((m.coverage() - expect).abs() < 1e-6);
    }

    #[test]
    fn a_silhouette_moving_right_pushes_right_at_both_edges() {
        let cfg = MaskConfig {
            half_life: 0.0,
            push_gain: 1.0,
            ..MaskConfig::default()
        };
        let mut m = body(cfg);
        m.update_f32(&body_rect(), W, H, false, DT30);
        m.update_f32(&body_rect_shifted(2), W, H, false, DT30);

        let e = m.edge_velocity();
        let row = 70;
        // Leading edge: one mask unit of advance in dt seconds, gain 1.
        assert!(
            (e.u.get(141, row) - 30.0).abs() < 1e-3,
            "leading edge {} (expected +30 cells/s)",
            e.u.get(141, row)
        );
        assert!(
            e.u.get(101, row) > 0.0,
            "trailing edge pushed {} (expected right)",
            e.u.get(101, row)
        );
        // A horizontal edge has no vertical component at mid height.
        assert!(e.v.get(141, row).abs() < 1e-6);
        // Zero deep inside the body and out in empty space.
        assert_eq!(e.u.get(120, row), 0.0, "pushed deep inside the body");
        assert_eq!(e.v.get(120, row), 0.0);
        assert_eq!(e.u.get(20, row), 0.0, "pushed far from the silhouette");
        assert_eq!(e.u.get(220, row), 0.0);
        // And the band is narrow: nothing three cells past the new edge.
        assert_eq!(e.u.get(145, row), 0.0);
        assert!(
            e.u.data.iter().sum::<f32>() > 0.0,
            "net push is not rightward"
        );
    }

    #[test]
    fn a_stationary_silhouette_produces_no_edge_velocity() {
        for cfg in [sharp(), MaskConfig::default()] {
            let mut m = body(cfg);
            let mask = body_rect();
            for _ in 0..8 {
                m.update_f32(&mask, W, H, false, DT30);
            }
            assert!(
                m.edge_velocity().max_speed() < 1e-3,
                "still body pushed at {} cells/s",
                m.edge_velocity().max_speed()
            );
            assert!(m.present());
        }
    }

    #[test]
    fn edge_velocity_is_per_second_not_per_frame() {
        let cfg = MaskConfig {
            half_life: 0.0,
            push_gain: 1.0,
            ..MaskConfig::default()
        };
        let mut slow = body(cfg);
        let mut fast = body(cfg);
        for (m, dt) in [(&mut slow, DT30), (&mut fast, DT30 * 0.5)] {
            m.update_f32(&body_rect(), W, H, false, dt);
            m.update_f32(&body_rect_shifted(2), W, H, false, dt);
        }
        let (a, b) = (
            slow.edge_velocity().u.get(141, 70),
            fast.edge_velocity().u.get(141, 70),
        );
        assert!(a > 0.0);
        assert!(
            (b - 2.0 * a).abs() < 1e-3,
            "same displacement in half the time gave {b}, expected 2 * {a}"
        );
    }

    #[test]
    fn edge_velocity_is_bounded_even_for_an_absurd_config() {
        let mut m = body(MaskConfig {
            half_life: 0.0,
            push_gain: 1e6,
            ..MaskConfig::default()
        });
        m.update_f32(&body_rect(), W, H, false, 1e-9);
        m.update_f32(&body_rect_shifted(20), W, H, false, 1e-9);
        assert!(all_finite(&m));
        assert!(
            m.edge_velocity().max_speed() <= MAX_EDGE_SPEED + 1e-3,
            "unclamped boundary speed {}",
            m.edge_velocity().max_speed()
        );
    }

    #[test]
    fn a_lost_body_fades_out_completely_and_clears_present() {
        let cfg = MaskConfig::default();
        let mut m = body(cfg);
        m.update_f32(&body_rect(), W, H, false, DT30);
        m.update_f32(&body_rect_shifted(2), W, H, false, DT30);
        let full = m.obstacle().sum();
        assert!(full > 0.0);
        assert!(m.edge_velocity().max_speed() > 0.0);

        // Halfway through the fade the wall is still there, weaker, and still
        // reported as present so the engine keeps re-uploading it.
        let half_steps = (0.5 * cfg.fade_seconds / (1.0 / 60.0)) as usize;
        for _ in 0..half_steps {
            m.decay(1.0 / 60.0);
        }
        assert!(m.present(), "present dropped before the field was empty");
        let mid = m.obstacle().sum();
        assert!(mid < full && mid > 0.0, "obstacle {mid} vs {full}");
        assert!(m.coverage() > 0.0);

        while m.staleness() < cfg.fade_seconds {
            m.decay(1.0 / 60.0);
        }
        assert_eq!(m.obstacle().sum(), 0.0, "a wall outlived the fade");
        assert_eq!(m.edge_velocity().max_speed(), 0.0);
        assert_eq!(m.coverage(), 0.0);
        assert!(!m.present());
        // And it stays gone.
        for _ in 0..600 {
            m.decay(1.0 / 60.0);
        }
        assert!(all_finite(&m) && m.staleness().is_finite());
    }

    #[test]
    fn the_fade_is_framerate_independent() {
        let mask = body_rect();
        let mut coarse = body(MaskConfig::default());
        let mut fine = body(MaskConfig::default());
        coarse.update_f32(&mask, W, H, false, DT30);
        fine.update_f32(&mask, W, H, false, DT30);
        for _ in 0..12 {
            coarse.decay(1.0 / 60.0);
        }
        for _ in 0..48 {
            fine.decay(1.0 / 240.0);
        }
        let (a, b) = (coarse.obstacle().sum(), fine.obstacle().sum());
        assert!(a > 0.0);
        assert!((a - b).abs() < 1e-3 * a, "fade diverged: {a} vs {b}");
    }

    #[test]
    fn a_30hz_mask_against_a_60hz_loop_barely_dips() {
        // Two render frames of staleness is the normal case, not a lost body:
        // a linear fade would pulse the obstacle by ~10% every mask period.
        let mask = body_rect();
        let mut m = body(MaskConfig::default());
        m.update_f32(&mask, W, H, false, DT30);
        let full = m.obstacle().sum();
        m.decay(1.0 / 60.0);
        m.decay(1.0 / 60.0);
        assert!(
            m.obstacle().sum() > 0.97 * full,
            "obstacle dipped to {} of {full} between mask frames",
            m.obstacle().sum()
        );
        m.update_f32(&mask, W, H, false, DT30);
        assert!((m.obstacle().sum() - full).abs() < 1e-6, "not restored");
    }

    #[test]
    fn an_almost_solid_mask_is_rejected() {
        let solid = vec![1.0f32; CELLS];
        let mut fresh = body(sharp());
        fresh.update_f32(&solid, W, H, false, DT30);
        assert_eq!(
            fresh.obstacle().sum(),
            0.0,
            "segmentation walled off the screen"
        );
        assert_eq!(fresh.coverage(), 0.0);
        assert!(!fresh.present());

        // And a bogus frame must not destroy a good silhouette.
        let mut m = body(sharp());
        m.update_f32(&body_rect(), W, H, false, DT30);
        let (good, cov) = (m.obstacle().sum(), m.coverage());
        m.update_f32(&solid, W, H, false, DT30);
        assert!(
            (m.obstacle().sum() - good).abs() < 1e-6,
            "bogus mask overwrote a good one"
        );
        assert!((m.coverage() - cov).abs() < 1e-6);

        // Just under the limit is a legitimate close-up body.
        let mut big = body(sharp());
        let rows = fy(0.8);
        big.update_f32(&rect_mask(0, W, 0, rows), W, H, false, DT30);
        assert!(big.present());
        let expected = rows as f32 / H as f32;
        assert!(
            (big.coverage() - expected).abs() < 1e-3,
            "coverage {} expected {expected}",
            big.coverage()
        );
    }

    #[test]
    fn a_tiny_blob_is_noise_rather_than_a_body() {
        let mut m = body(sharp());
        m.update_f32(&rect_mask(100, 110, 40, 50), W, H, false, DT30);
        // 100 cells of 36864 is 0.27%, below the noise floor: the field has to
        // be empty, not merely "not present" (see the module invariant).
        assert!(!m.present());
        assert_eq!(m.obstacle().sum(), 0.0);
        assert_eq!(m.coverage(), 0.0);
        assert_eq!(m.edge_velocity().max_speed(), 0.0);
    }

    #[test]
    fn degenerate_input_never_panics_and_never_writes_nan() {
        let mut m = body(MaskConfig::default());
        // Established body first, so there is state to corrupt.
        m.update_f32(&body_rect(), W, H, false, DT30);
        let good = m.obstacle().sum();

        m.update_f32(&[], 0, 0, false, DT30);
        m.update_f32(&[], 256, 144, false, DT30);
        m.update_f32(&[1.0], 1, 1, true, 0.0);
        m.update_f32(&[1.0, 0.0], 8, 8, false, f32::NAN);
        m.update_f32(&[0.5; 16], 4, 4, false, -5.0);
        m.update_f32(&rect_mask(0, 4, 0, 4), usize::MAX, usize::MAX, false, DT30);
        assert!(
            (m.obstacle().sum() - good).abs() < 1e-6,
            "an unusable mask was accepted"
        );

        // Garbage confidences, mixed with a real block.
        let mut junk = body_rect();
        for (i, v) in junk.iter_mut().enumerate() {
            match i % 5 {
                0 if *v == 0.0 => *v = f32::NAN,
                1 if *v == 0.0 => *v = f32::INFINITY,
                2 if *v == 0.0 => *v = -1e9,
                _ => {}
            }
        }
        m.update_f32(&junk, W, H, false, 1e9);
        assert!(all_finite(&m), "NaN reached the obstacle or velocity field");
        m.decay(f32::NAN);
        m.decay(-1.0);
        m.decay(1e9);
        assert!(all_finite(&m));
        assert!(m.staleness().is_finite());

        // A garbage config must not produce NaN either.
        m.set_config(MaskConfig {
            threshold: f32::NAN,
            half_life: f32::NAN,
            blur_passes: usize::MAX,
            push_gain: f32::INFINITY,
            fade_seconds: f32::NAN,
        });
        m.update_f32(&body_rect(), W, H, false, DT30);
        m.decay(1.0 / 60.0);
        assert!(all_finite(&m));
    }

    #[test]
    fn a_mask_resolution_change_is_handled() {
        let mut m = body(sharp());
        for &(mw, mh) in &[(64usize, 64usize), (32, 48), (256, 144), (17, 9)] {
            let mut src = vec![0.0f32; mw * mh];
            for y in (mh / 4)..(3 * mh / 4) {
                for x in (mw / 4)..(3 * mw / 4) {
                    src[y * mw + x] = 1.0;
                }
            }
            m.update_f32(&src, mw, mh, false, DT30);
            assert!(m.present(), "{mw}x{mh} mask produced no body");
            assert!(
                (0.15..0.35).contains(&m.coverage()),
                "{mw}x{mh} coverage {}",
                m.coverage()
            );
            assert!(all_finite(&m));
        }
    }

    #[test]
    fn the_u8_path_matches_the_float_path() {
        let float = body_rect();
        let bytes: Vec<u8> = float.iter().map(|&v| (v * 255.0) as u8).collect();
        let mut a = body(sharp());
        let mut b = body(sharp());
        a.update_f32(&float, W, H, false, DT30);
        b.update_u8(&bytes, W, H, false, DT30);
        assert_eq!(a.obstacle().data, b.obstacle().data);
        assert_eq!(a.coverage(), b.coverage());
    }

    #[test]
    fn reset_forgets_everything() {
        let mut m = body(MaskConfig::default());
        m.update_f32(&body_rect(), W, H, false, DT30);
        m.update_f32(&body_rect_shifted(4), W, H, false, DT30);
        assert!(m.present() && m.obstacle().sum() > 0.0);
        m.reset();
        assert!(!m.present());
        assert_eq!(m.obstacle().sum(), 0.0);
        assert_eq!(m.coverage(), 0.0);
        assert_eq!(m.edge_velocity().max_speed(), 0.0);
        assert_eq!(m.confidence().sum(), 0.0);
        // A fade started before the reset must not resurrect anything.
        m.decay(1.0 / 60.0);
        assert_eq!(m.obstacle().sum(), 0.0);
        // And the next body still engages on its first frame.
        m.update_f32(&body_rect(), W, H, false, DT30);
        assert!(m.present());
    }

    #[test]
    fn ema_keep_composes_and_degrades_safely() {
        let one = ema_keep(0.08, 0.1);
        let mut many = 1.0;
        for _ in 0..10 {
            many *= ema_keep(0.08, 0.01);
        }
        assert!((one - many).abs() < 1e-6, "{one} vs {many}");
        // Exactly one half-life keeps half.
        assert!((ema_keep(0.08, 0.08) - 0.5).abs() < 1e-6);
        // Garbage means "no smoothing", never "never update".
        assert_eq!(ema_keep(0.0, 0.016), 0.0);
        assert_eq!(ema_keep(-1.0, 0.016), 0.0);
        assert_eq!(ema_keep(f32::NAN, 0.016), 0.0);
        assert_eq!(ema_keep(f32::INFINITY, 0.016), 0.0);
    }

    #[test]
    fn fade_envelope_spans_exactly_fade_seconds() {
        assert_eq!(fade_envelope(0.0, 0.35), 1.0);
        assert_eq!(fade_envelope(0.35, 0.35), 0.0);
        assert_eq!(fade_envelope(10.0, 0.35), 0.0);
        assert_eq!(fade_envelope(f32::MAX, 0.35), 0.0);
        assert_eq!(fade_envelope(f32::NAN, 0.35), 0.0);
        // Zero and NaN fade windows retire the body at once rather than never.
        assert_eq!(fade_envelope(1e-6, 0.0), 0.0);
        assert_eq!(fade_envelope(1e-6, f32::NAN), 0.0);
        let mut prev = 1.0;
        for i in 0..=35 {
            let v = fade_envelope(i as f32 * 0.01, 0.35);
            assert!(
                v <= prev && (0.0..=1.0).contains(&v),
                "not monotonic at {i}"
            );
            prev = v;
        }
    }

    #[test]
    fn sane_dt_is_finite_and_positive() {
        assert_eq!(sane_dt(f32::NAN), NOMINAL_DT);
        assert_eq!(sane_dt(f32::INFINITY), NOMINAL_DT);
        assert_eq!(sane_dt(0.0), MIN_DT);
        assert_eq!(sane_dt(-1.0), MIN_DT);
        assert_eq!(sane_dt(1e9), MAX_DT);
        assert_eq!(sane_dt(DT30), DT30);
    }
}
