//! Incompressible 2-D fluid solver (Stam, *Stable Fluids*, SIGGRAPH 1999) with
//! vorticity confinement and interior obstacles.
//!
//! ## Contract
//!
//! One [`Fluid::step`] performs, in order:
//! 1. **Advection** — semi-Lagrangian backtrace of velocity and of the three
//!    dye channels through the velocity field.
//! 2. **Viscous diffusion** — Jacobi relaxation, skipped when viscosity is 0.
//! 3. **Vorticity confinement** — re-injects the curl that numerical
//!    dissipation eats, which is what makes the smoke look alive.
//! 4. **Projection** — divergence, Jacobi pressure solve, gradient subtract,
//!    leaving the field (numerically) divergence free.
//! 5. **Dissipation** — frame-rate independent exponential decay of velocity
//!    and dye.
//!
//! Obstacle cells (`obstacle >= 0.5`) must end the step with zero velocity, and
//! the domain border is a no-slip wall. Post-step the field must satisfy
//! `max |div u| << max |u|` in the fluid interior.
//!
//! ## Discretisation
//!
//! Velocity, pressure, dye and the obstacle mask all live on the same
//! [`crate::field::Grid`] lattice, one cell apart (`h = 1`). The projection
//! reads that storage the way a MAC grid is laid out: `u[i]` is the flux
//! through the face half a cell to the **right** of cell `i` and `v[i]` the
//! face half a cell **below** it. Divergence is therefore a backward
//! difference and the pressure gradient a forward difference, whose
//! composition is *exactly* the compact five-point Laplacian the Jacobi solve
//! inverts.
//!
//! That consistency is the whole reason for the half-cell reading. With
//! central differences on both sides the composed operator is the *wide*
//! Laplacian instead, which the five-point solve cannot see the difference
//! from: residual divergence then plateaus near 10% of `max |u|` however many
//! iterations are thrown at it, where the matched pair measures ~1% under the
//! engine's own drive. Half a cell at 256x144 is a fifth of a screen pixel, so
//! the rest of the engine goes on sampling velocity at lattice points.
//!
//! What is left is honest Jacobi under-convergence, not inconsistency: a
//! freshly splatted impulse carries smooth, grid-scale divergence, and Jacobi
//! needs O(radius^2) sweeps to see a feature that wide. Warm starting is what
//! makes 28 sweeps per frame enough in practice.
//!
//! Advection is MacCormack/BFECC error-corrected with an RK2 (midpoint)
//! backtrace and a local extremum limiter. Both halves matter: first-order
//! Euler backtracing turns the dye into mush within a second at the 200+
//! cells/s a swipe produces, and an *unlimited* MacCormack correction rings at
//! sharp dye edges and eventually blows up. The trace positions depend only on
//! the velocity field, so all five advected channels share one pair of maps.
//!
//! The trace is subdivided when it would cross more than
//! [`TRACE_CELLS_PER_STEP`] in one step. That is not polish: a single midpoint
//! step over an arc wider than the impulse carrying it lands in the still
//! fluid outside and returns a trace of length *zero*, which freezes the
//! impulse in place and makes the settled speed depend on the frame rate. See
//! [`TRACE_CELLS_PER_STEP`] for the measurements and the cost.
//!
//! Walls — the domain border and every obstacle cell — are collapsed into one
//! `solid` mask, which is the only thing the solver stages test. Solid cells
//! hold zero velocity and are excluded from every write, so the invariant
//! "solids are at rest" holds *between* stages instead of being patched up at
//! the end. Advection is the one exception: it backs away from the `obstacle`
//! mask only, because the border needs no backing away (the fetch clamps
//! there) and treating it as an obstacle would make the frame edge behave
//! differently depending on whether a body is in view.
//!
//! ## Cost
//!
//! Measured native release at 256x144 with the default parameters, mid-stir:
//! ~7 ms per step, split roughly trace build 2.6, advection of the five
//! channels 2.3, 28 pressure sweeps 1.3, everything else 0.9. The two Jacobi
//! kernels are the only loops with a hand-written wasm32 SIMD128 path; the
//! advection is gather-bound rather than arithmetic-bound, so the work there
//! went into halving the number of gathers instead (three passes per channel
//! down to two, with one shared trace map instead of five).

use crate::field::{Grid, VecField};
use crate::math::{decay, length, normalize};
use crate::par;

mod advect;
mod diffuse;
mod kernels;
mod project;
mod step;
#[cfg(test)]
mod tests;

use advect::{Advector, Reach};
use kernels::*;

/// Dye channel count (RGB).
pub const DYE_CHANNELS: usize = 3;

/// Jacobi sweeps used by the viscous diffusion solve.
const DIFFUSION_ITERS: usize = 20;

/// Viscosity at or below this leaves the field unchanged to within float noise,
/// so the diffusion stage is skipped entirely. The default viscosity (2e-5) is
/// well under it: diffusion costs nothing unless the HUD asks for it.
const VISCOSITY_EPSILON: f32 = 1e-4;

/// Ceiling on a cell's speed, in grid cells per second.
///
/// A hand push peaks around 40-60 here, so this is headroom rather than a
/// governor on ordinary motion; it exists only to stop the field ratcheting up
/// without bound while gestures keep feeding it.
///
/// In *cells* per second, so it tracks the grid: raising the resolution to
/// 256x144 raised this and [`MAX_CONFINE_IMPULSE`] by the same 4/3 the grid
/// grew by, leaving both ceilings at the screen-relative speed they were tuned
/// at.
const MAX_CELL_SPEED: f32 = 147.0;

/// Ceiling on the velocity change one confinement step may apply, in cells/s.
///
/// Confinement is an anti-dissipation term — it *adds* energy — so an
/// unbounded version turns one fast gesture into an explosion a few frames
/// later, on exactly the frames where `dt` also spikes.
const MAX_CONFINE_IMPULSE: f32 = 67.0;

/// Relaxation factor for the pressure sweep.
///
/// Plain (undamped) Jacobi on the five-point Laplacian has an eigenvalue of
/// exactly `-1` at the checkerboard mode, so that component of the error never
/// decays — it just flips sign every sweep, and with an even iteration count it
/// comes back untouched. Measured on a noisy field, 28 undamped sweeps leave
/// `max |div u|` at 19 and 200 of them still leave 3.7, where 28 *damped*
/// sweeps leave 2.6. Anything below 1 fixes it; 0.9 keeps nearly all of the
/// undamped convergence rate on the smooth modes while still knocking the
/// checkerboard down 0.8x per sweep, which matters when the HUD drops
/// `pressure_iters` low.
const PRESSURE_OMEGA: f32 = 0.9;

/// Fractions of an advection trace tried, in order, when the full backtrace
/// would fetch from inside a wall.
const TRACE_RETRIES: [f32; 3] = [0.5, 0.25, 0.1];

/// Cells one trace substep may cross before the trace is subdivided.
///
/// The backtrace is the only place this solver has a CFL condition, and a
/// *single* midpoint step hides it: it assumes the velocity is near-constant
/// over the whole arc, so once `dt * |u|` is wider than the impulse carrying
/// it the midpoint lands in the still fluid outside, the trace collapses to
/// (exactly) zero length and the impulse stops advecting at all. The next
/// frame's impulse then stacks onto a stationary spike instead of a moving
/// one, and the field climbs until dissipation catches it — which is a
/// frame-rate dependence, because the per-frame impulse grows with `dt`.
///
/// Measured with a 6000 cells/s^2 drive pinned at one point: one midpoint step
/// settles at 1817 cells/s at 30 fps against 350 at 240 fps; subdividing gives
/// 340 and 350. A span under one substep is bit-identical to the single step,
/// so a calm field pays nothing.
///
/// Two cells rather than one is the cost/accuracy line: the midpoint then sits
/// one cell upwind, still well inside the narrowest impulse the spell layer
/// makes (`spells::MIN_RADIUS` is 4), and each substep costs one more velocity
/// gather. That matters because the gather, not the arithmetic, is what the
/// trace build is made of: at the flow-drive benchmark's 980 cells/s, half the
/// grid spans 7+ cells per frame and the stage roughly triples (the whole
/// engine step goes 14.1 -> 18.8 ms native release). Dial these two constants
/// if that trade needs to move; both are safe to lower.
const TRACE_CELLS_PER_STEP: f32 = 2.0;

/// Ceiling on trace substeps, so one runaway cell cannot eat the frame. At the
/// cap the substep grows again and the trace degrades gracefully rather than
/// turning the trace build into an unbounded loop.
const MAX_TRACE_STEPS: usize = 4;

/// Bilinear weight below which a wall corner counts as not contributing.
///
/// Near zero on purpose. Its only job is to let a cell's *own* lattice point
/// pass the test when the body happens to be its neighbour — there the wall
/// corner has weight exactly zero — so that the rim of cells around a
/// silhouette can still advect instead of freezing. Anything larger is
/// tolerated bleed: at 5% a wall cell leaks a third of its dye into the fluid
/// beside it within eight frames.
const MIN_STENCIL_WEIGHT: f32 = 1e-3;

/// Extra dye dissipation rate, per second, in the fluid cells around the body.
///
/// The body is solid, so dye cannot pass through it — and the rim of cells
/// pressed against the silhouette is also where the advection trace runs out of
/// room and falls back to "stay put" ([`Fluid::trace_to_fluid`]), so dye that
/// arrives there stops moving *and* keeps being fed. On screen that is a thick
/// slab of colour welded to the face and shoulders. Fading it keeps the
/// collision — the flow still parts around the body — while the colour burns
/// off where it lands instead of piling up.
///
/// One full dye lifetime is `1 / dye_dissipation` ~ 0.4 s at the default; this
/// is several times faster, so the contact band never outlives the gesture.
const BODY_CONTACT_FADE: f32 = 22.0;

/// Half-width, in cells, of the faded band around the silhouette.
///
/// A single-cell rim is not enough: the pile-up is as wide as the impulse
/// driving it (`spells::MIN_RADIUS` is 4 cells), so fading only the cells that
/// literally touch the body leaves a bright edge sitting one cell out. Five
/// cells each way covers the stalled band and still leaves the rest of the
/// frame untouched.
const BODY_CONTACT_BAND: usize = 5;

/// A velocity + dye field on a fixed grid.
pub struct Fluid {
    w: usize,
    h: usize,
    vel: VecField,
    vel_tmp: VecField,
    dye: [Grid; DYE_CHANNELS],
    pressure: Grid,
    pressure_tmp: Grid,
    divergence: Grid,
    curl: Grid,
    obstacle: Grid,
    scratch: Grid,
    /// `obstacle` with the domain border forced solid. Every kernel tests this
    /// one array instead of re-deriving "am I on the border?" per cell.
    solid: Grid,
    /// Trace maps and workspace shared by all five advected channels.
    advector: Advector,
    /// Cells with a wall within reach, rebuilt per use: of any trace this step
    /// for the advection, of the contact band for the dye fade.
    reach: Reach,
    /// Whether any *interior* cell is solid, i.e. whether a body is in frame.
    has_obstacle: bool,
    /// Absolute maximum divergence measured at the end of the last step; the
    /// HUD shows it and the tests assert on it.
    last_divergence: f32,
}

impl Fluid {
    pub fn new(w: usize, h: usize) -> Self {
        let mut fluid = Self {
            w,
            h,
            vel: VecField::new(w, h),
            vel_tmp: VecField::new(w, h),
            dye: [Grid::new(w, h), Grid::new(w, h), Grid::new(w, h)],
            pressure: Grid::new(w, h),
            pressure_tmp: Grid::new(w, h),
            divergence: Grid::new(w, h),
            curl: Grid::new(w, h),
            obstacle: Grid::new(w, h),
            scratch: Grid::new(w, h),
            solid: Grid::new(w, h),
            advector: Advector::new(w * h),
            reach: Reach::new(w, h),
            has_obstacle: false,
            last_divergence: 0.0,
        };
        fluid.refresh_solid();
        fluid
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    /// Adds a velocity impulse, in grid units per second, around `(x, y)`.
    pub fn add_force(&mut self, x: f32, y: f32, fx: f32, fy: f32, radius: f32) {
        self.vel.splat(x, y, radius, fx, fy);
    }

    /// Injects coloured dye around `(x, y)`.
    pub fn add_dye(&mut self, x: f32, y: f32, rgb: [f32; 3], radius: f32) {
        for (channel, amount) in self.dye.iter_mut().zip(rgb) {
            channel.splat(x, y, radius, amount);
        }
    }

    /// Replaces the obstacle field. Values `>= 0.5` are solid.
    pub fn set_obstacle(&mut self, mask: &Grid) {
        self.obstacle.resample_from(mask);
        self.refresh_solid();
    }

    #[inline]
    pub fn obstacle(&self) -> &Grid {
        &self.obstacle
    }

    #[inline]
    pub fn velocity(&self) -> &VecField {
        &self.vel
    }

    #[inline]
    pub fn velocity_mut(&mut self) -> &mut VecField {
        &mut self.vel
    }

    #[inline]
    pub fn dye(&self) -> &[Grid; DYE_CHANNELS] {
        &self.dye
    }

    #[inline]
    pub fn curl(&self) -> &Grid {
        &self.curl
    }

    /// Curl (vorticity) sampled at a grid-space point; drives particle spin.
    pub fn curl_at(&self, x: f32, y: f32) -> f32 {
        self.curl.sample(x, y)
    }

    pub fn energy(&self) -> f32 {
        self.vel.energy()
    }

    pub fn max_speed(&self) -> f32 {
        self.vel.max_speed()
    }

    /// Absolute maximum divergence left over after the last projection.
    pub fn last_divergence(&self) -> f32 {
        self.last_divergence
    }

    /// Solver-side wall mask: obstacles plus the domain border.
    #[inline]
    pub fn solid(&self) -> &Grid {
        &self.solid
    }

    /// The pressure field left by the last projection. Warm-started, so it is
    /// also the next step's initial guess.
    #[inline]
    pub fn pressure(&self) -> &Grid {
        &self.pressure
    }

    pub fn reset(&mut self) {
        self.vel.zero();
        self.pressure.zero();
        for channel in &mut self.dye {
            channel.zero();
        }
        self.curl.zero();
        self.divergence.zero();
        self.last_divergence = 0.0;
    }

    /// Clears any non-finite cell. Returns the number of cells repaired.
    pub fn sanitize(&mut self) -> usize {
        let mut fixed = self.vel.sanitize() + self.pressure.sanitize();
        for channel in &mut self.dye {
            fixed += channel.sanitize();
        }
        fixed
    }
}
