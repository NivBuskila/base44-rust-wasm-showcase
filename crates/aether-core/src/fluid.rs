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
//! central differences on both sides the composed operator is the wide
//! Laplacian instead, which differs from the five-point one by a checkerboard
//! mode the solve cannot see: measured residual divergence then plateaus
//! around 10% of `max |u|` no matter how many iterations are thrown at it,
//! against ~0.1% here. Half a cell at 256x144 is a fifth of a screen pixel, so
//! the rest of the engine keeps sampling velocity at lattice points.
//!
//! Advection is MacCormack/BFECC error-corrected with an RK2 (midpoint)
//! backtrace and a local extremum limiter. Both halves matter: first-order
//! Euler backtracing turns the dye into mush within a second at the 200+
//! cells/s a swipe produces, and an *unlimited* MacCormack correction rings at
//! sharp dye edges and eventually blows up. The trace positions depend only on
//! the velocity field, so all five advected channels share one pair of maps.
//!
//! Walls — the domain border and every obstacle cell — are collapsed into a
//! single `solid` mask, which is the only thing the kernels branch on. Solid
//! cells hold zero velocity and are excluded from every write, so the invariant
//! "solids are at rest" holds between stages rather than being patched up at
//! the end.

use crate::config::Params;
use crate::field::{Grid, VecField};
use crate::math::{decay, normalize};

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
use core::arch::wasm32::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_ge, f32x4_mul, f32x4_splat, f32x4_sub, v128,
    v128_bitselect,
};

/// Dye channel count (RGB).
pub const DYE_CHANNELS: usize = 3;

/// Jacobi sweeps used by the viscous diffusion solve.
const DIFFUSION_ITERS: usize = 20;

/// Viscosity at or below this leaves the field unchanged to within float noise,
/// so the diffusion stage is skipped entirely. The default viscosity (2e-5) is
/// well under it: diffusion costs nothing unless the HUD asks for it.
const VISCOSITY_EPSILON: f32 = 1e-4;

/// Ceiling on the velocity change one confinement step may apply, in cells/s.
///
/// Confinement is an anti-dissipation term — it *adds* energy — so an
/// unbounded version turns one fast gesture into an explosion a few frames
/// later, on exactly the frames where `dt` also spikes.
const MAX_CONFINE_IMPULSE: f32 = 50.0;

/// Fractions of an advection trace tried, in order, when the full backtrace
/// lands inside a wall.
const TRACE_RETRIES: [f32; 3] = [0.5, 0.25, 0.1];

/// A velocity + dye field on a fixed grid.
pub struct Fluid {
    w: usize,
    h: usize,
    vel: VecField,
    vel_tmp: VecField,
    dye: [Grid; DYE_CHANNELS],
    dye_tmp: [Grid; DYE_CHANNELS],
    pressure: Grid,
    pressure_tmp: Grid,
    divergence: Grid,
    curl: Grid,
    obstacle: Grid,
    scratch: Grid,
    /// `obstacle` with the domain border forced solid. Every kernel tests this
    /// one array instead of re-deriving "am I on the border?" per cell.
    solid: Grid,
    /// Advection source positions for this step (`u` = x, `v` = y).
    trace_back: VecField,
    /// Forward trace positions, used by the MacCormack error estimate.
    trace_fwd: VecField,
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
            dye_tmp: [Grid::new(w, h), Grid::new(w, h), Grid::new(w, h)],
            pressure: Grid::new(w, h),
            pressure_tmp: Grid::new(w, h),
            divergence: Grid::new(w, h),
            curl: Grid::new(w, h),
            obstacle: Grid::new(w, h),
            scratch: Grid::new(w, h),
            solid: Grid::new(w, h),
            trace_back: VecField::new(w, h),
            trace_fwd: VecField::new(w, h),
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
        for c in 0..DYE_CHANNELS {
            self.dye[c].splat(x, y, radius, rgb[c]);
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
        for c in 0..DYE_CHANNELS {
            self.dye[c].zero();
        }
        self.curl.zero();
        self.divergence.zero();
        self.last_divergence = 0.0;
    }

    /// Advances the simulation by `dt` seconds.
    ///
    /// External forces and dye are expected to have been injected already (the
    /// spell layer runs first), so this is the pure solver: advect, diffuse,
    /// confine, project, dissipate.
    pub fn step(&mut self, dt: f32, params: &Params) {
        let dt = sane_dt(dt);

        self.refresh_solid();
        // A silhouette cell that only just turned solid still holds the
        // velocity it had as fluid, and every stage below assumes otherwise.
        self.enforce_boundaries();

        self.advect(dt);

        if params.viscosity > VISCOSITY_EPSILON {
            self.diffuse(dt, params.viscosity);
        }

        // Curl is recomputed every step even when confinement is off: the
        // particle system reads it through `curl_at` for spin.
        self.compute_curl();
        if params.vorticity > 0.0 {
            self.confine_vorticity(dt, params.vorticity);
        }

        self.close_solid_faces();
        self.project(params.pressure_iters);

        // A uniform scale cannot introduce divergence, so dissipation is safe
        // to apply after the projection rather than before it.
        self.vel.scale(decay(params.velocity_dissipation, dt));
        for c in 0..DYE_CHANNELS {
            self.dye[c].scale(decay(params.dye_dissipation, dt));
        }

        self.last_divergence = self.max_divergence();
    }

    // ------------------------------------------------------------- advection

    fn advect(&mut self, dt: f32) {
        self.build_traces(dt);

        advect_field(
            &self.vel.u,
            &mut self.vel_tmp.u,
            &mut self.scratch,
            &self.trace_back,
            &self.trace_fwd,
        );
        advect_field(
            &self.vel.v,
            &mut self.vel_tmp.v,
            &mut self.scratch,
            &self.trace_back,
            &self.trace_fwd,
        );
        core::mem::swap(&mut self.vel, &mut self.vel_tmp);
        self.enforce_boundaries();

        for c in 0..DYE_CHANNELS {
            advect_field(
                &self.dye[c],
                &mut self.dye_tmp[c],
                &mut self.scratch,
                &self.trace_back,
                &self.trace_fwd,
            );
            core::mem::swap(&mut self.dye[c], &mut self.dye_tmp[c]);
        }
    }

    /// Fills the backward and forward trace maps for this step.
    ///
    /// Shared by all five advected channels, which is the whole reason they are
    /// materialised instead of recomputed: the RK2 trace costs two velocity
    /// samples plus the obstacle walk, and doing that five times per cell is
    /// the difference between a 3 ms and an 8 ms frame.
    fn build_traces(&mut self, dt: f32) {
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let fx = x as f32;
                let fy = y as f32;
                let u0 = self.vel.u.data[i];
                let v0 = self.vel.v.data[i];

                let (um, vm) = self.vel.sample(fx - 0.5 * dt * u0, fy - 0.5 * dt * v0);
                let (bx, by) = self.trace_to_fluid(fx, fy, fx - dt * um, fy - dt * vm);
                self.trace_back.u.data[i] = bx;
                self.trace_back.v.data[i] = by;

                let (up, vp) = self.vel.sample(fx + 0.5 * dt * u0, fy + 0.5 * dt * v0);
                let (gx, gy) = self.trace_to_fluid(fx, fy, fx + dt * up, fy + dt * vp);
                self.trace_fwd.u.data[i] = gx;
                self.trace_fwd.v.data[i] = gy;
            }
        }
    }

    /// Shortens a trace until its endpoint is out of the walls.
    ///
    /// Without this the bilinear fetch mixes the stale values sitting inside
    /// the body silhouette into the fluid, which reads on screen as dye
    /// bleeding straight through the body.
    fn trace_to_fluid(&self, x0: f32, y0: f32, tx: f32, ty: f32) -> (f32, f32) {
        if !self.solid_at(tx, ty) {
            return (tx, ty);
        }
        for &t in &TRACE_RETRIES {
            let px = x0 + (tx - x0) * t;
            let py = y0 + (ty - y0) * t;
            if !self.solid_at(px, py) {
                return (px, py);
            }
        }
        // Everything along the ray is wall (the cell itself is solid): stay
        // put, which leaves the field there untouched.
        (x0, y0)
    }

    /// Nearest-cell wall test. Bilinear would smear the silhouette by half a
    /// cell in every direction and let traces reach one cell into the body.
    #[inline]
    fn solid_at(&self, x: f32, y: f32) -> bool {
        let xi = round_index(x, self.w);
        let yi = round_index(y, self.h);
        self.solid.data[yi * self.w + xi] >= 0.5
    }

    // ------------------------------------------------------------- diffusion

    fn diffuse(&mut self, dt: f32, viscosity: f32) {
        // Cells are square in screen space (256x144 over 16:9), so a single
        // spacing h = 1/w serves both axes and the discrete Laplacian picks up
        // a w^2 factor. Folding it in here is what makes one slider value mean
        // the same viscosity at any grid resolution.
        let a = viscosity * dt * (self.w * self.w) as f32;
        let (w, h) = (self.w, self.h);

        // Jacobi needs the pre-diffusion field pinned as the right-hand side
        // for every sweep, so it gets a copy; the iterate itself ping-pongs
        // with `scratch`.
        self.vel_tmp.u.data.copy_from_slice(&self.vel.u.data);
        for _ in 0..DIFFUSION_ITERS {
            jacobi_diffuse(
                &self.vel.u.data,
                &mut self.scratch.data,
                &self.vel_tmp.u.data,
                &self.solid.data,
                w,
                h,
                a,
            );
            core::mem::swap(&mut self.vel.u.data, &mut self.scratch.data);
        }

        self.vel_tmp.v.data.copy_from_slice(&self.vel.v.data);
        for _ in 0..DIFFUSION_ITERS {
            jacobi_diffuse(
                &self.vel.v.data,
                &mut self.scratch.data,
                &self.vel_tmp.v.data,
                &self.solid.data,
                w,
                h,
                a,
            );
            core::mem::swap(&mut self.vel.v.data, &mut self.scratch.data);
        }
    }

    // ------------------------------------------------------------- vorticity

    /// Central-difference curl, `omega = dv/dx - du/dy`, zero inside walls.
    fn compute_curl(&mut self) {
        let (w, h) = (self.w, self.h);
        self.curl.zero();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                let dvdx = self.vel.v.data[i + 1] - self.vel.v.data[i - 1];
                let dudy = self.vel.u.data[i + w] - self.vel.u.data[i - w];
                self.curl.data[i] = 0.5 * (dvdx - dudy);
            }
        }
    }

    /// Fedkiw-style confinement: `f = eps * (N x omega)` with
    /// `N = grad|omega| / |grad|omega||`, i.e. a force that spins each cell
    /// back towards the nearest vorticity peak.
    fn confine_vorticity(&mut self, dt: f32, strength: f32) {
        let (w, h) = (self.w, self.h);
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                let gx = 0.5 * (self.curl.data[i + 1].abs() - self.curl.data[i - 1].abs());
                let gy = 0.5 * (self.curl.data[i + w].abs() - self.curl.data[i - w].abs());
                // Returns (0, 0) on a vanishing gradient, so flat regions get
                // no force instead of a normalised division by ~0.
                let (nx, ny) = normalize(gx, gy);
                let f = strength * self.curl.data[i] * dt;
                // `max`/`min` rather than `clamp`: they swallow a NaN instead
                // of propagating it, and NaN * 0 shows up here whenever curl
                // has already gone bad.
                self.vel.u.data[i] += (ny * f).max(-MAX_CONFINE_IMPULSE).min(MAX_CONFINE_IMPULSE);
                self.vel.v.data[i] += (-nx * f).max(-MAX_CONFINE_IMPULSE).min(MAX_CONFINE_IMPULSE);
            }
        }
    }

    // ------------------------------------------------------------ projection

    fn project(&mut self, iters: usize) {
        self.compute_divergence();
        let (w, h) = (self.w, self.h);

        // `self.pressure` still holds last frame's solution and the field
        // barely changes between frames, so Jacobi starts far closer to the
        // answer than a zeroed guess: same iteration count, visibly lower
        // residual divergence.
        for _ in 0..iters.clamp(1, 200) {
            jacobi_pressure(
                &self.pressure.data,
                &mut self.pressure_tmp.data,
                &self.divergence.data,
                &self.solid.data,
                w,
                h,
            );
            core::mem::swap(&mut self.pressure.data, &mut self.pressure_tmp.data);
        }

        self.recentre_pressure();
        self.subtract_pressure_gradient();
    }

    /// Kills the flux through every face that touches a wall.
    ///
    /// A face between fluid and solid is not a degree of freedom: the pressure
    /// solve treats it as Neumann and so can never correct it, which means an
    /// uncorrected inflow there is mass vanishing *into* the body — the leak
    /// this whole boundary treatment exists to prevent. Only the component
    /// normal to the face is removed, so fluid still slides along a silhouette.
    fn close_solid_faces(&mut self) {
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w && self.solid.data[i + 1] >= 0.5 {
                    self.vel.u.data[i] = 0.0;
                }
                if i + w < w * h && self.solid.data[i + w] >= 0.5 {
                    self.vel.v.data[i] = 0.0;
                }
            }
        }
    }

    fn compute_divergence(&mut self) {
        let (w, h) = (self.w, self.h);
        self.divergence.zero();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                // Faces touching a wall carry no flux and have already been
                // zeroed, so the no-penetration condition needs no branch here.
                let du = self.vel.u.data[i] - self.vel.u.data[i - 1];
                let dv = self.vel.v.data[i] - self.vel.v.data[i - w];
                self.divergence.data[i] = du + dv;
            }
        }
    }

    /// Removes the constant component of the pressure.
    ///
    /// An all-Neumann Poisson problem only fixes pressure up to a constant,
    /// and each Jacobi sweep shifts that constant by `-mean(div) / 4`. Warm
    /// starting feeds the shift back in every frame, so after a few thousand
    /// frames the field would drift to infinity — invisible in the velocity
    /// (the gradient of a constant is zero) right up until it overflows.
    fn recentre_pressure(&mut self) {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (p, s) in self.pressure.data.iter().zip(&self.solid.data) {
            if *s < 0.5 {
                sum += *p as f64;
                count += 1;
            }
        }
        if count == 0 {
            return;
        }
        let mean = (sum / count as f64) as f32;
        if !mean.is_finite() {
            self.pressure.zero();
            return;
        }
        for (p, s) in self.pressure.data.iter_mut().zip(&self.solid.data) {
            if *s < 0.5 {
                *p -= mean;
            } else {
                *p = 0.0;
            }
        }
    }

    fn subtract_pressure_gradient(&mut self) {
        let (w, h) = (self.w, self.h);
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                let pc = self.pressure.data[i];
                // Forward difference, matching the backward-difference
                // divergence: `wall_pick` makes the gradient vanish across a
                // closed face, leaving that face's zero flux untouched.
                let r = wall_pick(self.pressure.data[i + 1], self.solid.data[i + 1], pc);
                let d = wall_pick(self.pressure.data[i + w], self.solid.data[i + w], pc);
                self.vel.u.data[i] -= r - pc;
                self.vel.v.data[i] -= d - pc;
            }
        }
    }

    /// `max |div u|` over the non-solid interior — the residual the projection
    /// failed to remove.
    fn max_divergence(&self) -> f32 {
        let (w, h) = (self.w, self.h);
        let mut worst = 0.0f32;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                if self.solid.data[i] >= 0.5 {
                    continue;
                }
                let du = self.vel.u.data[i] - self.vel.u.data[i - 1];
                let dv = self.vel.v.data[i] - self.vel.v.data[i - w];
                // `max` drops NaN rather than latching it, so a poisoned field
                // still reports a usable number to the HUD.
                worst = worst.max((du + dv).abs());
            }
        }
        worst
    }

    // -------------------------------------------------------------- boundary

    /// Rebuilds the solver's wall mask from the obstacle field.
    fn refresh_solid(&mut self) {
        let (w, h) = (self.w, self.h);
        self.solid.data.copy_from_slice(&self.obstacle.data);
        for x in 0..w {
            self.solid.data[x] = 1.0;
            self.solid.data[(h - 1) * w + x] = 1.0;
        }
        for y in 0..h {
            self.solid.data[y * w] = 1.0;
            self.solid.data[y * w + w - 1] = 1.0;
        }
    }

    /// Zeroes velocity inside obstacles and applies no-slip at the border.
    /// Shared by the solver stages, so it lives here rather than inline.
    pub(crate) fn enforce_boundaries(&mut self) {
        for y in 0..self.h {
            for x in 0..self.w {
                let i = y * self.w + x;
                let solid = self.obstacle.data[i] >= 0.5;
                let border = x == 0 || y == 0 || x == self.w - 1 || y == self.h - 1;
                if solid || border {
                    self.vel.u.data[i] = 0.0;
                    self.vel.v.data[i] = 0.0;
                }
            }
        }
    }

    /// Clears any non-finite cell. Returns the number of cells repaired.
    pub fn sanitize(&mut self) -> usize {
        let mut fixed = self.vel.sanitize() + self.pressure.sanitize();
        for c in 0..DYE_CHANNELS {
            fixed += self.dye[c].sanitize();
        }
        fixed
    }
}

// ------------------------------------------------------------------ helpers

/// Clamps a timestep into the range the explicit stages stay stable over.
///
/// A backgrounded tab hands back a `dt` of several seconds; advection would
/// survive it but confinement and the dye splats would not.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 480.0, 1.0 / 20.0)
}

/// Nearest lattice index to `x`, clamped into `[0, n - 1]` and NaN-safe.
#[inline]
fn round_index(x: f32, n: usize) -> usize {
    if x.is_nan() {
        return 0;
    }
    x.round().clamp(0.0, (n - 1) as f32) as usize
}

/// Substitutes the centre value for a wall neighbour.
///
/// For pressure this is the Neumann condition `dp/dn = 0` on a solid face.
/// Reading the wall cell's own pressure instead is precisely the bug that lets
/// fluid pour through the body silhouette.
#[inline]
fn wall_pick(value: f32, solid: f32, centre: f32) -> f32 {
    if solid >= 0.5 {
        centre
    } else {
        value
    }
}

/// Min and max of the four lattice values a bilinear fetch at `(x, y)` would
/// blend. Mirrors [`Grid::sample`]'s clamping exactly, which is what makes it a
/// valid limiter for the value that fetch produced.
#[inline]
fn bilinear_bounds(g: &Grid, x: f32, y: f32) -> (f32, f32) {
    let x = if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, (g.w - 1) as f32)
    };
    let y = if y.is_nan() {
        0.0
    } else {
        y.clamp(0.0, (g.h - 1) as f32)
    };
    let ix0 = x as usize;
    let iy0 = y as usize;
    let ix1 = (ix0 + 1).min(g.w - 1);
    let iy1 = (iy0 + 1).min(g.h - 1);
    let row0 = iy0 * g.w;
    let row1 = iy1 * g.w;
    let a = g.data[row0 + ix0];
    let b = g.data[row0 + ix1];
    let c = g.data[row1 + ix0];
    let d = g.data[row1 + ix1];
    (a.min(b).min(c).min(d), a.max(b).max(c).max(d))
}

/// MacCormack (error-corrected) semi-Lagrangian advection of one scalar field.
///
/// `out` receives the advected field and `scratch` is clobbered. The three
/// passes are: backward trace, forward trace of that result, then the
/// second-order correction `q_hat + (q - q_bar) / 2` limited to the range the
/// backward fetch could legitimately have returned.
fn advect_field(src: &Grid, out: &mut Grid, scratch: &mut Grid, back: &VecField, fwd: &VecField) {
    let n = src.data.len();
    if out.data.len() != n
        || scratch.data.len() != n
        || back.u.data.len() != n
        || back.v.data.len() != n
        || fwd.u.data.len() != n
        || fwd.v.data.len() != n
    {
        return;
    }

    for (o, (bx, by)) in out
        .data
        .iter_mut()
        .zip(back.u.data.iter().zip(&back.v.data))
    {
        *o = src.sample(*bx, *by);
    }

    for (s, (gx, gy)) in scratch
        .data
        .iter_mut()
        .zip(fwd.u.data.iter().zip(&fwd.v.data))
    {
        *s = out.sample(*gx, *gy);
    }

    for (i, (bx, by)) in back.u.data.iter().zip(&back.v.data).enumerate() {
        let (lo, hi) = bilinear_bounds(src, *bx, *by);
        let corrected = out.data[i] + 0.5 * (src.data[i] - scratch.data[i]);
        // The limiter is not optional: unlimited MacCormack overshoots at every
        // sharp dye edge, and the overshoot compounds into ringing and then
        // into NaN within seconds. `max`/`min` also scrub a stray NaN, since
        // both return the bound when one operand is not a number.
        out.data[i] = corrected.max(lo).min(hi);
    }
}

// ------------------------------------------------------------------ kernels
//
// The two Jacobi sweeps are the hot loops of the whole engine: at 256x144 and
// 28 pressure iterations they are ~1M cell updates per frame. Both have a
// wasm32 SIMD128 path and a scalar fallback that stays compiled everywhere, so
// the two can be diffed against each other.

/// One Jacobi sweep of `laplacian(p) = div`, with walls as Neumann boundaries.
///
/// Writes *every* cell of `out`, walls included, because the caller ping-pongs
/// the two buffers and stale values would otherwise survive forever.
#[inline]
fn jacobi_pressure(p: &[f32], out: &mut [f32], div: &[f32], solid: &[f32], w: usize, h: usize) {
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    jacobi_pressure_simd(p, out, div, solid, w, h);
    #[cfg(not(all(target_arch = "wasm32", feature = "simd")))]
    jacobi_pressure_scalar(p, out, div, solid, w, h);
}

/// One Jacobi sweep of `(I - a * laplacian) u = rhs`.
#[inline]
fn jacobi_diffuse(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    jacobi_diffuse_simd(x, out, rhs, solid, w, h, a);
    #[cfg(not(all(target_arch = "wasm32", feature = "simd")))]
    jacobi_diffuse_scalar(x, out, rhs, solid, w, h, a);
}

/// True when the kernels cannot run over these buffers at all.
#[inline]
fn kernel_unusable(n: usize, w: usize, h: usize, lens: [usize; 4]) -> bool {
    w < 3 || h < 3 || lens.iter().any(|&l| l < n)
}

#[inline]
fn zero_walls(out: &mut [f32], w: usize, h: usize) {
    if w < 2 || h < 2 || out.len() < w * h {
        out.fill(0.0);
        return;
    }
    out[..w].fill(0.0);
    out[(h - 1) * w..h * w].fill(0.0);
    for y in 1..h - 1 {
        out[y * w] = 0.0;
        out[y * w + w - 1] = 0.0;
    }
}

/// Pressure update for one cell. Also the tail of the SIMD loop, so the two
/// paths cannot drift apart in the arithmetic.
#[inline]
fn pressure_cell(p: &[f32], div: &[f32], solid: &[f32], i: usize, w: usize) -> f32 {
    if solid[i] >= 0.5 {
        return 0.0;
    }
    let pc = p[i];
    let l = wall_pick(p[i - 1], solid[i - 1], pc);
    let r = wall_pick(p[i + 1], solid[i + 1], pc);
    let u = wall_pick(p[i - w], solid[i - w], pc);
    let d = wall_pick(p[i + w], solid[i + w], pc);
    // Grouped exactly as the SIMD path adds it, so the results are bit-equal
    // and the cross-check test can use a tight tolerance.
    ((l + r) + (u + d) - div[i]) * 0.25
}

/// Diffusion update for one cell.
#[inline]
fn diffuse_cell(x: &[f32], rhs: &[f32], solid: &[f32], i: usize, w: usize, a: f32, inv: f32) -> f32 {
    if solid[i] >= 0.5 {
        return 0.0;
    }
    // A wall neighbour contributes its own (zero) velocity rather than a
    // mirrored copy of the centre: a wall *drags* the fluid beside it, which is
    // the entire physical content of no-slip.
    let sum = (x[i - 1] + x[i + 1]) + (x[i - w] + x[i + w]);
    (rhs[i] + a * sum) * inv
}

#[cfg_attr(all(target_arch = "wasm32", feature = "simd"), allow(dead_code))]
fn jacobi_pressure_scalar(
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [p.len(), out.len(), div.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    zero_walls(out, w, h);
    for y in 1..h - 1 {
        let row = y * w;
        for x in 1..w - 1 {
            let i = row + x;
            out[i] = pressure_cell(p, div, solid, i, w);
        }
    }
}

#[cfg_attr(all(target_arch = "wasm32", feature = "simd"), allow(dead_code))]
fn jacobi_diffuse_scalar(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [x.len(), out.len(), rhs.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let inv = 1.0 / (1.0 + 4.0 * a);
    zero_walls(out, w, h);
    for y in 1..h - 1 {
        let row = y * w;
        for xi in 1..w - 1 {
            let i = row + xi;
            out[i] = diffuse_cell(x, rhs, solid, i, w, a, inv);
        }
    }
}

// --- wasm32 SIMD128 paths -------------------------------------------------
//
// Lanes are assembled with the safe `f32x4` constructor rather than
// `v128_load`, which takes a raw pointer and would need `unsafe`; this crate
// forbids it. The arithmetic below is still genuine packed f32x4 work, which is
// where the time goes.

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
fn lanes(s: &[f32], i: usize) -> v128 {
    f32x4(s[i], s[i + 1], s[i + 2], s[i + 3])
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
fn store_lanes(s: &mut [f32], i: usize, v: v128) {
    s[i] = f32x4_extract_lane::<0>(v);
    s[i + 1] = f32x4_extract_lane::<1>(v);
    s[i + 2] = f32x4_extract_lane::<2>(v);
    s[i + 3] = f32x4_extract_lane::<3>(v);
}

/// Lane-wise [`wall_pick`].
#[cfg(all(target_arch = "wasm32", feature = "simd"))]
#[inline]
fn wall_pick_v(value: v128, solid: v128, centre: v128) -> v128 {
    v128_bitselect(centre, value, f32x4_ge(solid, f32x4_splat(0.5)))
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
fn jacobi_pressure_simd(
    p: &[f32],
    out: &mut [f32],
    div: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [p.len(), out.len(), div.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let zero = f32x4_splat(0.0);
    let quarter = f32x4_splat(0.25);
    let half = f32x4_splat(0.5);

    zero_walls(out, w, h);
    for y in 1..h - 1 {
        let row = y * w;
        let mut x = 1;
        // A row is contiguous, so the left/right neighbour vectors are just
        // this row shifted by one element — no shuffles needed.
        while x + 4 <= w - 1 {
            let i = row + x;
            let pc = lanes(p, i);
            let l = wall_pick_v(lanes(p, i - 1), lanes(solid, i - 1), pc);
            let r = wall_pick_v(lanes(p, i + 1), lanes(solid, i + 1), pc);
            let u = wall_pick_v(lanes(p, i - w), lanes(solid, i - w), pc);
            let d = wall_pick_v(lanes(p, i + w), lanes(solid, i + w), pc);
            let sum = f32x4_add(f32x4_add(l, r), f32x4_add(u, d));
            let res = f32x4_mul(f32x4_sub(sum, lanes(div, i)), quarter);
            let solid_centre = f32x4_ge(lanes(solid, i), half);
            store_lanes(out, i, v128_bitselect(zero, res, solid_centre));
            x += 4;
        }
        while x < w - 1 {
            let i = row + x;
            out[i] = pressure_cell(p, div, solid, i, w);
            x += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "simd"))]
fn jacobi_diffuse_simd(
    x: &[f32],
    out: &mut [f32],
    rhs: &[f32],
    solid: &[f32],
    w: usize,
    h: usize,
    a: f32,
) {
    let n = w * h;
    if kernel_unusable(n, w, h, [x.len(), out.len(), rhs.len(), solid.len()]) {
        out.fill(0.0);
        return;
    }
    let inv_scalar = 1.0 / (1.0 + 4.0 * a);
    let zero = f32x4_splat(0.0);
    let half = f32x4_splat(0.5);
    let av = f32x4_splat(a);
    let inv = f32x4_splat(inv_scalar);

    zero_walls(out, w, h);
    for y in 1..h - 1 {
        let row = y * w;
        let mut xi = 1;
        while xi + 4 <= w - 1 {
            let i = row + xi;
            let sum = f32x4_add(
                f32x4_add(lanes(x, i - 1), lanes(x, i + 1)),
                f32x4_add(lanes(x, i - w), lanes(x, i + w)),
            );
            let res = f32x4_mul(f32x4_add(lanes(rhs, i), f32x4_mul(av, sum)), inv);
            let solid_centre = f32x4_ge(lanes(solid, i), half);
            store_lanes(out, i, v128_bitselect(zero, res, solid_centre));
            xi += 4;
        }
        while xi < w - 1 {
            let i = row + xi;
            out[i] = diffuse_cell(x, rhs, solid, i, w, a, inv_scalar);
            xi += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUID_H, FLUID_W};
    use crate::rng::Rng;

    const DT: f32 = 1.0 / 60.0;

    /// Params with every decay and side effect off, so a test measures exactly
    /// the stage it is aiming at.
    fn inert_params() -> Params {
        Params {
            velocity_dissipation: 0.0,
            dye_dissipation: 0.0,
            vorticity: 0.0,
            viscosity: 0.0,
            pressure_iters: 30,
            ..Params::default()
        }
    }

    fn dye_mass(f: &Fluid) -> f32 {
        f.dye[0].sum()
    }

    /// Dye centroid in grid space; `None` when there is no dye to speak of.
    fn dye_centroid(f: &Fluid) -> Option<(f32, f32)> {
        let mut wsum = 0.0;
        let mut xs = 0.0;
        let mut ys = 0.0;
        for y in 0..f.h {
            for x in 0..f.w {
                let m = f.dye[0].data[y * f.w + x];
                wsum += m;
                xs += m * x as f32;
                ys += m * y as f32;
            }
        }
        if wsum <= 1e-6 {
            return None;
        }
        Some((xs / wsum, ys / wsum))
    }

    fn total_curl(f: &Fluid) -> f32 {
        f.curl.data.iter().map(|c| c.abs()).sum()
    }

    /// A vertical solid bar spanning the full height.
    fn bar_mask(w: usize, h: usize, x0: usize, x1: usize) -> Grid {
        let mut m = Grid::new(w, h);
        for y in 0..h {
            for x in x0..x1 {
                m.data[y * w + x] = 1.0;
            }
        }
        m
    }

    #[test]
    fn projection_leaves_the_field_nearly_divergence_free() {
        let mut f = Fluid::new(FLUID_W, FLUID_H);
        let params = Params::default();
        // Stir hard enough that the divergence being removed is large.
        for k in 0..24 {
            let t = k as f32 * 0.3;
            f.add_force(
                90.0 + 40.0 * t.sin(),
                72.0 + 30.0 * t.cos(),
                260.0 * t.cos(),
                -220.0 * t.sin(),
                14.0,
            );
            f.add_dye(90.0, 72.0, [1.0, 0.4, 0.1], 10.0);
            f.step(DT, &params);
        }
        let speed = f.max_speed();
        assert!(speed > 10.0, "the stir did nothing: max speed {speed}");
        let ratio = f.last_divergence() / speed;
        assert!(
            ratio < 0.02,
            "field is not incompressible: max|div| {} vs max|u| {speed} (ratio {ratio})",
            f.last_divergence()
        );
    }

    #[test]
    fn uniform_velocity_survives_advection() {
        let mut f = Fluid::new(48, 32);
        f.vel.u.data.fill(30.0);
        f.vel.v.data.fill(0.0);
        f.advect(1.0 / 30.0);
        for y in 4..28 {
            for x in 4..44 {
                let i = y * 48 + x;
                assert!(
                    (f.vel.u.data[i] - 30.0).abs() < 1e-3,
                    "u drifted at {x},{y}: {}",
                    f.vel.u.data[i]
                );
                assert!(f.vel.v.data[i].abs() < 1e-3, "v appeared at {x},{y}");
            }
        }
    }

    #[test]
    fn a_dye_blob_rides_the_flow_by_v_times_dt() {
        let mut f = Fluid::new(64, 48);
        f.vel.u.data.fill(20.0);
        f.add_dye(16.0, 24.0, [1.0, 0.0, 0.0], 6.0);
        let (x0, y0) = dye_centroid(&f).expect("dye was injected");

        let dt = 0.05;
        let steps = 4;
        for _ in 0..steps {
            f.advect(dt);
        }
        let (x1, y1) = dye_centroid(&f).expect("dye survived");

        let expected = 20.0 * dt * steps as f32;
        assert!(
            (x1 - x0 - expected).abs() < 0.2,
            "centroid moved {} cells, expected {expected}",
            x1 - x0
        );
        assert!((y1 - y0).abs() < 0.05, "centroid drifted in y by {}", y1 - y0);
    }

    #[test]
    fn advection_alone_conserves_dye_mass() {
        let mut f = Fluid::new(64, 48);
        f.vel.u.data.fill(10.0);
        f.vel.v.data.fill(4.0);
        f.add_dye(16.0, 16.0, [1.0, 0.0, 0.0], 7.0);
        let before = dye_mass(&f);
        assert!(before > 0.0);
        for _ in 0..30 {
            f.advect(0.05);
        }
        let after = dye_mass(&f);
        let drift = (after - before).abs() / before;
        assert!(drift < 0.03, "dye mass drifted {:.2}%", drift * 100.0);
    }

    #[test]
    fn obstacle_cells_are_exactly_at_rest_after_a_step() {
        let mut f = Fluid::new(64, 48);
        f.set_obstacle(&bar_mask(64, 48, 28, 34));
        let params = Params::default();
        for _ in 0..20 {
            // Aim straight at the bar.
            f.add_force(20.0, 24.0, 300.0, 0.0, 9.0);
            f.step(DT, &params);
        }
        for y in 0..48 {
            for x in 0..64 {
                let i = y * 64 + x;
                if f.solid.data[i] >= 0.5 {
                    assert_eq!(f.vel.u.data[i], 0.0, "u nonzero in wall at {x},{y}");
                    assert_eq!(f.vel.v.data[i], 0.0, "v nonzero in wall at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn fluid_does_not_leak_through_a_solid_wall() {
        let (w, h) = (64, 48);
        let mut f = Fluid::new(w, h);
        f.set_obstacle(&bar_mask(w, h, 30, 34));
        let params = Params::default();
        for _ in 0..300 {
            f.add_force(12.0, 24.0, 320.0, 0.0, 8.0);
            f.add_dye(12.0, 24.0, [0.4, 0.4, 0.4], 6.0);
            f.step(DT, &params);
        }
        assert!(f.max_speed() > 10.0, "the push never took");

        let mut far_speed = 0.0f32;
        let mut far_dye = 0.0f32;
        for y in 0..h {
            for x in 38..w {
                let i = y * w + x;
                far_speed = far_speed
                    .max((f.vel.u.data[i] * f.vel.u.data[i] + f.vel.v.data[i] * f.vel.v.data[i]).sqrt());
                far_dye += f.dye[0].data[i];
            }
        }
        assert!(
            far_speed < 1e-4,
            "velocity leaked past the wall: {far_speed} (near side {})",
            f.max_speed()
        );
        assert!(far_dye < 1e-6, "dye leaked past the wall: {far_dye}");
    }

    #[test]
    fn vorticity_confinement_adds_curl() {
        // A shear layer: two opposed jets with a thin gap between them.
        fn run(vorticity: f32) -> f32 {
            let mut f = Fluid::new(96, 64);
            let params = Params {
                vorticity,
                ..inert_params()
            };
            for _ in 0..90 {
                f.add_force(40.0, 28.0, 120.0, 0.0, 8.0);
                f.add_force(40.0, 36.0, -120.0, 0.0, 8.0);
                f.step(DT, &params);
            }
            total_curl(&f)
        }
        let without = run(0.0);
        let with = run(20.0);
        assert!(without > 0.0, "the shear layer produced no curl at all");
        assert!(
            with > without * 1.05,
            "confinement did not add curl: {with} vs {without}"
        );
    }

    #[test]
    fn six_hundred_aggressive_steps_stay_finite_and_bounded() {
        let mut f = Fluid::new(128, 72);
        f.set_obstacle(&bar_mask(128, 72, 60, 64));
        let params = Params::default();
        let mut rng = Rng::new(0x51DE_5EED);
        for k in 0..600 {
            let x = rng.range(2.0, 126.0);
            let y = rng.range(2.0, 70.0);
            f.add_force(x, y, rng.signed() * 400.0, rng.signed() * 400.0, 10.0);
            f.add_dye(x, y, [1.0, 0.6, 0.2], 8.0);
            // Alternate 30 and 240 fps to exercise the whole dt range.
            f.step(if k % 2 == 0 { 1.0 / 30.0 } else { 1.0 / 240.0 }, &params);
            assert_eq!(f.sanitize(), 0, "non-finite cell appeared at step {k}");
        }
        let speed = f.max_speed();
        assert!(speed.is_finite() && speed < 4000.0, "max speed blew up: {speed}");
        assert!(f.pressure.max_abs() < 1e6, "pressure drifted: {}", f.pressure.max_abs());
        assert!(f.last_divergence() < 0.03 * speed, "divergence blew up");
    }

    #[test]
    fn extreme_dt_and_degenerate_forces_are_harmless() {
        let mut f = Fluid::new(32, 24);
        let params = Params::default();
        // Zero-size and non-finite forces must be ignored, not propagated.
        f.add_force(16.0, 12.0, 50.0, 50.0, 0.0);
        f.add_force(f32::NAN, 12.0, 50.0, 0.0, 5.0);
        f.add_force(16.0, 12.0, f32::INFINITY, 0.0, 5.0);
        f.add_dye(16.0, 12.0, [f32::NAN, 1.0, 0.0], 0.0);
        for dt in [0.0, -1.0, f32::NAN, f32::INFINITY, 1e9, 1e-9] {
            f.step(dt, &params);
        }
        assert_eq!(f.sanitize(), 0, "degenerate input poisoned the field");
        assert!(f.last_divergence().is_finite());
    }

    #[test]
    fn a_stray_nan_does_not_spread_through_advection() {
        let mut f = Fluid::new(32, 24);
        f.vel.u.data.fill(12.0);
        f.dye[0].data.fill(1.0);
        f.dye[0].data[12 * 32 + 16] = f32::NAN;
        f.advect(DT);
        // The extremum limiter clamps to real neighbouring values, so the NaN
        // cannot survive its own advection step.
        let bad = f.dye[0].data.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite dye cells survived");
    }

    #[test]
    fn a_degenerate_grid_does_not_panic() {
        let mut f = Fluid::new(2, 2);
        let params = Params::default();
        f.add_force(1.0, 1.0, 10.0, 10.0, 3.0);
        f.add_dye(1.0, 1.0, [1.0, 1.0, 1.0], 3.0);
        f.step(DT, &params);
        assert_eq!(f.last_divergence(), 0.0, "a 2x2 grid has no interior");
        assert_eq!(f.max_speed(), 0.0, "every cell of a 2x2 grid is a wall");
    }

    #[test]
    fn dissipation_is_frame_rate_independent() {
        let params = Params {
            velocity_dissipation: 2.0,
            dye_dissipation: 2.0,
            vorticity: 0.0,
            viscosity: 0.0,
            pressure_iters: 1,
            ..Params::default()
        };
        // Same simulated duration, four times the frame rate.
        let mut slow = Fluid::new(32, 24);
        slow.dye[0].data.fill(1.0);
        for _ in 0..12 {
            slow.step(1.0 / 24.0, &params);
        }
        let mut fast = Fluid::new(32, 24);
        fast.dye[0].data.fill(1.0);
        for _ in 0..48 {
            fast.step(1.0 / 96.0, &params);
        }
        let (a, b) = (slow.dye[0].data[12 * 32 + 16], fast.dye[0].data[12 * 32 + 16]);
        assert!((a - b).abs() < 1e-3, "dye decay differs with frame rate: {a} vs {b}");
    }

    #[test]
    fn default_viscosity_skips_diffusion_entirely() {
        assert!(
            Params::default().viscosity <= VISCOSITY_EPSILON,
            "the default viscosity should be below the skip threshold"
        );
        // A raised viscosity must actually smooth: a single spike spreads.
        let mut f = Fluid::new(32, 24);
        let i = 12 * 32 + 16;
        f.vel.u.data[i] = 100.0;
        f.diffuse(DT, 0.01);
        assert!(f.vel.u.data[i] < 100.0, "the spike did not flatten");
        assert!(f.vel.u.data[i + 1] > 0.1, "nothing diffused to the neighbour");
        let moved = f.vel.u.data.iter().filter(|v| v.abs() > 1e-6).count();
        assert!(moved > 5, "diffusion touched only {moved} cells");
    }

    #[test]
    fn diffusion_leaves_walls_at_rest() {
        let mut f = Fluid::new(32, 24);
        f.set_obstacle(&bar_mask(32, 24, 14, 18));
        f.vel.u.data.fill(50.0);
        f.enforce_boundaries();
        f.diffuse(DT, 0.01);
        for (i, s) in f.solid.data.iter().enumerate() {
            if *s >= 0.5 {
                assert_eq!(f.vel.u.data[i], 0.0, "diffusion moved a wall cell at {i}");
            }
        }
    }

    // --- kernel cross-checks ---------------------------------------------

    /// Deliberately naive pressure sweep, written from the 2-D indices with an
    /// explicit per-neighbour wall test. Independent of the shipping kernels,
    /// so agreement means something.
    fn reference_pressure(
        p: &[f32],
        div: &[f32],
        solid: &[f32],
        w: usize,
        h: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        let is_solid = |x: usize, y: usize| solid[y * w + x] >= 0.5;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if is_solid(x, y) {
                    continue;
                }
                let pc = p[y * w + x];
                let at = |x: usize, y: usize| if is_solid(x, y) { pc } else { p[y * w + x] };
                out[y * w + x] = (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1)
                    - div[y * w + x])
                    * 0.25;
            }
        }
        out
    }

    fn reference_diffuse(
        x: &[f32],
        rhs: &[f32],
        solid: &[f32],
        w: usize,
        h: usize,
        a: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        for j in 1..h - 1 {
            for i in 1..w - 1 {
                if solid[j * w + i] >= 0.5 {
                    continue;
                }
                let sum = x[j * w + i - 1] + x[j * w + i + 1] + x[(j - 1) * w + i] + x[(j + 1) * w + i];
                out[j * w + i] = (rhs[j * w + i] + a * sum) / (1.0 + 4.0 * a);
            }
        }
        out
    }

    /// Pseudo-random pressure, divergence and wall fields.
    fn kernel_fixture(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(0xBEEF_1234);
        let n = w * h;
        let p: Vec<f32> = (0..n).map(|_| rng.range(-5.0, 5.0)).collect();
        let div: Vec<f32> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
        let mut solid = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
                // A blob of interior walls plus the border, so both branches of
                // every neighbour select get exercised.
                let blob = (x as i32 - (w as i32) / 2).abs() < 3 && (y as i32 - 4).abs() < 2;
                if border || blob {
                    solid[y * w + x] = 1.0;
                }
            }
        }
        (p, div, solid)
    }

    #[test]
    fn pressure_kernel_matches_the_reference() {
        // 19 leaves a non-multiple-of-4 tail, 18 does not: both SIMD lanes and
        // the scalar tail get covered.
        for (w, h) in [(19usize, 11usize), (18, 12), (5, 5)] {
            let (p, div, solid) = kernel_fixture(w, h);
            let mut got = vec![0.0f32; w * h];
            jacobi_pressure(&p, &mut got, &div, &solid, w, h);
            let want = reference_pressure(&p, &div, &solid, w, h);
            for i in 0..w * h {
                assert!(
                    (got[i] - want[i]).abs() < 1e-5,
                    "{w}x{h} cell {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn diffusion_kernel_matches_the_reference() {
        for (w, h) in [(19usize, 11usize), (18, 12)] {
            let (rhs, x, solid) = kernel_fixture(w, h);
            let mut got = vec![0.0f32; w * h];
            jacobi_diffuse(&x, &mut got, &rhs, &solid, w, h, 0.37);
            let want = reference_diffuse(&x, &rhs, &solid, w, h, 0.37);
            for i in 0..w * h {
                assert!(
                    (got[i] - want[i]).abs() < 1e-5,
                    "{w}x{h} cell {i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn dispatched_kernels_match_the_scalar_fallback() {
        // On wasm32 with `simd` this diffs SIMD128 against scalar; elsewhere it
        // pins the dispatcher to the fallback it is supposed to select.
        let (w, h) = (23usize, 13usize);
        let (p, div, solid) = kernel_fixture(w, h);
        let mut dispatched = vec![0.0f32; w * h];
        let mut scalar = vec![0.0f32; w * h];

        jacobi_pressure(&p, &mut dispatched, &div, &solid, w, h);
        jacobi_pressure_scalar(&p, &mut scalar, &div, &solid, w, h);
        for i in 0..w * h {
            assert!(
                (dispatched[i] - scalar[i]).abs() <= 1e-6 * scalar[i].abs().max(1.0),
                "pressure cell {i}: {} vs {}",
                dispatched[i],
                scalar[i]
            );
        }

        jacobi_diffuse(&p, &mut dispatched, &div, &solid, w, h, 0.25);
        jacobi_diffuse_scalar(&p, &mut scalar, &div, &solid, w, h, 0.25);
        for i in 0..w * h {
            assert!(
                (dispatched[i] - scalar[i]).abs() <= 1e-6 * scalar[i].abs().max(1.0),
                "diffusion cell {i}: {} vs {}",
                dispatched[i],
                scalar[i]
            );
        }
    }

    #[test]
    fn kernels_refuse_undersized_buffers_instead_of_panicking() {
        let (w, h) = (8usize, 6usize);
        let short = vec![0.0f32; 4];
        let full = vec![0.0f32; w * h];
        let mut out = vec![9.0f32; w * h];
        jacobi_pressure(&short, &mut out, &full, &full, w, h);
        assert!(out.iter().all(|v| *v == 0.0));
        out.fill(9.0);
        jacobi_diffuse(&short, &mut out, &full, &full, w, h, 0.5);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn walls_are_neumann_not_dirichlet_in_the_pressure_solve() {
        // One fluid column beside a wall: a Dirichlet (read-the-wall) update
        // would pull the pressure towards the wall's stored value instead of
        // mirroring it, so the two differ by a wide margin here.
        let (w, h) = (7usize, 5usize);
        let n = w * h;
        let mut solid = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                if x == 0 || y == 0 || x == w - 1 || y == h - 1 || x == 4 {
                    solid[y * w + x] = 1.0;
                }
            }
        }
        let mut p = vec![0.0f32; n];
        // Poison the wall column: a correct solve never reads it.
        for y in 0..h {
            p[y * w + 4] = 1000.0;
        }
        let div = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        jacobi_pressure(&p, &mut out, &div, &solid, w, h);
        for y in 1..h - 1 {
            assert!(
                out[y * w + 3].abs() < 1e-6,
                "wall pressure leaked into the fluid: {}",
                out[y * w + 3]
            );
        }
    }

    #[test]
    fn warm_started_pressure_stays_centred() {
        // Hundreds of frames of warm starting must not let the Neumann null
        // space drift the pressure off to infinity.
        let mut f = Fluid::new(64, 48);
        let params = Params::default();
        for _ in 0..200 {
            f.add_force(20.0, 24.0, 200.0, 40.0, 8.0);
            f.step(DT, &params);
        }
        let mean: f32 = f.pressure.sum() / f.pressure.len() as f32;
        assert!(mean.abs() < 1.0, "pressure mean drifted to {mean}");
        assert!(f.pressure.max_abs() < 1e4, "pressure magnitude {}", f.pressure.max_abs());
    }

    #[test]
    fn a_still_field_stays_still() {
        let mut f = Fluid::new(64, 48);
        let params = Params::default();
        for _ in 0..10 {
            f.step(DT, &params);
        }
        assert_eq!(f.max_speed(), 0.0, "the solver invented motion from nothing");
        assert_eq!(f.last_divergence(), 0.0);
    }
}

#[cfg(test)]
mod probe {
    use super::*;
    use crate::config::{FLUID_H, FLUID_W};

    #[test]
    #[ignore]
    fn probe_variants() {
        // A: adversarial splat every frame, measured on the injection frame.
        // B: same, then a few frames with no new injection.
        // C: wide gentle splats (closer to the optical-flow drive).
        for (name, radius, force, quiet) in [("A", 14.0f32, 260.0f32, 0usize), ("B", 14.0, 260.0, 6), ("C", 34.0, 120.0, 0), ("D", 34.0, 120.0, 6)] {
            let mut f = Fluid::new(FLUID_W, FLUID_H);
            let params = Params::default();
            for k in 0..24 {
                let t = k as f32 * 0.3;
                f.add_force(90.0 + 40.0 * t.sin(), 72.0 + 30.0 * t.cos(), force * t.cos(), -force * 0.85 * t.sin(), radius);
                f.step(1.0/60.0, &params);
            }
            for _ in 0..quiet { f.step(1.0/60.0, &params); }
            println!("{name}: post {:.4} speed {:.2} ratio {:.5}", f.last_divergence(), f.max_speed(), f.last_divergence()/f.max_speed());
        }
    }

    #[test]
    #[ignore]
    fn probe_convergence() {
        for iters in [28usize, 60, 150, 400] {
            let mut f = Fluid::new(FLUID_W, FLUID_H);
            let params = Params { pressure_iters: iters, ..Params::default() };
            let mut pre = 0.0f32;
            for k in 0..24 {
                let t = k as f32 * 0.3;
                f.add_force(90.0 + 40.0 * t.sin(), 72.0 + 30.0 * t.cos(), 260.0 * t.cos(), -220.0 * t.sin(), 14.0);
                f.add_dye(90.0, 72.0, [1.0, 0.4, 0.1], 10.0);
                f.refresh_solid();
                f.enforce_boundaries();
                f.advect(1.0/60.0);
                f.compute_curl();
                f.confine_vorticity(1.0/60.0, params.vorticity);
                f.close_solid_faces();
                pre = f.max_divergence();
                f.project(iters);
                f.vel.scale(crate::math::decay(params.velocity_dissipation, 1.0/60.0));
                f.last_divergence = f.max_divergence();
            }
            println!("iters {iters}: pre {pre:.3} post {:.4} speed {:.2} ratio {:.5}", f.last_divergence(), f.max_speed(), f.last_divergence()/f.max_speed());
        }
    }
}
