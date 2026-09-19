//! The per-step machinery: the resolved [`Frame`], the RK2 [`Advance`] of one
//! particle, and the [`Lane`] — a contiguous band of the pool that one thread
//! advances without locks.

use super::*;

/// Everything one `step` needs that is not per-particle, resolved once: the
/// fields, the bounds, and every tunable already folded through `dt`. Bundling
/// them keeps the per-particle helpers pure and short-signatured, and keeps the
/// clamping and `decay` calls out of the loop.
pub(super) struct Frame<'a> {
    pub(super) vel: &'a VecField,
    /// `None` when there is no usable obstacle field, which skips the test
    /// entirely for the common no-body case.
    pub(super) obstacle: Option<&'a Grid>,
    /// Grid-space -> obstacle-index scale, for the case where the obstacle
    /// field is not the same resolution as the velocity field.
    pub(super) osx: f32,
    pub(super) osy: f32,
    /// Inclusive upper bounds of the velocity grid, `(w - 1, h - 1)`.
    pub(super) maxx: f32,
    pub(super) maxy: f32,
    /// Base lifetime for respawns, before jitter.
    pub(super) life: f32,
    pub(super) dt: f32,
    /// `params.particle_drag`, clamped: the multiplier on the fluid velocity.
    pub(super) drag: f32,
    /// Fraction of a particle's own velocity surviving this step.
    pub(super) keep: f32,
    /// Fraction of `heat` surviving this step.
    pub(super) heat_keep: f32,
    /// Fraction of the gap to the speed-derived heat floor closed this step.
    pub(super) heat_rise: f32,
    /// `1 / dt`, for turning this step's displacement into a speed.
    pub(super) inv_dt: f32,
    /// `curl_influence * dt`.
    pub(super) swirl: f32,
    /// `gravity * dt`.
    pub(super) gravity: f32,
}

/// One particle's candidate next state.
#[derive(Clone, Copy)]
pub(super) struct Advance {
    pub(super) nx: f32,
    pub(super) ny: f32,
    pub(super) ox: f32,
    pub(super) oy: f32,
    /// False when the particle left the grid, landed in an obstacle, or went
    /// non-finite — all of which mean "recycle it".
    pub(super) ok: bool,
}

impl Advance {
    /// The state of a particle that is not worth integrating at all.
    const RECYCLE: Self = Self {
        nx: 0.0,
        ny: 0.0,
        ox: 0.0,
        oy: 0.0,
        ok: false,
    };
}

impl Frame<'_> {
    /// Integrates one particle for a step: RK2 midpoint advection through the
    /// fluid, plus the particle's own velocity carrying gravity, curl swirl and
    /// whatever gestures kicked into it.
    ///
    /// Pure, so the caller can run two of these back to back and let the core
    /// interleave them.
    #[inline(always)]
    pub(super) fn advance(&self, px: f32, py: f32, ovx: f32, ovy: f32) -> Advance {
        let vel = self.vel;
        let (fu1, fv1) = sample_uv(vel, px, py);

        let mut ox = ovx * self.keep;
        let mut oy = ovy * self.keep;

        // Swirl: perpendicular to the direction of travel, signed by the local
        // curl, so a particle entering a vortex is bent onto a spiral instead
        // of crossing it in a straight line. Scaled off the *unit* travel
        // direction, which keeps the magnitude proportional to omega alone —
        // taken off the unscaled velocity instead, fast particles spin and slow
        // ones (the ones actually trapped in the vortex) barely turn at all.
        if self.swirl != 0.0 {
            let tvx = self.drag * fu1 + ox;
            let tvy = self.drag * fv1 + oy;
            let len2 = tvx * tvx + tvy * tvy;
            // Same guard as `math::normalize` (len > 1e-6), but folding the
            // normalisation into the scale keeps one division off the critical
            // path instead of two.
            if len2 > 1e-12 {
                let k = curl_at(vel, px, py) * self.swirl / len2.sqrt();
                ox -= tvy * k;
                oy += tvx * k;
            }
        }
        oy += self.gravity;
        ox = tame(ox);
        oy = tame(oy);

        // RK2 midpoint: sample, step half, sample again, apply. The own
        // velocity is constant across the step so it needs no correction; only
        // the field sample does, and it is exactly the correction that keeps a
        // fast curved gesture from cutting the corner.
        let dt = self.dt;
        let hx = px + 0.5 * dt * (self.drag * fu1 + ox);
        let hy = py + 0.5 * dt * (self.drag * fv1 + oy);
        let (fu2, fv2) = sample_uv(vel, hx, hy);
        let nx = px + dt * (self.drag * fu2 + ox);
        let ny = py + dt * (self.drag * fv2 + oy);

        let ok = nx.is_finite()
            && ny.is_finite()
            && nx >= 0.0
            && nx <= self.maxx
            && ny >= 0.0
            && ny <= self.maxy
            && !self.is_solid(nx, ny);
        Advance { nx, ny, ox, oy, ok }
    }

    /// Whether grid-space `(x, y)` falls in an obstacle cell.
    ///
    /// Nearest-cell rather than bilinear: the obstacle is a hard binary
    /// predicate, the caller only needs to know which side of 0.5 it is on,
    /// and this is one load instead of four in a loop that runs 220k times a
    /// frame.
    #[inline(always)]
    pub(super) fn is_solid(&self, x: f32, y: f32) -> bool {
        match self.obstacle {
            None => false,
            Some(o) => {
                // `as usize` saturates: NaN and negatives land on 0, huge
                // values on usize::MAX, and `min` brings both into range.
                let xi = ((x * self.osx + 0.5) as usize).min(o.w - 1);
                let yi = ((y * self.osy + 0.5) as usize).min(o.h - 1);
                o.data[yi * o.w + xi] >= 0.5
            }
        }
    }
}

/// Bilinear sample of both velocity components at once.
///
/// Identical in result to two [`Grid::sample`] calls, but the clamp, floor and
/// index arithmetic are shared: the compiler cannot prove `u.w == v.w`, so
/// going through `VecField::sample` computes all of it twice, and `step` does
/// two of these per particle.
#[inline(always)]
pub(super) fn sample_uv(vel: &VecField, x: f32, y: f32) -> (f32, f32) {
    let w = vel.u.w;
    let h = vel.u.h;
    // Clamp and integer split exactly as `Grid::sample` does them, for the
    // reasons documented there: `max`/`min` fold NaN to the origin without a
    // branch, and truncation avoids `f32::floor`, which is a libm call on the
    // SSE2 baseline. Four of those per particle measured at 25 ns here, more
    // than the rest of the step put together.
    let x = x.max(0.0).min((w - 1) as f32);
    let y = y.max(0.0).min((h - 1) as f32);

    let ix0 = x as usize;
    let iy0 = y as usize;
    let tx = x - ix0 as f32;
    let ty = y - iy0 as f32;

    let ix1 = (ix0 + 1).min(w - 1);
    let iy1 = (iy0 + 1).min(h - 1);
    let row0 = iy0 * w;
    let row1 = iy1 * w;

    // Nested `lerp`, bit-for-bit what `Grid::sample` computes: the fluid
    // backtraces through the same field, and a particle that disagreed with
    // the solver in the last bit would be a needless source of drift between
    // what the dye shows and where the particles are. (A flattened
    // weighted-sum form measured no faster here, so there is nothing to trade
    // that agreement for.)
    let u = lerp(
        lerp(vel.u.data[row0 + ix0], vel.u.data[row0 + ix1], tx),
        lerp(vel.u.data[row1 + ix0], vel.u.data[row1 + ix1], tx),
        ty,
    );
    let v = lerp(
        lerp(vel.v.data[row0 + ix0], vel.v.data[row0 + ix1], tx),
        lerp(vel.v.data[row1 + ix0], vel.v.data[row1 + ix1], tx),
        ty,
    );
    (u, v)
}

/// Curl `omega = dv/dx - du/dy` at the lattice point nearest `(x, y)`.
///
/// The fluid solver keeps its own curl grid, but `step` is handed only the
/// velocity field, and recomputing it here from four loads is cheaper than the
/// eight bilinear samples an interpolated curl would cost. Matches
/// `Fluid::compute_curl` at interior lattice points, and degrades to a
/// one-sided difference on the border.
#[inline(always)]
pub(super) fn curl_at(vel: &VecField, x: f32, y: f32) -> f32 {
    let w = vel.u.w;
    let h = vel.u.h;
    let xi = ((x + 0.5) as usize).min(w - 1);
    let yi = ((y + 0.5) as usize).min(h - 1);
    let xm = xi.saturating_sub(1);
    let xp = (xi + 1).min(w - 1);
    let ym = yi.saturating_sub(1);
    let yp = (yi + 1).min(h - 1);
    // The span is 2 in the interior and 1 against a border, so the per-cell
    // normalisation is a select on an integer, not a divide. Two `divss` here
    // measured at ~7 ns per particle, a tenth of the entire step: this runs
    // once per particle per frame, 220k times, on the critical path.
    let inv_dx = if xp - xm == 2 { 0.5 } else { 1.0 };
    let inv_dy = if yp - ym == 2 { 0.5 } else { 1.0 };
    let dvdx = (vel.v.data[yi * w + xp] - vel.v.data[yi * w + xm]) * inv_dx;
    let dudy = (vel.u.data[yp * w + xi] - vel.u.data[ym * w + xi]) * inv_dy;
    dvdx - dudy
}

/// Smallest lane worth scheduling on its own thread: below this the bilinear
/// gathers cost less than the task hand-off.
pub(super) const LANE_MIN: usize = 4_096;

/// One contiguous band of the pool — every SoA stream for particles
/// `[start, start + len)` — plus the private state advancing it needs.
///
/// Lanes never overlap, so any number of them can run at once without a lock:
/// a lane reads only the shared, immutable [`Frame`] and writes only its own
/// slices. The RNG is per lane so respawn positions never serialise on one
/// generator, and the respawn budget is split up front rather than shared.
pub(super) struct Lane<'a> {
    pub(super) x: &'a mut [f32],
    pub(super) y: &'a mut [f32],
    pub(super) vx: &'a mut [f32],
    pub(super) vy: &'a mut [f32],
    pub(super) life: &'a mut [f32],
    pub(super) max_life: &'a mut [f32],
    pub(super) heat: &'a mut [f32],
    pub(super) rng: Rng,
    /// Respawns this lane may still perform, in particles.
    pub(super) credit: f32,
}

impl Lane<'_> {
    /// Advances every particle in the lane; returns how many are alive after.
    ///
    /// One linear pass, seven independent streams, no branch in the common
    /// case beyond the cull test.
    ///
    /// The per-particle cost is dominated by the two bilinear gathers RK2
    /// needs, which are dependent (the second samples the midpoint the
    /// first produced). Measured on a 2.8 GHz Xeon, one gather of both
    /// velocity components costs ~10.5 ns whether the grid fits in L1 or
    /// not, and the whole step ~58 ns; hand-pipelining two particles per
    /// iteration to overlap the chains made no difference there, so the
    /// simple form is what stays. On a core that does reorder across
    /// iterations there is nothing here to stop it: `advance` is pure and
    /// every iteration is independent.
    pub(super) fn run(&mut self, frame: &Frame) -> usize {
        let mut alive = 0usize;
        for i in 0..self.x.len() {
            let life = self.life[i] - frame.dt;
            // The one branch worth taking: a starved pool would otherwise pay
            // two bilinear gathers per frame for particles nobody can see.
            let adv = if life > 0.0 {
                frame.advance(self.x[i], self.y[i], self.vx[i], self.vy[i])
            } else {
                Advance::RECYCLE
            };
            alive += usize::from(self.commit(i, adv, life, frame));
        }
        alive
    }

    /// Writes one particle's advanced state back, or recycles it.
    ///
    /// `life` is the already-aged lifetime and `adv` the candidate state from
    /// [`Frame::advance`]. Returns whether the particle is alive afterwards, so
    /// the caller's `alive` tally costs an add rather than a second pass.
    #[inline(always)]
    pub(super) fn commit(&mut self, i: usize, adv: Advance, life: f32, f: &Frame) -> bool {
        if life > 0.0 && adv.ok {
            // Speed from the displacement actually applied, which already
            // includes fluid transport, the particle's own residual velocity
            // and the curl swirl — no extra field sample needed.
            let speed = length(
                (adv.nx - self.x[i]) * f.inv_dt,
                (adv.ny - self.y[i]) * f.inv_dt,
            );
            let floor = (speed * (1.0 / HEAT_SPEED_FULL)).min(1.0);

            self.x[i] = adv.nx;
            self.y[i] = adv.ny;
            self.vx[i] = adv.ox;
            self.vy[i] = adv.oy;
            let cooled = self.heat[i] * f.heat_keep;
            // Whichever is hotter: a spell's heat decaying away, or the
            // ambient heat this particle's own motion earns it.
            self.heat[i] = if floor > cooled {
                cooled + (floor - cooled) * f.heat_rise
            } else {
                cooled
            };
            self.life[i] = life;
            return true;
        }
        // Old age, out of the world, or inside the silhouette: all the same
        // fate, and all subject to the same respawn budget.
        if self.credit >= 1.0 {
            self.credit -= 1.0;
            return self.respawn(i, f);
        }
        // No budget: park it. Position is kept in bounds so a dead particle can
        // never render off-screen if the shader ignores `life`, and the velocity
        // is dropped so it does not resume mid-flight several frames later.
        self.life[i] = 0.0;
        self.vx[i] = 0.0;
        self.vy[i] = 0.0;
        self.heat[i] = 0.0;
        self.x[i] = clamp_finite(self.x[i], f.maxx);
        self.y[i] = clamp_finite(self.y[i], f.maxy);
        false
    }

    /// Scatters particle `i` to a fresh random position with a fresh jittered
    /// lifetime, returning whether it came back alive. Uniform over the grid
    /// rather than over the borders: the pool has to stay spatially uniform for
    /// the fluid to be legible everywhere, and edge respawns leave a visible
    /// dead band in the middle of the screen whenever the flow is slow.
    ///
    /// Every call spends one unit of budget whether or not it found a free
    /// cell. Refunding a failed attempt instead would let a silhouette that
    /// covers most of the frame put the *whole* pool through `RESPAWN_TRIES`
    /// samples every frame; spending it caps the work at the budget and merely
    /// thins the population while the obstacle is that large.
    #[cold]
    #[inline(never)]
    pub(super) fn respawn(&mut self, i: usize, frame: &Frame) -> bool {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut free = false;
        for _ in 0..RESPAWN_TRIES {
            x = self.rng.range(0.0, frame.maxx);
            y = self.rng.range(0.0, frame.maxy);
            if !frame.is_solid(x, y) {
                free = true;
                break;
            }
        }
        self.x[i] = x;
        self.y[i] = y;
        self.vx[i] = 0.0;
        self.vy[i] = 0.0;
        self.heat[i] = 0.0;
        self.max_life[i] = frame.life * self.rng.range(LIFE_JITTER.0, LIFE_JITTER.1);
        self.life[i] = if free { self.max_life[i] } else { 0.0 };
        free
    }
}

#[inline(always)]
pub(super) fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Clamps a particle's own velocity component into the representable band,
/// mapping a non-finite value to rest rather than propagating it.
#[inline(always)]
pub(super) fn tame(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED)
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn clamp_finite(v: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, max)
    } else {
        0.0
    }
}
