//! Viscous diffusion and vorticity confinement: the Jacobi viscosity solve
//! (skipped at zero viscosity), the curl field, and the confinement force that
//! re-injects the swirl numerical dissipation eats.

use super::*;

impl Fluid {

    pub(super) fn diffuse(&mut self, dt: f32, viscosity: f32) {
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
    pub(super) fn compute_curl(&mut self) {
        let (w, h) = (self.w, self.h);
        // Taken out of `self` so the rows can be written while the velocity and
        // the wall mask are read through `&Self` — see `interior_rows`.
        let mut curl = core::mem::take(&mut self.curl.data);
        curl.fill(0.0);
        let this: &Self = self;
        interior_rows(&mut curl, w, h, |y, row| {
            for x in 1..w - 1 {
                let i = y * w + x;
                if this.solid.data[i] >= 0.5 {
                    continue;
                }
                let dvdx = this.vel.v.data[i + 1] - this.vel.v.data[i - 1];
                let dudy = this.vel.u.data[i + w] - this.vel.u.data[i - w];
                row[x] = 0.5 * (dvdx - dudy);
            }
        });
        self.curl.data = curl;
    }

    /// Fedkiw-style confinement: `f = eps * (N x omega)` with
    /// `N = grad|omega| / |grad|omega||`, i.e. a force that spins each cell
    /// back towards the nearest vorticity peak.
    pub(super) fn confine_vorticity(&mut self, dt: f32, strength: f32) {
        let (w, h) = (self.w, self.h);
        let mut u = core::mem::take(&mut self.vel.u.data);
        let mut v = core::mem::take(&mut self.vel.v.data);
        let this: &Self = self;
        interior_row_pairs(&mut u, &mut v, w, h, |y, ur, vr| {
            for x in 1..w - 1 {
                let i = y * w + x;
                if this.solid.data[i] >= 0.5 {
                    continue;
                }
                let gx = 0.5 * (this.curl.data[i + 1].abs() - this.curl.data[i - 1].abs());
                let gy = 0.5 * (this.curl.data[i + w].abs() - this.curl.data[i - w].abs());
                // Returns (0, 0) on a vanishing gradient, so flat regions get
                // no force instead of a normalised division by ~0.
                let (nx, ny) = normalize(gx, gy);
                let f = strength * this.curl.data[i] * dt;
                ur[x] += bounded_impulse(ny * f, MAX_CONFINE_IMPULSE);
                vr[x] += bounded_impulse(-nx * f, MAX_CONFINE_IMPULSE);
            }
        });
        self.vel.u.data = u;
        self.vel.v.data = v;
    }
}
