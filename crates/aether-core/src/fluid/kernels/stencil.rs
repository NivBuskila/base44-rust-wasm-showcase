//! The five-point row window both Jacobi sweeps read through, plus the two
//! per-cell updates in their scalar form — the definition the SIMD lanes are
//! diffed against.

use crate::fluid::PRESSURE_OMEGA;

/// Substitutes the centre value for a wall neighbour.
///
/// For pressure this is the Neumann condition `dp/dn = 0` on a solid face.
/// Reading the wall cell's own pressure instead is precisely the bug that lets
/// fluid pour through the body silhouette.
#[inline]
pub(in crate::fluid) fn wall_pick(value: f32, solid: f32, centre: f32) -> f32 {
    if solid >= 0.5 {
        centre
    } else {
        value
    }
}

/// The row windows one output row of a five-point sweep reads.
///
/// Slicing the rows out up front is not cosmetic: indexing `w`-long slices with
/// `x in 1..w - 1` lets the compiler discharge the bounds checks, which is
/// worth roughly 3x on the pressure sweep over indexing the flat grid.
pub(in crate::fluid) struct StencilRows<'a> {
    pub(super) up: &'a [f32],
    pub(super) mid: &'a [f32],
    pub(super) down: &'a [f32],
    pub(super) solid_up: &'a [f32],
    pub(super) solid_mid: &'a [f32],
    pub(super) solid_down: &'a [f32],
    pub(super) rhs: &'a [f32],
}

impl<'a> StencilRows<'a> {
    /// Windows around row `y`. Requires `1 <= y <= h - 2` and slices of at
    /// least `w * h`.
    #[inline]
    pub(in crate::fluid) fn new(
        field: &'a [f32],
        solid: &'a [f32],
        rhs: &'a [f32],
        w: usize,
        y: usize,
    ) -> Self {
        let row = y * w;
        Self {
            up: &field[row - w..row],
            mid: &field[row..row + w],
            down: &field[row + w..row + 2 * w],
            solid_up: &solid[row - w..row],
            solid_mid: &solid[row..row + w],
            solid_down: &solid[row + w..row + 2 * w],
            rhs: &rhs[row..row + w],
        }
    }

    /// Damped Jacobi pressure update for one cell of this row.
    ///
    /// The wall test comes last, as a select on an already-computed value,
    /// rather than as an early return. That is not a style choice: an early
    /// return here costs 4.2 ns/cell against 1.2 ns/cell for the select,
    /// because the branch is data-dependent and blocks vectorisation.
    #[inline]
    pub(in crate::fluid) fn pressure(&self, x: usize) -> f32 {
        let pc = self.mid[x];
        let l = wall_pick(self.mid[x - 1], self.solid_mid[x - 1], pc);
        let r = wall_pick(self.mid[x + 1], self.solid_mid[x + 1], pc);
        let u = wall_pick(self.up[x], self.solid_up[x], pc);
        let d = wall_pick(self.down[x], self.solid_down[x], pc);
        // Grouped exactly as the SIMD path adds it, so the two are bit-equal
        // and the cross-check test can use a tight tolerance.
        let sweep = ((l + r) + (u + d) - self.rhs[x]) * 0.25;
        let relaxed = pc + PRESSURE_OMEGA * (sweep - pc);
        if self.solid_mid[x] >= 0.5 {
            0.0
        } else {
            relaxed
        }
    }

    /// Jacobi diffusion update for one cell of this row.
    #[inline]
    pub(in crate::fluid) fn diffuse(&self, x: usize, a: f32, inv: f32) -> f32 {
        // A wall neighbour contributes its own (zero) velocity rather than a
        // mirrored copy of the centre: a wall *drags* the fluid beside it,
        // which is the entire physical content of no-slip.
        let sum = (self.mid[x - 1] + self.mid[x + 1]) + (self.up[x] + self.down[x]);
        let relaxed = (self.rhs[x] + a * sum) * inv;
        if self.solid_mid[x] >= 0.5 {
            0.0
        } else {
            relaxed
        }
    }
}
