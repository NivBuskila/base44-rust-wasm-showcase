//! Which cells have a wall within reach: a square dilation of the obstacle
//! mask, built separably with running counts so its cost does not grow with
//! the radius.

/// `near(i)` is true when some obstacle cell (`>= 0.5`) lies within `radius`
/// cells of `i` on both axes, clamped at the grid edge.
pub(in crate::fluid) struct Reach {
    cells: Vec<u8>,
    /// The horizontal pass: an obstacle within `radius` on this row.
    rows: Vec<u8>,
    /// Per-column count of `rows` hits inside the vertical window.
    counts: Vec<u32>,
}

impl Reach {
    pub(in crate::fluid) fn new(w: usize, h: usize) -> Self {
        Self {
            cells: vec![0; w * h],
            rows: vec![0; w * h],
            counts: vec![0; w],
        }
    }

    #[inline]
    pub(in crate::fluid) fn near(&self, i: usize) -> bool {
        self.cells[i] != 0
    }

    /// Rebuilds the dilation of `obstacle` by `radius`, in `O(w * h)`.
    pub(in crate::fluid) fn build(&mut self, obstacle: &[f32], w: usize, h: usize, radius: usize) {
        let n = w * h;
        if w == 0 || h == 0 || obstacle.len() < n || self.cells.len() < n {
            self.cells.fill(0);
            return;
        }
        // Past the grid's size a larger radius changes nothing, and capping it
        // keeps the window arithmetic from overflowing.
        let r = radius.min(w.max(h));

        // Row by row, slide an `[x - r, x + r]` window and count what enters
        // and leaves it.
        for (src, out) in obstacle[..n].chunks(w).zip(self.rows.chunks_mut(w)) {
            let mut inside = src[..(r + 1).min(w)].iter().filter(|&&o| o >= 0.5).count();
            for x in 0..w {
                out[x] = u8::from(inside > 0);
                if x + 1 + r < w && src[x + 1 + r] >= 0.5 {
                    inside += 1;
                }
                if x >= r && src[x - r] >= 0.5 {
                    inside -= 1;
                }
            }
        }

        // The same window down the columns, carried as one count per column so
        // the pass still walks memory row by row.
        self.counts.fill(0);
        for row in self.rows.chunks(w).take(r + 1) {
            for (c, &hit) in self.counts.iter_mut().zip(row) {
                *c += u32::from(hit);
            }
        }
        for y in 0..h {
            let out = &mut self.cells[y * w..(y + 1) * w];
            for (o, &c) in out.iter_mut().zip(&self.counts) {
                *o = u8::from(c > 0);
            }
            if y + 1 + r < h {
                let entering = &self.rows[(y + 1 + r) * w..(y + 2 + r) * w];
                for (c, &hit) in self.counts.iter_mut().zip(entering) {
                    *c += u32::from(hit);
                }
            }
            if y >= r {
                let leaving = &self.rows[(y - r) * w..(y - r + 1) * w];
                for (c, &hit) in self.counts.iter_mut().zip(leaving) {
                    *c -= u32::from(hit);
                }
            }
        }
    }
}
