//! Building the Gaussian pyramid and carrying a coarse solution down to the
//! next finer level.

use super::work::Work;
use super::{DOWN_BLUR, PRE_BLUR};
use crate::field::Grid;

/// Fills `pyr` (index 0 = full resolution) with a Gaussian pyramid of `src`.
pub(super) fn build_pyramid(src: &Grid, pyr: &mut [Grid], work: &mut [Work]) {
    pyr[0].resample_from(src);
    pyr[0].blur(&mut work[0].tmp, PRE_BLUR);
    for k in 1..pyr.len() {
        // Blur a copy, not the level itself: the unsmoothed level is the image
        // its own solve runs on, and over-filtering it would throw away the
        // fine detail that is the whole point of having a finest level.
        let src_level = {
            let w = &mut work[k - 1];
            w.smooth.data.copy_from_slice(&pyr[k - 1].data);
            w.smooth.blur(&mut w.tmp, DOWN_BLUR);
            &w.smooth
        };
        pyr[k].resample_from(src_level);
    }
}

/// Upsamples the coarse solution into `fine` as its initial guess.
pub(super) fn prolong(fine: &mut Work, coarse: &Work) {
    // The scale factor is the classic bug here: a displacement of one coarse
    // cell is `(fine - 1) / (coarse - 1)` fine cells, and `resample_from` maps
    // the lattices with exactly that ratio. Drop it and every level halves the
    // motion it inherited.
    let sx = (fine.u.w - 1) as f32 / (coarse.u.w - 1) as f32;
    let sy = (fine.u.h - 1) as f32 / (coarse.u.h - 1) as f32;
    fine.u.resample_from(&coarse.u);
    fine.u.scale(sx);
    fine.v.resample_from(&coarse.v);
    fine.v.scale(sy);
}
