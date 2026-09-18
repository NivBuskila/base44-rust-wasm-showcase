//! # Visual metrics — the parts of "how it looks" that are measurable
//!
//! The engine's numeric tests answer "did the solver stay finite and stable".
//! They say nothing about the questions tuning actually turns on: is the dye a
//! flat saturated blob or a field of filaments, is there swirl left after the
//! dissipation eats the momentum, did the projection stop converging. Those are
//! visual judgements, but they are not *only* visual — each one has a scalar
//! behind it that can be computed from the same [`Fluid`] the renderer uploads.
//!
//! This module computes those scalars so a tuning change can be measured before
//! anyone opens a browser tab. It deliberately reads the **dye grids**, which
//! are the renderer's input: the shader adds bloom and tone mapping on top, so
//! these numbers describe the structure being handed to the shader, not the
//! final pixels. That is the right level — bloom cannot invent filaments that
//! are not in the dye, and it cannot rescue a channel already pinned at the
//! ceiling. What it cannot tell you is whether the result is *pretty*; that
//! judgement stays with a human in a standalone tab.
//!
//! Nothing here runs in the browser build's hot path; it is a measurement tool
//! for tests and benches.

use crate::field::Grid;
use crate::fluid::{Fluid, DYE_CHANNELS};

/// The dye value the renderer's tone map treats as full brightness. A cell at
/// or above this is white on screen no matter how much more dye is piled into
/// it, so the fraction of such cells is the "blown out" measure.
const TONE_CEILING: f32 = 1.0;

/// Below this a cell is black once the dye texture is quantised to 8 bits.
const VISIBLE_DYE: f32 = 1.0 / 255.0;

/// A snapshot of the visually meaningful structure in a fluid state.
///
/// Every field is resolution independent — fractions, means and ratios rather
/// than sums — so a metric measured on a small test grid stays comparable to
/// the full simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualMetrics {
    /// Fraction of cells whose brightest dye channel is at or past the tone-map
    /// ceiling. This is the blown-out-blob detector: past a few percent the
    /// screen has flat white regions where the solver's detail is invisible.
    pub saturated: f32,
    /// Mean dye luminance. Low means a dim, empty screen; high on its own is
    /// not a problem unless `saturated` follows it up.
    pub brightness: f32,
    /// Mean absolute luminance gradient divided by mean luminance — how sharp
    /// the dye is relative to how much of it there is. This is the filament
    /// measure, and both an over-smoothing advection scheme and a saturated
    /// field destroy it, from opposite directions.
    pub contrast: f32,
    /// Mean absolute curl of the velocity field: how much rotational structure
    /// is left. Velocity dissipation trades this away, which is exactly the
    /// tradeoff that needed a number attached to it.
    pub swirl: f32,
    /// Fraction of cells carrying visible dye at all — how much of the frame is
    /// doing something rather than sitting black.
    pub coverage: f32,
    /// Residual divergence left by the pressure solve, straight from the
    /// solver. Rising values mean the projection is no longer keeping up, which
    /// shows up on screen as dye that stretches and smears instead of swirling.
    pub divergence: f32,
    /// Peak cell speed, in cells/second. Included so a metrics run also reports
    /// how close the field is to the clamp that keeps it stable.
    pub max_speed: f32,
}

impl VisualMetrics {
    /// Measures a fluid state. Cheap enough to call every frame of a test run,
    /// but it walks the whole grid, so it does not belong in the render loop.
    pub fn measure(fluid: &Fluid) -> Self {
        let (w, h) = (fluid.width(), fluid.height());
        let dye = fluid.dye();
        let cells = (w * h) as f32;

        let mut saturated = 0u32;
        let mut covered = 0u32;
        let mut lum_sum = 0.0f32;
        for i in 0..w * h {
            let (r, g, b) = (dye[0].data[i], dye[1].data[i], dye[2].data[i]);
            let peak = r.max(g).max(b);
            if peak >= TONE_CEILING {
                saturated += 1;
            }
            if peak >= VISIBLE_DYE {
                covered += 1;
            }
            lum_sum += luminance(r, g, b);
        }
        let brightness = lum_sum / cells;

        // Gradient magnitude over the luminance a viewer perceives, rather than
        // per channel: three channels moving together is one edge, not three.
        let lum = luminance_grid(dye, w, h);
        let mut grad_sum = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let (gx, gy) = lum.gradient(x, y);
                grad_sum += (gx * gx + gy * gy).sqrt();
            }
        }
        // Normalising by brightness is what makes this a *structure* measure:
        // scaling the whole field up scales its gradients with it and leaves
        // contrast unchanged, so only real edges move the number. The epsilon
        // keeps an empty field at 0 instead of NaN.
        let contrast = if brightness > 1e-6 {
            (grad_sum / cells) / brightness
        } else {
            0.0
        };

        let curl = fluid.curl();
        let mut curl_sum = 0.0f32;
        for i in 0..w * h {
            curl_sum += curl.data[i].abs();
        }

        Self {
            saturated: saturated as f32 / cells,
            brightness,
            contrast,
            swirl: curl_sum / cells,
            coverage: covered as f32 / cells,
            divergence: fluid.last_divergence(),
            max_speed: fluid.max_speed(),
        }
    }
}

/// Rec. 709 luma. The renderer's tone map is luminance driven, so the same
/// weighting is what decides whether a cell reads as bright on screen.
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn luminance_grid(dye: &[Grid; DYE_CHANNELS], w: usize, h: usize) -> Grid {
    let mut out = Grid::new(w, h);
    for i in 0..w * h {
        out.data[i] = luminance(dye[0].data[i], dye[1].data[i], dye[2].data[i]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Params;

    /// A driven run that does not depend on the gesture layer: a stirring force
    /// that orbits the centre while injecting dye, which is the motion a hand
    /// circling in front of the camera produces. Deterministic, so metrics from
    /// two runs are directly comparable.
    fn stirred(params: &Params, steps: usize) -> Fluid {
        const DT: f32 = 1.0 / 60.0;
        let mut fluid = Fluid::new(96, 54);
        let (cx, cy) = (48.0, 27.0);
        for step in 0..steps {
            let t = step as f32 * DT;
            let angle = t * 3.0;
            let (sx, sy) = (cx + 12.0 * angle.cos(), cy + 12.0 * angle.sin());
            // Force tangent to the orbit: this is what makes it stir rather
            // than just push dye outward.
            let (fx, fy) = (-angle.sin() * 900.0, angle.cos() * 900.0);
            fluid.add_force(sx, sy, fx, fy, 6.0);
            fluid.add_dye(sx, sy, [1.4, 0.7, 0.25], 5.0);
            fluid.step(DT, params);
        }
        fluid
    }

    #[test]
    fn an_empty_field_measures_as_empty_not_nan() {
        let m = VisualMetrics::measure(&Fluid::new(32, 18));
        assert_eq!(m.saturated, 0.0);
        assert_eq!(m.coverage, 0.0);
        // The guard in `contrast`: 0/0 must not escape as NaN.
        assert_eq!(m.contrast, 0.0);
        assert!(m.brightness.abs() < 1e-9);
    }

    #[test]
    fn contrast_sees_structure_not_overall_brightness() {
        // The same pattern at two exposures must report the same contrast, or
        // the metric is measuring brightness and cannot compare tunings that
        // change how much dye ends up in the field.
        let stripes = |scale: f32| {
            let mut f = Fluid::new(32, 18);
            for x in (0..32).step_by(2) {
                for y in 0..18 {
                    f.add_dye(x as f32, y as f32, [scale, scale, scale], 0.5);
                }
            }
            VisualMetrics::measure(&f)
        };
        let dim = stripes(0.2);
        let bright = stripes(0.6);
        assert!(bright.brightness > dim.brightness * 2.0, "exposure did not change");
        assert!(
            (bright.contrast - dim.contrast).abs() / dim.contrast < 0.05,
            "contrast tracked brightness: {} vs {}",
            dim.contrast,
            bright.contrast
        );
        // A striped field is nothing but edges; a single wide blob has few.
        let mut flat = Fluid::new(32, 18);
        flat.add_dye(16.0, 9.0, [0.4, 0.4, 0.4], 64.0);
        assert!(
            VisualMetrics::measure(&flat).contrast < dim.contrast * 0.5,
            "a flat blob scored as structured"
        );
    }

    /// The regression this module exists for: the tuning shipped in
    /// `Params::default` has to keep producing a *structured* frame — not a
    /// white blob, not an empty screen.
    #[test]
    fn shipped_tuning_produces_a_structured_frame() {
        let m = VisualMetrics::measure(&stirred(&Params::default(), 240));
        assert!(
            m.saturated < 0.02,
            "frame is blowing out: {:.1}% of cells at the tone ceiling",
            m.saturated * 100.0
        );
        assert!(
            m.coverage > 0.4,
            "frame is nearly empty: {:.1}% coverage",
            m.coverage * 100.0
        );
        assert!(m.contrast > 0.25, "dye has no filament structure: {:.3}", m.contrast);
        assert!(m.swirl > 5.0, "no rotational structure left: {:.3}", m.swirl);
        // `last_divergence` is an absolute residual in the same units as the
        // velocity, so it scales with how hard the field is being driven — the
        // meaningful check is the residual *relative* to the flow it is left in
        // (measured at ~1%; 5% is the point where smearing becomes visible).
        assert!(
            m.divergence < m.max_speed * 0.05,
            "pressure solve is not keeping up: residual {:.3} against {:.1} cells/s",
            m.divergence,
            m.max_speed
        );
    }

    /// Guards the tradeoff behind `velocity_dissipation` in both directions, so
    /// a future change to that value faces numbers instead of taste: the
    /// original 0.15 is what let a held gesture ratchet the field up, and a
    /// heavy 1.0 is what flattens the swirl out of it.
    #[test]
    fn velocity_dissipation_trades_swirl_against_blowout() {
        let with = |vd: f32| {
            let params = Params {
                velocity_dissipation: vd,
                ..Params::default()
            };
            VisualMetrics::measure(&stirred(&params, 240))
        };
        let low = with(0.15);
        let shipped = with(Params::default().velocity_dissipation);
        let high = with(1.0);

        assert!(
            low.max_speed > shipped.max_speed,
            "weak decay did not accumulate more momentum: {:.1} vs {:.1}",
            low.max_speed,
            shipped.max_speed
        );
        assert!(
            high.swirl < shipped.swirl,
            "heavy decay did not cost swirl: {:.3} vs {:.3}",
            high.swirl,
            shipped.swirl
        );
    }
}
