//! Fixtures: the procedural texture, its warps, the interior mean and the shared recovery assertion.

pub(super) use crate::flow::*;

pub(super) use crate::rng::Rng;
pub(super) use core::f32::consts::TAU;
impl Texture {
    pub(super) fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut waves = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 3];
        // Wavelengths in full-resolution cells; the longest survives three
        // decimations, the shortest gives the finest level something to do.
        let spec = [(30.0f32, 0.45f32), (14.0, 0.30), (8.0, 0.25)];
        // The seed picks the overall orientation, but the three waves are
        // then spread evenly: a texture whose gradients all point the same
        // way leaves the flow component across them unobservable (the
        // aperture problem), and the ground truth would be unrecoverable
        // for reasons that have nothing to do with the solver.
        let base = rng.next_f32() * TAU;
        for (i, (slot, &(wavelength, amp))) in waves.iter_mut().zip(spec.iter()).enumerate() {
            let theta = base + i as f32 * TAU / 3.0;
            let k = TAU / wavelength;
            *slot = (k * theta.cos(), k * theta.sin(), rng.next_f32() * TAU, amp);
        }
        let norm = waves.iter().map(|w| w.3.abs()).sum::<f32>().max(1e-6);
        Self { waves, norm }
    }
    pub(super) fn at(&self, x: f32, y: f32) -> f32 {
        let s: f32 = self
            .waves
            .iter()
            .map(|&(kx, ky, phase, amp)| amp * (kx * x + ky * y + phase).sin())
            .sum();
        0.5 + 0.5 * s / self.norm
    }
    pub(super) fn render(
        &self,
        w: usize,
        h: usize,
        map: impl Fn(f32, f32) -> (f32, f32),
    ) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = map(x as f32, y as f32);
                out[y * w + x] = (self.at(sx, sy).clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        out
    }
    pub(super) fn still(&self, w: usize, h: usize) -> Vec<u8> {
        self.render(w, h, |x, y| (x, y))
    }
    /// Content shifted by `(dx, dy)` cells, so the true flow is `(dx, dy)`.
    pub(super) fn translated(&self, w: usize, h: usize, dx: f32, dy: f32) -> Vec<u8> {
        self.render(w, h, |x, y| (x - dx, y - dy))
    }
    /// Content magnified by `s` about the frame centre: outward flow.
    pub(super) fn zoomed(&self, w: usize, h: usize, s: f32) -> Vec<u8> {
        let cx = (w - 1) as f32 * 0.5;
        let cy = (h - 1) as f32 * 0.5;
        self.render(w, h, |x, y| (cx + (x - cx) / s, cy + (y - cy) / s))
    }
}
pub(super) const W: usize = 128;
pub(super) const H: usize = 72;
pub(super) const DT: f32 = 1.0 / 30.0;

/// Deterministic band-limited test texture.
///
/// Sampled analytically so a translated or zoomed copy is exact instead of
/// resampled, and built from one wave per octave so that *every* pyramid
/// level still has real gradients: a single high-frequency octave is gone
/// by the second decimation and the coarse solve would have nothing to
/// lock onto, which makes a "flow is nonzero" assertion vacuous.
pub(super) struct Texture {
    waves: [(f32, f32, f32, f32); 3],
    norm: f32,
}

pub(super) fn flow_of(prev: &[u8], cur: &[u8], dt: f32) -> OpticalFlow {
    let mut f = OpticalFlow::new(W, H, FlowConfig::default());
    f.update(prev, dt);
    f.update(cur, dt);
    f
}

/// Mean flow over the interior, away from the border where the clamped
/// warp invents data that was never in frame.
pub(super) fn mean_interior(f: &VecField, margin: usize) -> (f32, f32) {
    let mut su = 0.0f64;
    let mut sv = 0.0f64;
    let mut n = 0u32;
    for y in margin..f.h() - margin {
        for x in margin..f.w() - margin {
            let (u, v) = f.get(x, y);
            su += u as f64;
            sv += v as f64;
            n += 1;
        }
    }
    let inv = 1.0 / n.max(1) as f64;
    ((su * inv) as f32, (sv * inv) as f32)
}

pub(super) fn assert_recovers(prev: &[u8], cur: &[u8], want: (f32, f32)) {
    let f = flow_of(prev, cur, DT);
    let (u, v) = mean_interior(f.flow(), 16);
    let (wu, wv) = (want.0 / DT, want.1 / DT);
    let scale = (wu * wu + wv * wv).sqrt();
    assert!(
        (u - wu).abs() < 0.35 * scale,
        "u {u} should be near {wu} (v {v})"
    );
    assert!(
        (v - wv).abs() < 0.35 * scale,
        "v {v} should be near {wv} (u {u})"
    );
    // Sign is the part a magnitude-only test would never catch.
    if wu.abs() > 1.0 {
        assert!(u.signum() == wu.signum(), "u sign flipped: {u} vs {wu}");
    }
    if wv.abs() > 1.0 {
        assert!(v.signum() == wv.signum(), "v sign flipped: {v} vs {wv}");
    }
}
