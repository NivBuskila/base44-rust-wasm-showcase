//! Deterministic PRNG. Seeded explicitly so every simulation run is
//! reproducible, which is what makes the particle system testable at all.

/// xoshiro256++ — fast, tiny, and good enough for spawn jitter.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to expand one seed into a well-distributed state.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> exact f32 mantissa, no rounding to 1.0.
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniform in `[-1, 1)`.
    #[inline]
    pub fn signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// A uniformly distributed point in the unit disc.
    #[inline]
    pub fn in_disc(&mut self) -> (f32, f32) {
        let theta = self.next_f32() * core::f32::consts::TAU;
        // sqrt keeps the distribution area-uniform rather than centre-heavy.
        let r = self.next_f32().sqrt();
        (r * theta.cos(), r * theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(
            (0..8).map(|_| a.next_u64()).collect::<Vec<_>>(),
            (0..8).map(|_| b.next_u64()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn f32_stays_in_unit_interval() {
        let mut r = Rng::new(7);
        for _ in 0..100_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} out of range");
        }
    }

    #[test]
    fn f32_mean_is_roughly_half() {
        let mut r = Rng::new(9);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| r.next_f32() as f64).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn disc_points_are_inside_unit_circle() {
        let mut r = Rng::new(11);
        let mut inner = 0u32;
        let n = 50_000;
        for _ in 0..n {
            let (x, y) = r.in_disc();
            let d2 = x * x + y * y;
            assert!(d2 <= 1.0 + 1e-5, "point outside disc: {d2}");
            // Area-uniform => half the points fall inside radius 1/sqrt(2).
            if d2 <= 0.5 {
                inner += 1;
            }
        }
        let frac = inner as f32 / n as f32;
        assert!((frac - 0.5).abs() < 0.02, "not area-uniform: {frac}");
    }
}
