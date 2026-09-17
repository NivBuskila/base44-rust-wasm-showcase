//! Small scalar helpers. Kept dependency-free and `#[inline]` so the fluid and
//! particle hot loops pay nothing for using them.

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
pub fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Hermite ease between `edge0` and `edge1`, matching GLSL `smoothstep`.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

/// Frame-rate independent exponential decay: the fraction of a quantity that
/// survives `dt` seconds given a per-second decay `rate`.
///
/// Using this instead of `1.0 - rate * dt` keeps the look identical at 30 and
/// 144 fps, and never goes negative for large `dt`.
#[inline]
pub fn decay(rate: f32, dt: f32) -> f32 {
    (-rate * dt).exp()
}

#[inline]
pub fn length(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Normalises `(x, y)`, returning `(0, 0)` for a zero-length input.
#[inline]
pub fn normalize(x: f32, y: f32) -> (f32, f32) {
    let len = length(x, y);
    if len > 1e-6 {
        (x / len, y / len)
    } else {
        (0.0, 0.0)
    }
}

/// Fully saturated HSV -> linear-ish RGB, `hue` in turns (wraps automatically).
pub fn hue_to_rgb(hue: f32) -> [f32; 3] {
    let h = (hue.fract() + 1.0).fract() * 6.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_is_clamped_and_monotonic() {
        assert_eq!(smoothstep(0.0, 1.0, -5.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 5.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        let mut prev = -1.0;
        for i in 0..=20 {
            let v = smoothstep(0.0, 1.0, i as f32 / 20.0);
            assert!(v >= prev, "not monotonic at {i}");
            prev = v;
        }
    }

    #[test]
    fn decay_is_framerate_independent() {
        // One 100 ms step must equal ten 10 ms steps.
        let one = decay(2.0, 0.1);
        let mut many = 1.0;
        for _ in 0..10 {
            many *= decay(2.0, 0.01);
        }
        assert!((one - many).abs() < 1e-6, "{one} vs {many}");
    }

    #[test]
    fn decay_never_negative_for_large_dt() {
        assert!(decay(5.0, 100.0) >= 0.0);
    }

    #[test]
    fn hue_wraps_and_stays_in_gamut() {
        for i in -30..30 {
            let rgb = hue_to_rgb(i as f32 / 7.0);
            for c in rgb {
                assert!((0.0..=1.0).contains(&c), "channel {c} out of gamut");
            }
            // A fully saturated hue always has one channel at full and one at zero.
            let max = rgb.iter().cloned().fold(f32::MIN, f32::max);
            assert!((max - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn normalize_handles_zero() {
        assert_eq!(normalize(0.0, 0.0), (0.0, 0.0));
        let (x, y) = normalize(3.0, 4.0);
        assert!((length(x, y) - 1.0).abs() < 1e-6);
    }
}
