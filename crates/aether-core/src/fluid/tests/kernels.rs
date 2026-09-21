//! The Jacobi kernels against a straightforward reference, the dispatch against the scalar fallback, and their refusal to read past an undersized buffer.

use super::support::*;

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
