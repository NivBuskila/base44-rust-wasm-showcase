//! The finite guards every write goes through. A non-finite value reaching the
//! pool is not a rounding problem, it is a particle that never renders again, so
//! each of these maps it to rest or to the origin instead of propagating it.

use crate::particles::*;

#[inline(always)]
pub(in crate::particles) fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Clamps a particle's own velocity component into the representable band,
/// mapping a non-finite value to rest rather than propagating it.
#[inline(always)]
pub(in crate::particles) fn tame(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-MAX_OWN_SPEED, MAX_OWN_SPEED)
    } else {
        0.0
    }
}

#[inline(always)]
pub(in crate::particles) fn clamp_finite(v: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, max)
    } else {
        0.0
    }
}
