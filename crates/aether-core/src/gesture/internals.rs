//! Filters, geometry and the raw-landmark reader behind the tracker.

use super::*;

/// EMA coefficient for a half-life: `alpha = 1 - 2^(-dt / half_life)`.
///
/// Written through [`decay`] so it composes exactly across frame rates — ten
/// 10 ms steps and one 100 ms step leave the same fraction of the error, which
/// is the whole reason the config is expressed as a half-life rather than as a
/// per-frame coefficient.
#[inline]
pub fn ema_alpha(half_life: f32, dt: f32) -> f32 {
    if !half_life.is_finite() || half_life <= 0.0 {
        return 1.0;
    }
    1.0 - decay(core::f32::consts::LN_2 / half_life, dt)
}

/// [`ema_alpha`] with the one-euro speed-dependent cutoff applied.
#[inline]
pub(super) fn adaptive_alpha(half_life: f32, speed: f32, dt: f32) -> f32 {
    let speed = finite_or(speed, 0.0).abs();
    let adapted = (half_life / (1.0 + ADAPT_BETA * speed)).max(MIN_HALF_LIFE);
    ema_alpha(adapted, dt)
}

#[inline]
pub(super) fn clamp_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(MIN_DT, MAX_DT)
}

#[inline]
pub(super) fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

#[inline]
pub(super) fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    crate::math::length(a[0] - b[0], a[1] - b[1])
}

/// Folds an angle into `[-pi, pi]`.
#[inline]
pub(super) fn wrap_pi(a: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    if !a.is_finite() {
        return 0.0;
    }
    (a + PI).rem_euclid(TAU) - PI
}

pub(super) fn centroid(lm: &[[f32; 3]; HAND_LANDMARKS], joints: &[usize]) -> [f32; 2] {
    let mut sum = [0.0f32, 0.0];
    for &j in joints {
        sum[0] += lm[j][0];
        sum[1] += lm[j][1];
    }
    let n = joints.len().max(1) as f32;
    [sum[0] / n, sum[1] / n]
}

/// Extracts one hand's landmarks, or `None` if the slot is absent, truncated or
/// non-finite.
///
/// A single NaN coordinate rejects the whole hand: it would otherwise enter the
/// EMA state and poison every later frame, and one hand's worth of NaN in the
/// fluid blanks the screen.
pub(super) fn read_hand(packed: &[f32], base: usize) -> Option<[[f32; 3]; HAND_LANDMARKS]> {
    if base + HAND_STRIDE > packed.len() {
        return None;
    }
    // A NaN presence flag must read as absent, which `flag <= 0.5` would get
    // backwards.
    let flagged = packed[base] > 0.5;
    if !flagged {
        return None;
    }
    let mut out = [[0.0f32; 3]; HAND_LANDMARKS];
    for (l, dst) in out.iter_mut().enumerate() {
        let o = base + 4 + l * 3;
        let (x, y) = (packed[o], packed[o + 1]);
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        *dst = [x, y, finite_or(packed[o + 2], 0.0)];
    }
    Some(out)
}

/// The spell this frame's geometry and label argue for, before hysteresis.
///
/// A confident canned label wins outright; the geometric paths exist because
/// the classifier is frequently unconfident and occasionally unavailable, and a
/// hand that produces no spell at all feels broken.
pub(super) fn candidate_spell(hand: &HandState, cfg: &GestureConfig) -> Spell {
    if hand.canned_score >= cfg.min_score {
        if let Some(spell) = spell_for_label(hand.canned) {
            return spell;
        }
    }
    // Below that the features are not measurements, so the geometric fallbacks
    // must not vote — `openness` of 0 would otherwise read as a fist.
    if hand.scale < MIN_CREDIBLE_SCALE {
        return Spell::Idle;
    }
    if hand.pinch >= PINCH_COMMIT && pinch_reach(hand) >= PINCH_REACH {
        return Spell::Vortex;
    }
    if hand.openness >= OPEN_FALLBACK_REPEL {
        Spell::Repel
    } else if hand.openness <= OPEN_FALLBACK_ATTRACT {
        Spell::Attract
    } else {
        Spell::Idle
    }
}

/// How far the thumb/index midpoint sits from the wrist, in hand scales.
/// See [`PINCH_REACH`]: this is what separates a pinch from a fist.
#[inline]
pub(super) fn pinch_reach(hand: &HandState) -> f32 {
    let wrist = [hand.landmarks[LM_WRIST][0], hand.landmarks[LM_WRIST][1]];
    let mid = [
        (hand.index_tip[0] + hand.thumb_tip[0]) * 0.5,
        (hand.index_tip[1] + hand.thumb_tip[1]) * 0.5,
    ];
    distance(wrist, mid) / hand.scale.max(MIN_SCALE)
}

/// Canned label to spell. `None` for labels Aether does not bind, which fall
/// through to the geometric paths.
pub(super) fn spell_for_label(canned: CannedGesture) -> Option<Spell> {
    match canned {
        CannedGesture::ClosedFist => Some(Spell::Attract),
        CannedGesture::OpenPalm => Some(Spell::Repel),
        CannedGesture::PointingUp => Some(Spell::Ignite),
        CannedGesture::Victory => Some(Spell::Freeze),
        CannedGesture::ThumbUp => Some(Spell::Shatter),
        CannedGesture::None | CannedGesture::ThumbDown | CannedGesture::ILoveYou => None,
    }
}
