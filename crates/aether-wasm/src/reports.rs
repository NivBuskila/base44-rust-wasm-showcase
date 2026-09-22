//! Everything the HUD reads back: the packed stats array, latched spell names,
//! the combo and duet spellbooks with their progress, practice mode and the
//! lens-rush staging.

use wasm_bindgen::prelude::*;

use crate::AetherEngine;

#[wasm_bindgen]
impl AetherEngine {
    // ----------------------------------------------------------------- stats

    /// Packed stats, [`STATS_LEN`] floats. Indices are mirrored in
    /// `web/src/constants.ts` as `STAT`.
    ///
    /// `0` fluid energy, `1` max speed, `2` residual divergence,
    /// `3` motion energy, `4` flow dx, `5` flow dy, `6` particles alive,
    /// `7` mask coverage, `8` mask present, `9` hands present,
    /// `10` pose present, `11` time scale, `12` bursts, `13` NaN repairs,
    /// `14` frame, `15` ambient mode.
    pub fn stats(&self) -> Vec<f32> {
        let s = self.inner.stats();
        vec![
            s.fluid_energy,
            s.fluid_max_speed,
            s.fluid_divergence,
            s.motion_energy,
            s.flow_dx,
            s.flow_dy,
            s.particles_alive as f32,
            s.mask_coverage,
            if s.mask_present { 1.0 } else { 0.0 },
            s.hands_present as f32,
            if s.pose_present { 1.0 } else { 0.0 },
            s.time_scale,
            s.bursts as f32,
            s.nan_repairs as f32,
            s.frame as f32,
            if s.ambient { 1.0 } else { 0.0 },
        ]
    }

    /// Latched spell name for hand slot 0 or 1.
    pub fn spell_name(&self, slot: usize) -> String {
        self.inner.spell_name(slot).to_string()
    }

    /// The gesture-sequence spellbook, as `name:step,step;name:step,...`.
    ///
    /// Serialised from `aether_core::combo::COMBOS` rather than duplicated in
    /// TypeScript: the HUD teaches the sequences, and a spellbook that drifts
    /// from the recogniser is worse than none at all.
    pub fn combo_book(&self) -> String {
        aether_core::combo::COMBOS
            .iter()
            .map(|c| {
                let steps: Vec<&str> = c.steps.iter().map(|s| s.spell.name()).collect();
                format!("{}:{}", c.name, steps.join(","))
            })
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Sequence progress: `0` combo index (-1 when idle), `1` steps matched,
    /// `2` charge 0..1 through the current step, `3` combo that just fired
    /// (-1 when none).
    pub fn combo_progress(&self) -> Vec<f32> {
        let p = self.inner.combo_progress();
        let idx = |v: usize| if v == usize::MAX { -1.0 } else { v as f32 };
        vec![idx(p.combo), p.matched as f32, p.charge, idx(p.fired)]
    }

    /// The two-hand duet spellbook, as `name:step,step;...`. Steps are spell
    /// names joined by `+` (both hands at once) or a motion word.
    pub fn duet_book(&self) -> String {
        aether_core::duet::DUETS
            .iter()
            .map(|d| format!("{}:{}", d.name, d.steps.join(",")))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Duet progress: `0` duet index being wound up (-1 when idle), `1` charge
    /// 0..1, `2` duet that just fired (-1 when none).
    pub fn duet_progress(&self) -> Vec<f32> {
        let p = self.inner.duet_progress();
        let idx = |v: usize| if v == usize::MAX { -1.0 } else { v as f32 };
        vec![idx(p.active), p.charge, idx(p.fired)]
    }

    /// Practice mode for the tutorial: single gestures still cast, sequences
    /// and duets are held back.
    pub fn set_practice(&mut self, on: bool) {
        self.inner.set_practice(on);
    }

    /// Packed lens-rush staging for the renderer: `[x, y, progress, power]`,
    /// with `power == 0` meaning nothing is in flight.
    pub fn rush_state(&self) -> Vec<f32> {
        let r = self.inner.rush_state();
        vec![r.at[0], r.at[1], r.progress, r.power, r.kind.as_f32()]
    }
}
