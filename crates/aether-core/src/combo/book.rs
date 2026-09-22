//! The combo spellbook: a sequence's shape and the three castable entries.
//! Data only — recognition is the parent's `ComboTracker`.

use crate::gesture::Spell;

/// One element of a sequence: a spell, and how long it must be held.
///
/// `hold` of 0 means "as soon as it commits"; a positive hold on a final step is
/// what makes a charged combo feel like winding up rather than a twitch.
#[derive(Clone, Copy, Debug)]
pub struct ComboStep {
    pub spell: Spell,
    pub hold: f32,
}

impl ComboStep {
    const fn new(spell: Spell, hold: f32) -> Self {
        Self { spell, hold }
    }
}

/// What a completed sequence does, resolved by [`crate::spells`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComboEffect {
    /// A bright expanding ring: the easy, always-lands combo.
    Nova,
    /// A wide swirling storm around the hand.
    Tempest,
    /// The charged one: a huge shell, a hard outward kick and a heat flash.
    Supernova,
}

/// A castable sequence.
#[derive(Clone, Copy, Debug)]
pub struct ComboDef {
    /// Stable identifier, also the HUD label.
    pub name: &'static str,
    pub steps: &'static [ComboStep],
    pub effect: ComboEffect,
}

/// The spellbook. Three entries on purpose: two steps are learnable by watching
/// someone else do it, four are not, and a caster who cannot remember the
/// sequence never reaches the part where the recognition is impressive.
pub static COMBOS: &[ComboDef] = &[
    ComboDef {
        name: "nova",
        steps: &[
            ComboStep::new(Spell::Attract, 0.1),
            ComboStep::new(Spell::Repel, 0.05),
        ],
        effect: ComboEffect::Nova,
    },
    ComboDef {
        name: "tempest",
        steps: &[
            ComboStep::new(Spell::Repel, 0.1),
            ComboStep::new(Spell::Vortex, 0.12),
        ],
        effect: ComboEffect::Tempest,
    },
    ComboDef {
        name: "supernova",
        steps: &[
            ComboStep::new(Spell::Attract, 0.08),
            ComboStep::new(Spell::Freeze, 0.12),
            ComboStep::new(Spell::Attract, 0.3),
        ],
        effect: ComboEffect::Supernova,
    },
];
