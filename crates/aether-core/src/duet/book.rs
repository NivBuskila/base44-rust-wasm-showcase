//! The duet spellbook: what a two-hand cast does and the three castable
//! entries, with the HUD's own step wording. Data only — recognition is the
//! parent's `DuetTracker`.

/// What a duet does, resolved by [`crate::spells`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuetEffect {
    /// A continuous jet from the open palm into the fist.
    Tide,
    /// A shockwave from the point where the hands met.
    Clap,
    /// Two squeezed fists let go together: a supernova from between them.
    BigBang,
}

/// A castable duet.
#[derive(Clone, Copy, Debug)]
pub struct DuetDef {
    /// Stable identifier, also the HUD label.
    pub name: &'static str,
    /// HUD instructions. Spell names joined by `+` are both hands at once;
    /// anything else is a motion word the HUD shows verbatim.
    pub steps: &'static [&'static str],
    pub effect: DuetEffect,
}

/// The duet spellbook. Index order is what the HUD and
/// [`DuetProgress`] refer to.
pub static DUETS: &[DuetDef] = &[
    DuetDef {
        name: "tide",
        steps: &["attract+repel"],
        effect: DuetEffect::Tide,
    },
    DuetDef {
        name: "clap",
        steps: &["repel+repel", "clap"],
        effect: DuetEffect::Clap,
    },
    DuetDef {
        name: "big bang",
        steps: &["attract+attract", "squeeze", "release+release"],
        effect: DuetEffect::BigBang,
    },
];
