//! The gesture and spell vocabularies: MediaPipe's canned labels and the
//! spells the engine casts from them, with their names and dye hues.

/// MediaPipe's canned gesture labels, in classifier index order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CannedGesture {
    #[default]
    None,
    ClosedFist,
    OpenPalm,
    PointingUp,
    ThumbDown,
    ThumbUp,
    Victory,
    ILoveYou,
}

impl CannedGesture {
    /// Maps the classifier index the JS side sends. Unknown -> `None`.
    pub fn from_id(id: i32) -> Self {
        match id {
            1 => Self::ClosedFist,
            2 => Self::OpenPalm,
            3 => Self::PointingUp,
            4 => Self::ThumbDown,
            5 => Self::ThumbUp,
            6 => Self::Victory,
            7 => Self::ILoveYou,
            _ => Self::None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ClosedFist => "closed_fist",
            Self::OpenPalm => "open_palm",
            Self::PointingUp => "pointing_up",
            Self::ThumbDown => "thumb_down",
            Self::ThumbUp => "thumb_up",
            Self::Victory => "victory",
            Self::ILoveYou => "i_love_you",
        }
    }
}

/// What the engine actually does with a hand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Spell {
    /// Hand present but not commanding anything; still pushes the fluid.
    #[default]
    Idle,
    /// Pinched fingers spin a vortex around the pinch point.
    Vortex,
    /// Open palm pushes fluid and particles away.
    Repel,
    /// Closed fist pulls everything into a gravity well.
    Attract,
    /// Pointing finger paints a hot dye trail.
    Ignite,
    /// Victory sign chills the fluid: heavy damping, frost palette.
    Freeze,
    /// Thumb up bursts particles outward.
    Shatter,
    /// A held fist opened: an expanding ring shockwave. Reported by the spell
    /// layer for the moment after the release; never latched by the tracker.
    Release,
}

impl Spell {
    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Vortex => "vortex",
            Self::Repel => "repel",
            Self::Attract => "attract",
            Self::Ignite => "ignite",
            Self::Freeze => "freeze",
            Self::Shatter => "shatter",
            Self::Release => "release",
        }
    }

    /// Hue (in turns) the spell tints its dye with.
    pub fn hue(self) -> f32 {
        match self {
            Self::Idle => 0.55,
            Self::Vortex => 0.74,
            Self::Repel => 0.45,
            Self::Attract => 0.87,
            Self::Ignite => 0.05,
            Self::Freeze => 0.55,
            Self::Shatter => 0.14,
            Self::Release => 0.62,
        }
    }
}
