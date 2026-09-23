//! Legacy renderer state retained to keep the WASM/rendering contract stable.
//! The throw gesture was removed, so the engine always reports an idle rush.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RushKind {
    #[default]
    Push,
}

impl RushKind {
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Push => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RushState {
    pub at: [f32; 2],
    pub progress: f32,
    pub power: f32,
    pub kind: RushKind,
}

impl RushState {
    pub const IDLE: Self = Self {
        at: [0.5, 0.5],
        progress: 0.0,
        power: 0.0,
        kind: RushKind::Push,
    };
}
