//! Scalar and vector grids, plus the sampling primitives the whole engine
//! shares.
//!
//! # Coordinate convention
//!
//! A grid value is defined **at the integer lattice point** `(i, j)`, so a
//! `w * h` grid spans `[0, w - 1] x [0, h - 1]` in grid space. [`Grid::sample`]
//! bilinearly interpolates and clamps outside that box — clamping rather than
//! wrapping is what keeps semi-Lagrangian advection stable at the borders.
//!
//! Every module in this crate works in grid space; only the engine boundary
//! converts to and from normalised `[0, 1]` screen space.

mod grid;
#[cfg(test)]
mod tests;
mod vec_field;

pub use grid::Grid;
pub use vec_field::VecField;
