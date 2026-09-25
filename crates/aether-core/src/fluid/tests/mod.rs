//! Solver tests, split by stage. The reference kernels and the grid fixtures live in `support.rs`.

mod advection;
mod incompressible;
mod kernels;
mod obstacles;
mod stability;
mod support;
#[cfg(feature = "parallel")]
mod threads;
