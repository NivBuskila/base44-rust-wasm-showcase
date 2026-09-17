//! # Aether — a gesture-driven fluid reality engine
//!
//! `aether-core` is the whole simulation: an incompressible fluid solver, a
//! particle system, dense optical flow, body-silhouette collision and a
//! gesture/spell layer. It is **dependency-free** and compiles unchanged for
//! native (tests, benches) and `wasm32` (the browser app), which is what lets
//! the physics be unit-tested on the host instead of poked at through a canvas.
//!
//! The crate knows nothing about cameras, WebGL or MediaPipe. Everything
//! crosses the boundary as plain buffers described in [`config`], so the
//! browser layer is replaceable and the engine stays testable.
//!
//! ```no_run
//! use aether_core::engine::Engine;
//! let mut engine = Engine::new(0xA37E);
//! engine.step(1.0 / 60.0);
//! let pixels = engine.dye_rgba();       // RGBA8, FLUID_W x FLUID_H
//! let points = engine.particle_buffer(); // x, y, heat, life per particle
//! ```

pub mod config;
pub mod engine;
pub mod field;
pub mod fluid;
pub mod flow;
pub mod gesture;
pub mod mask;
pub mod math;
pub mod par;
pub mod particles;
pub mod rng;
pub mod spells;

pub use config::Params;
pub use engine::Engine;
