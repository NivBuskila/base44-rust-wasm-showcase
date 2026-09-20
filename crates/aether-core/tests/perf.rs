//! Per-stage cost of one engine step, on the host.
//!
//! Not a pass/fail benchmark — the thresholds are deliberately loose, because
//! CI machines vary by an order of magnitude and a tight bound would just be a
//! flaky test. The point is the printed breakdown: run with
//! `cargo test --release -p aether-core --test perf -- --nocapture` to see where
//! the frame budget actually goes, which is the only honest way to decide what
//! to optimise.
//!
//! Native numbers are an imperfect proxy for the wasm cost, in both
//! directions. wasm32 with SIMD128 lands within roughly 2x of native for dense
//! float work, so a stage that is already 8 ms here has no chance of fitting a
//! 16 ms browser frame. But native is also *penalised* in one specific way:
//! `f32::floor` is a libm call on the SSE2 baseline (no `roundss` without
//! SSE4.1) whereas wasm has a native `f32.floor` opcode, so anything
//! rounding-heavy looks worse here than it will in the browser. Treat the
//! breakdown as a guide to what to look at, and confirm wins against
//! `diagnostics().stepMs` in a real browser.

use aether_core::config::{FLOW_CELLS, FLUID_H, FLUID_W};
use aether_core::engine::Engine;
use std::time::Instant;

/// Steps to average over. Enough to swamp one-off cache warming.
///
/// A debug build runs this roughly 30x slower, which would turn the default
/// `cargo test` into a two-minute wait for numbers that are meaningless without
/// optimisation anyway. In debug the test degrades to a smoke check.
const ITERS: usize = if cfg!(debug_assertions) { 4 } else { 120 };
const DT: f32 = 1.0 / 60.0;

fn time_steps(engine: &mut Engine, iters: usize) -> f64 {
    // One untimed step so the first-touch page faults and the pressure
    // warm-start do not land in the measurement.
    engine.step(DT);
    let start = Instant::now();
    for _ in 0..iters {
        engine.step(DT);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

/// Drives the engine the way the browser does: a moving luma plane so optical
/// flow has real work, and a body mask so the solver carries obstacles.
fn feed_perception(engine: &mut Engine, frame: usize) {
    let shift = frame as f32 * 1.7;
    for y in 0..aether_core::config::FLOW_H {
        for x in 0..aether_core::config::FLOW_W {
            let fx = x as f32 + shift;
            let v = 128.0 + 90.0 * (fx * 0.21).sin() * (y as f32 * 0.17).cos();
            engine.luma_mut()[y * aether_core::config::FLOW_W + x] = v as u8;
        }
    }
    engine.push_luma(1.0 / 30.0);
}

#[test]
fn step_cost_breakdown() {
    let mut label_width = 0usize;
    let mut rows: Vec<(String, f64)> = Vec::new();
    let mut record = |name: &str, ms: f64| {
        label_width = label_width.max(name.len());
        rows.push((name.to_string(), ms));
    };

    // Particles alone: everything else at its cheapest.
    let mut bare = Engine::new(1);
    bare.set_particle_count(0);
    bare.set_param("flow_force", 0.0);
    record(
        "fluid only (no particles, no flow)",
        time_steps(&mut bare, ITERS),
    );

    let mut with_particles = Engine::new(2);
    with_particles.set_particle_count(120_000);
    with_particles.set_param("flow_force", 0.0);
    record("+ 120k particles", time_steps(&mut with_particles, ITERS));

    // The flow drive samples the flow field over the whole fluid grid, so it
    // costs real time even when the flow field itself is stale.
    let mut with_flow = Engine::new(3);
    with_flow.set_particle_count(120_000);
    with_flow.set_param("flow_force", 0.45);
    for f in 0..4 {
        feed_perception(&mut with_flow, f);
    }
    record("+ flow drive", time_steps(&mut with_flow, ITERS));

    // Pressure iterations are the one knob with a linear, predictable cost.
    for iters in [8usize, 28, 60] {
        let mut e = Engine::new(4);
        e.set_particle_count(120_000);
        e.set_param("flow_force", 0.45);
        e.set_param("pressure_iters", iters as f32);
        record(
            &format!("+ pressure_iters = {iters}"),
            time_steps(&mut e, ITERS),
        );
    }

    // Optical flow runs on the camera clock, not the render clock, so its cost
    // is reported per camera frame rather than folded into the step.
    let mut flow_engine = Engine::new(5);
    flow_engine.set_particle_count(0);
    let mut luma = vec![0u8; FLOW_CELLS];
    for (i, px) in luma.iter_mut().enumerate() {
        *px = (i % 251) as u8;
    }
    flow_engine.luma_mut().copy_from_slice(&luma);
    flow_engine.push_luma(1.0 / 30.0);
    let start = Instant::now();
    for f in 0..60 {
        feed_perception(&mut flow_engine, f);
    }
    record(
        "optical flow (per camera frame)",
        start.elapsed().as_secs_f64() * 1000.0 / 60.0,
    );

    println!(
        "\n  engine step cost, {}, {FLUID_W}x{FLUID_H} grid",
        build_kind()
    );
    println!("  {}", "-".repeat(label_width + 14));
    for (name, ms) in &rows {
        println!("  {name:<label_width$}  {ms:>7.3} ms");
    }
    println!("  {}", "-".repeat(label_width + 14));
    println!("  16.7 ms = one 60 fps frame, all stages included\n");

    // The only hard assertion, and it is deliberately generous: a debug build
    // is ~30x slower than the release build these numbers are about, and CI
    // machines vary by an order of magnitude, so anything tighter is a flaky
    // test rather than a performance guard.
    let budget = if cfg!(debug_assertions) {
        4000.0
    } else {
        200.0
    };
    let worst = rows.iter().map(|(_, ms)| *ms).fold(0.0f64, f64::max);
    assert!(
        worst < budget,
        "an engine step took {worst:.1} ms ({}), which cannot work in a browser",
        build_kind()
    );
}

fn build_kind() -> &'static str {
    if cfg!(debug_assertions) {
        "DEBUG BUILD - numbers are meaningless, rerun with --release"
    } else {
        "native release"
    }
}

#[test]
fn particle_scaling_is_linear() {
    // Particles are the one cost the user can dial, so the HUD's slider is only
    // meaningful if the cost actually tracks it. A superlinear curve here means
    // something per-particle is touching a shared structure.
    let mut timings = Vec::new();
    for count in [10_000usize, 40_000, 160_000] {
        let mut e = Engine::new(7);
        e.set_particle_count(count);
        e.set_param("flow_force", 0.0);
        timings.push((count, time_steps(&mut e, ITERS / 2)));
    }

    println!("\n  particle scaling");
    for (count, ms) in &timings {
        println!("  {count:>7} particles  {ms:>7.3} ms");
    }

    let (c0, t0) = timings[0];
    let (c2, t2) = timings[2];
    // Subtract the fixed fluid cost before comparing, or it dominates and the
    // comparison says nothing about the particles.
    let per_particle_low = t0 / c0 as f64;
    let per_particle_high = t2 / c2 as f64;
    println!(
        "  per particle: {:.1} ns -> {:.1} ns\n",
        per_particle_low * 1e6,
        per_particle_high * 1e6
    );
    assert!(
        t2 < t0 * 24.0,
        "16x the particles cost {:.1}x the time; that is not linear",
        t2 / t0
    );
}
