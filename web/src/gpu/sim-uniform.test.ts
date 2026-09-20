import { describe, expect, it } from "vitest";

import type { GpuSimFrame } from "../types";
import { SIM_UNIFORM_FLOATS } from "./shaders/particles";
import {
  HEAT_DECAY,
  HEAT_RISE,
  MAX_SIM_STEP,
  packSimUniform,
  type SimUniformInput,
} from "./sim-uniform";

/**
 * The slot names of the WGSL `Sim` struct, in order, so a failure reads as the
 * field that moved rather than as an index.
 */
const SLOT = {
  maxx: 0,
  maxy: 1,
  osx: 2,
  osy: 3,
  dt: 4,
  keep: 5,
  heatKeep: 6,
  heatRise: 7,
  invDt: 8,
  swirl: 9,
  gravity: 10,
  drag: 11,
  life: 12,
  obstacleW: 13,
  obstacleH: 14,
  hasObstacle: 15,
  active: 16,
  ops: 17,
  frame: 18,
  respawnBudget: 19,
  gridW: 20,
  gridH: 21,
} as const;

/** `params`, in the order `offload.rs` writes them. */
function params(over: Partial<Record<string, number>> = {}): Float32Array {
  const p = new Float32Array(8);
  p[0] = over.dt ?? 1 / 60;
  p[1] = over.life ?? 3;
  p[2] = over.drag ?? 1.5;
  p[3] = over.spawnRate ?? 0;
  p[4] = over.damping ?? 0.4;
  p[5] = over.swirl ?? 2;
  p[6] = over.gravity ?? -1;
  return p;
}

function frame(over: Partial<GpuSimFrame> = {}): GpuSimFrame {
  return {
    active: 1000,
    gridW: 200,
    gridH: 120,
    velU: new Float32Array(200 * 120),
    velV: new Float32Array(200 * 120),
    obstacle: new Float32Array(200 * 120),
    obstacleInfo: new Float32Array([200, 120, 1]),
    ops: new Float32Array(64),
    opCount: 0,
    params: params(),
    ...over,
  } as GpuSimFrame;
}

function input(over: Partial<SimUniformInput> = {}): SimUniformInput {
  return {
    sim: frame(),
    active: 1000,
    opCount: 0,
    obstacleW: 200,
    obstacleH: 120,
    hasObstacle: true,
    frameIndex: 7,
    respawnCredit: 0,
    ...over,
  };
}

function pack(over: Partial<SimUniformInput> = {}) {
  const block = new ArrayBuffer(SIM_UNIFORM_FLOATS * 4);
  const f = new Float32Array(block);
  const u = new Uint32Array(block);
  const out = packSimUniform(f, u, input(over));
  return { f, u, out };
}

describe("packSimUniform", () => {
  it("writes the slots the WGSL Sim struct reads", () => {
    const { f, u } = pack({ active: 4242, opCount: 9, frameIndex: 13 });
    expect(f[SLOT.maxx]).toBe(199);
    expect(f[SLOT.maxy]).toBe(119);
    expect(f[SLOT.dt]).toBeCloseTo(1 / 60, 6);
    expect(f[SLOT.life]).toBe(3);
    expect(f[SLOT.drag]).toBe(1.5);
    expect(f[SLOT.obstacleW]).toBe(200);
    expect(f[SLOT.hasObstacle]).toBe(1);
    expect(u[SLOT.active]).toBe(4242);
    expect(u[SLOT.ops]).toBe(9);
    expect(u[SLOT.frame]).toBe(13);
    expect(u[SLOT.gridW]).toBe(200);
    expect(u[SLOT.gridH]).toBe(120);
  });

  it("folds swirl and gravity through dt, as the Rust frame does", () => {
    const dt = 1 / 30;
    const { f } = pack({
      sim: frame({ params: params({ dt, swirl: 3, gravity: -2 }) }),
    });
    expect(f[SLOT.swirl]).toBeCloseTo(3 * dt, 6);
    expect(f[SLOT.gravity]).toBeCloseTo(-2 * dt, 6);
    expect(f[SLOT.invDt]).toBeCloseTo(30, 4);
  });

  it("turns damping and heat rates into per-frame keep factors", () => {
    const dt = 1 / 60;
    const { f } = pack({
      sim: frame({ params: params({ dt, damping: 0.4 }) }),
    });
    expect(f[SLOT.keep]).toBeCloseTo(Math.exp(-0.4 * dt), 6);
    expect(f[SLOT.heatKeep]).toBeCloseTo(Math.exp(-HEAT_DECAY * dt), 6);
    expect(f[SLOT.heatRise]).toBeCloseTo(1 - Math.exp(-HEAT_RISE * dt), 6);
  });

  it("clamps dt to the solver ceiling and reports it", () => {
    const { f, out } = pack({ sim: frame({ params: params({ dt: 10 }) }) });
    expect(f[SLOT.dt]).toBe(MAX_SIM_STEP);
    expect(out.dt).toBe(MAX_SIM_STEP);
  });

  it("survives a non-finite parameter instead of poisoning the block", () => {
    const p = params();
    p[0] = Number.NaN;
    p[5] = Number.POSITIVE_INFINITY;
    const { f } = pack({ sim: frame({ params: p }) });
    expect(f[SLOT.dt]).toBe(0);
    expect(f[SLOT.invDt]).toBe(0);
    expect(f[SLOT.swirl]).toBe(0);
  });

  it("carries fractional respawns so a slow rate still respawns", () => {
    const dt = 1 / 60;
    const sim = frame({ params: params({ dt, spawnRate: 30 }) });
    // Half a particle per frame: every other frame must spawn one.
    let credit = 0;
    const budgets: number[] = [];
    for (let i = 0; i < 4; i++) {
      const { u, out } = pack({ sim, respawnCredit: credit });
      budgets.push(u[SLOT.respawnBudget]);
      credit = out.respawnCredit;
    }
    expect(budgets).toEqual([0, 1, 0, 1]);
  });

  it("caps the carried credit so a paused tab cannot bank a burst", () => {
    const dt = 1 / 60;
    const sim = frame({ params: params({ dt, spawnRate: 60 }) });
    const { u } = pack({ sim, respawnCredit: 10_000 });
    expect(u[SLOT.respawnBudget]).toBe(2);
  });

  it("rejects a scratch block smaller than the uniform", () => {
    const small = new ArrayBuffer(4 * 4);
    expect(() =>
      packSimUniform(new Float32Array(small), new Uint32Array(small), input()),
    ).toThrow(/too small/);
  });
});
