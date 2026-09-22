import { describe, expect, it } from "vitest";

import {
  aspectOf,
  bloomAmount,
  cameraFit,
  exposureAmount,
  particleGain,
  particleSize,
  rushFocus,
  usesCamera,
} from "./look";
import { STYLES } from "./styles";

const style = STYLES.blend;

describe("usesCamera", () => {
  it("is false for a style with no camera tint or edge", () => {
    expect(
      usesCamera({ ...style, camTint: [0, 0, 0], camEdge: [0, 0, 0] }),
    ).toBe(false);
  });

  it("is true as soon as either contributes", () => {
    expect(
      usesCamera({ ...style, camTint: [0, 0, 0], camEdge: [0, 0.2, 0] }),
    ).toBe(true);
    expect(
      usesCamera({ ...style, camTint: [0.1, 0, 0], camEdge: [0, 0, 0] }),
    ).toBe(true);
  });
});

describe("aspectOf", () => {
  it("falls back when either side has no extent", () => {
    expect(aspectOf(0, 480, 1.5)).toBe(1.5);
    expect(aspectOf(640, 0, 1.5)).toBe(1.5);
    expect(aspectOf(640, 480, 1.5)).toBeCloseTo(4 / 3, 6);
  });
});

describe("cameraFit", () => {
  it("shrinks the width when the stream is wider than the canvas", () => {
    const [x, y] = cameraFit(1, 4 / 3);
    expect(x).toBeCloseTo(3 / 4, 6);
    expect(y).toBe(1);
  });

  it("shrinks the height when the canvas is wider than the stream", () => {
    const [x, y] = cameraFit(8 / 3, 4 / 3);
    expect(x).toBe(1);
    expect(y).toBeCloseTo(0.5, 6);
  });

  it("does not scale at all without a stream aspect", () => {
    expect(cameraFit(8 / 3, 0)).toEqual([1, 1]);
  });
});

describe("bloomAmount / exposureAmount", () => {
  it("pushes bloom with intensity", () => {
    expect(bloomAmount(style, 0)).toBeCloseTo(style.bloom * 0.72, 6);
    expect(bloomAmount(style, 1)).toBeCloseTo(style.bloom * 1.47, 6);
  });

  it("only breathes the exposure of a graded style", () => {
    const graded = { ...style, grade: 1 };
    const raw = { ...style, grade: 0 };
    expect(exposureAmount(graded, 0.5)).toBeCloseTo(
      graded.exposure * (0.96 + 0.11),
      6,
    );
    expect(exposureAmount(raw, 0.5)).toBe(raw.exposure);
    expect(exposureAmount(raw, 1)).toBe(raw.exposure);
  });
});

describe("particleGain / particleSize", () => {
  it("keeps emitted light roughly constant across pool sizes", () => {
    const small = particleGain(style, 2_000) * 2_000;
    const large = particleGain(style, 220_000) * 220_000;
    expect(large / small).toBeGreaterThan(1);
    expect(large / small).toBeLessThan(30);
  });

  it("clamps the gain at both ends of the pool range", () => {
    expect(particleGain(style, 1)).toBeCloseTo(style.particles * 2, 6);
    expect(particleGain(style, 10_000_000)).toBeCloseTo(
      style.particles * 0.1,
      6,
    );
  });

  it("shrinks the dot as the pool grows, and stops at the floor", () => {
    expect(particleSize(1, 0)).toBeCloseTo(3.1, 6);
    expect(particleSize(1, 200_000)).toBeCloseTo(1.35, 6);
    expect(particleSize(1, 1_000_000)).toBeCloseTo(1.35, 6);
    expect(particleSize(2, 200_000)).toBeCloseTo(2.7, 6);
  });
});

describe("rushFocus", () => {
  it("is centred and inert without a staged rush", () => {
    expect(rushFocus(null, false)).toEqual({
      x: 0.5,
      y: 0.5,
      progress: 0,
      power: 0,
    });
    expect(rushFocus(new Float32Array([0.2, 0.3, 0.7, 0]), true).power).toBe(0);
    expect(rushFocus(new Float32Array([0.2, 0.3]), false).power).toBe(0);
  });

  it("passes the focus through once its power is positive", () => {
    const r = rushFocus(new Float32Array([0.2, 0.3, 0.7, 1.5]), false);
    expect(r.x).toBeCloseTo(0.2, 6);
    expect(r.y).toBeCloseTo(0.3, 6);
    expect(r.progress).toBeCloseTo(0.7, 6);
    expect(r.power).toBeCloseTo(1.5, 6);
  });

  it("flips y for the GL scene target only", () => {
    const rush = new Float32Array([0.2, 0.3, 0.7, 1.5]);
    expect(rushFocus(rush, true).y).toBeCloseTo(0.7, 6);
    expect(rushFocus(rush, false).y).toBeCloseTo(0.3, 6);
  });

  it("ignores a non-finite power instead of poisoning the pass", () => {
    const r = rushFocus(new Float32Array([0.2, 0.3, 0.7, Number.NaN]), false);
    expect(r.power).toBe(0);
    expect(r.x).toBe(0.5);
  });
});
