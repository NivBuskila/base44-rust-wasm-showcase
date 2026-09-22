import { describe, expect, it } from "vitest";

import {
  MAX_DPR,
  MIN_SCENE_SCALE,
  SCENE_PIXEL_BUDGET,
  bufferSize,
  deviceScale,
  scaledSize,
  sceneScale,
  viewportScale,
} from "./sizing";

describe("deviceScale", () => {
  it("caps at the DPR ceiling but keeps a zoomed-out ratio below 1", () => {
    expect(deviceScale(3)).toBe(MAX_DPR);
    expect(deviceScale(0.5)).toBe(0.5);
  });

  it("falls back to 1 for a missing ratio", () => {
    expect(deviceScale(0)).toBe(1);
  });
});

describe("bufferSize", () => {
  it("rounds the CSS size at the DPR and never returns zero", () => {
    expect(bufferSize(800, 600, 1.5)).toEqual([1200, 900]);
    expect(bufferSize(0, 0, 2)).toEqual([1, 1]);
  });
});

describe("viewportScale", () => {
  it("is 1 at and above the reference width and floors on a phone", () => {
    expect(viewportScale(1200)).toBe(1);
    expect(viewportScale(2400)).toBe(1);
    expect(viewportScale(390)).toBe(0.45);
  });
});

describe("sceneScale", () => {
  it("leaves a frame under the budget untouched", () => {
    expect(sceneScale(2560, 1440)).toBe(1);
  });

  it("pulls a frame over the budget back to the budgeted fraction", () => {
    const scale = sceneScale(5120, 2880);
    expect(scale).toBeCloseTo(
      Math.sqrt(SCENE_PIXEL_BUDGET / (5120 * 2880)),
      6,
    );
  });

  it("multiplies the governor in and clamps at the floor", () => {
    expect(sceneScale(1920, 1080, { quality: 0.7 })).toBeCloseTo(0.7, 6);
    expect(sceneScale(1920, 1080, { quality: 0.1 })).toBe(MIN_SCENE_SCALE);
  });

  it("lets the pin outrank both the budget and the governor", () => {
    expect(sceneScale(5120, 2880, { quality: 0.5, pinned: 1 })).toBe(1);
  });

  it("honours a tighter budget", () => {
    expect(sceneScale(1920, 1080, { budget: 1_600_000 })).toBeLessThan(1);
  });
});

describe("scaledSize", () => {
  it("rounds and never returns zero", () => {
    expect(scaledSize(1000, 500, 0.5)).toEqual([500, 250]);
    expect(scaledSize(1, 1, 0.4)).toEqual([1, 1]);
  });
});
