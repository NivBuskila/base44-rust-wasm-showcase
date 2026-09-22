import { describe, expect, it } from "vitest";

import { FrameTimings } from "./frame-timings";

const spin = (ms: number): void => {
  const until = performance.now() + ms;
  while (performance.now() < until) {
    /* busy wait: the point is measurable wall-clock cost */
  }
};

describe("FrameTimings", () => {
  it("charges a stage its own wall-clock cost and returns its value", () => {
    const t = new FrameTimings();
    t.begin(3);
    const stats = t.measure("stepMs", () => {
      spin(2);
      return 42;
    });
    expect(stats).toBe(42);
    expect(t.stepMs).toBeGreaterThan(0);
    expect(t.renderMs).toBe(0);
    expect(t.outsideMs).toBe(3);
  });

  it("frameMs covers the whole callback, not one stage", () => {
    const t = new FrameTimings();
    t.begin(0);
    t.measure("stepMs", () => spin(1));
    t.measure("renderMs", () => spin(1));
    t.end();
    expect(t.frameMs).toBeGreaterThanOrEqual(t.stepMs);
    expect(t.frameMs).toBeGreaterThanOrEqual(t.renderMs);
  });

  it("reports every row as data", () => {
    const t = new FrameTimings();
    t.begin(1);
    t.end();
    expect(Object.keys(t.rows).sort()).toEqual([
      "cameraMs",
      "frameMs",
      "hudMs",
      "outsideMs",
      "renderMs",
      "stepMs",
    ]);
  });
});
