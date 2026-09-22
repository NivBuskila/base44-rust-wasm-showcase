import { describe, expect, it } from "vitest";

import { BUDGET_MS, DutyLimiter, MAX_DUTY } from "./limiter";

describe("DutyLimiter", () => {
  it("stays out of the way below the budget", () => {
    const l = new DutyLimiter(true);
    for (let i = 0; i < 20; i++) l.charge(20, 1000, i === 0);
    expect(l.costMs).toBeLessThan(BUDGET_MS);
    expect(l.gapMs).toBe(0);
    expect(l.ready(1000)).toBe(true);
  });

  it("inserts idle time once inference costs more than the budget", () => {
    const l = new DutyLimiter(true);
    for (let i = 0; i < 40; i++) l.charge(200, 1000, i === 0);
    expect(l.costMs).toBeGreaterThan(BUDGET_MS);
    expect(l.gapMs).toBeCloseTo(l.costMs * (1 / MAX_DUTY - 1), 5);
    expect(l.ready(1000)).toBe(false);
    expect(l.ready(1000 + l.gapMs)).toBe(true);
  });

  it("never rations when disabled, however expensive the pass", () => {
    const l = new DutyLimiter(false);
    for (let i = 0; i < 40; i++) l.charge(200, 1000, i === 0);
    expect(l.gapMs).toBe(0);
    expect(l.ready(1000)).toBe(true);
  });

  it("ignores the first inference's warm-up cost in the estimate", () => {
    const l = new DutyLimiter(true);
    l.charge(4000, 0, true);
    expect(l.costMs).toBe(0);
  });

  it("forgets its estimate on reset", () => {
    const l = new DutyLimiter(true);
    for (let i = 0; i < 40; i++) l.charge(200, 1000, i === 0);
    l.reset();
    expect(l.costMs).toBe(0);
    expect(l.ready(0)).toBe(true);
  });
});
