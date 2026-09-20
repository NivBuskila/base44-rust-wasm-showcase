import { describe, expect, it } from 'vitest';

import { poseLandmarks } from './pose-landmarks';
import { STEPS } from './tutorial-steps';

const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const MCP = { index: 5, middle: 9 };

function dist(a: number[], b: number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

describe('poseLandmarks', () => {
  it('has a pose for every taught step', () => {
    for (const step of STEPS) expect(poseLandmarks(step.spell), step.spell).toHaveLength(21);
  });

  it('points the index up and folds the rest for ignite', () => {
    const p = poseLandmarks('ignite')!;
    // Extended: the tip is well above its knuckle. Folded: tucked back down.
    expect(p[TIP.index][1]).toBeLessThan(p[MCP.index][1] - 0.15);
    expect(p[TIP.middle][1]).toBeGreaterThan(p[MCP.middle][1] - 0.08);
  });

  it('brings thumb and index tips together for the pinch', () => {
    const p = poseLandmarks('vortex')!;
    expect(dist(p[TIP.thumb], p[TIP.index])).toBeLessThan(0.1);
  });

  it('spreads every finger for repel', () => {
    const p = poseLandmarks('repel')!;
    const xs = [TIP.index, TIP.middle, TIP.ring, TIP.pinky].map((i) => p[i][0]);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
  });
});
