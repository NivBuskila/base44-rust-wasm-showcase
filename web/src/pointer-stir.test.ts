import { describe, expect, it } from 'vitest';
import { paintStir } from './pointer-stir';

const W = 128;
const H = 72;

function brightest(plane: Uint8Array): { x: number; y: number; value: number } {
  let best = 0;
  for (let i = 1; i < plane.length; i++) if (plane[i] > plane[best]) best = i;
  return { x: best % W, y: Math.floor(best / W), value: plane[best] };
}

describe('paintStir', () => {
  it('centres a full-brightness blob on the pointer', () => {
    const plane = new Uint8Array(W * H);
    paintStir(plane, W, H, 0.25, 0.5);
    const peak = brightest(plane);
    // The centre falls between cells, so the peak is just under 255.
    expect(peak.value).toBeGreaterThan(245);
    expect(Math.abs(peak.x - 0.25 * (W - 1))).toBeLessThanOrEqual(1);
    expect(Math.abs(peak.y - 0.5 * (H - 1))).toBeLessThanOrEqual(1);
  });

  it('leaves the rest of the plane black, so only the blob can move', () => {
    const plane = new Uint8Array(W * H).fill(200);
    paintStir(plane, W, H, 0.5, 0.5);
    expect(plane[0]).toBe(0);
    expect(plane[W * H - 1]).toBe(0);
    const lit = plane.filter((v) => v > 0).length;
    expect(lit).toBeGreaterThan(20);
    expect(lit).toBeLessThan(W * H * 0.15);
  });

  it('clips at the border instead of wrapping', () => {
    const plane = new Uint8Array(W * H);
    paintStir(plane, W, H, 1, 0);
    expect(plane[W - 1]).toBe(255);
    // The opposite edge of the same row must stay dark.
    expect(plane[0]).toBe(0);
  });
});
