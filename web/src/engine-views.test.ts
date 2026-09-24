import { describe, expect, it } from 'vitest';
import type { AetherEngine } from './wasm/aether';
import { EngineViews } from './engine-views';

describe('EngineViews particle storage', () => {
  it('reuses the view for frames and rebuilds it after pointer, count, or memory changes', () => {
    const memory = new WebAssembly.Memory({ initial: 1 });
    let ptr = 128;
    let count = 2;
    const engine = {
      dye_ptr: () => 0,
      dye_len: () => 4,
      luma_ptr: () => 4,
      luma_len: () => 4,
      mask_ptr: () => 8,
      mask_capacity: () => 1,
      stats_ptr: () => 16,
      particle_ptr: () => ptr,
      particle_count: () => count,
    } as unknown as AetherEngine;
    const views = new EngineViews(engine, memory);

    const first = views.particles();
    first[0] = 0.5;
    expect(views.particles()).toBe(first);
    expect(views.particles()[0]).toBe(0.5);

    count = 3;
    const resized = views.particles();
    expect(resized).not.toBe(first);
    expect(resized.length).toBe(12);
    expect(views.particles()).toBe(resized);

    ptr = 256;
    const moved = views.particles();
    expect(moved).not.toBe(resized);
    expect(moved.byteOffset).toBe(ptr);

    memory.grow(1);
    const grown = views.particles();
    expect(Object.is(grown, moved)).toBe(false);
    expect(grown.buffer).toBe(memory.buffer);
    expect(views.particles()).toBe(grown);

    views.rebuild();
    expect(views.particles()).not.toBe(grown);
  });
});

describe('EngineViews stats storage', () => {
  it('recomputes only through stats(); lastStats() reads the same view', () => {
    const memory = new WebAssembly.Memory({ initial: 1 });
    let refreshes = 0;
    const engine = {
      dye_ptr: () => 0,
      dye_len: () => 4,
      luma_ptr: () => 4,
      luma_len: () => 4,
      mask_ptr: () => 8,
      mask_capacity: () => 1,
      stats_ptr: () => {
        refreshes++;
        return 64;
      },
    } as unknown as AetherEngine;
    const views = new EngineViews(engine, memory);
    const afterBuild = refreshes;

    const frame = views.stats();
    expect(refreshes).toBe(afterBuild + 1);
    expect(views.lastStats()).toBe(frame);
    expect(refreshes).toBe(afterBuild + 1);
  });
});
