import { afterEach, expect, it, vi } from 'vitest';
import type { AetherEngine } from './wasm/aether';
import { PerceptionPump } from './perception-pump';
import type { PerceptionSource } from './types';

afterEach(() => vi.restoreAllMocks());

it('reports completed inference throughput, not the configured cadence', () => {
  let now = 1000;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  const engine = {
    push_hands: vi.fn(),
    push_pose: vi.fn(),
    clear_perception: vi.fn(),
  } as unknown as AetherEngine;
  const source: PerceptionSource = {
    status: { kind: 'ready', delegate: 'GPU' },
    init: async () => {},
    close: vi.fn(),
    process: () => ({
      hands: new Float32Array(0), pose: new Float32Array(0),
      mask: null, latencyMs: 0, anyHand: false,
    }),
  };
  const pump = new PerceptionPump({
    engine,
    frame: () => ({}) as HTMLVideoElement,
    mask: () => new Float32Array(0),
  });
  pump.setSource(source);
  expect(pump.actualHz).toBe(0);
  pump.pump(now);
  expect(pump.actualHz).toBe(0); // One result does not establish a rate.
  now += 100;
  pump.pump(now);
  expect(pump.actualHz).toBeCloseTo(10);
  expect(pump.hz).toBeCloseTo(30); // Budget is still configured independently.
  now += 2100;
  expect(pump.actualHz).toBe(0); // A stopped worker cannot claim live throughput.
  pump.setSource(null);
  expect(pump.actualHz).toBe(0);
});
