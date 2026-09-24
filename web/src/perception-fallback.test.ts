import { afterEach, expect, it, vi } from 'vitest';
import type { AetherEngine } from './wasm/aether';

const { workers, inlines, behavior } = vi.hoisted(() => ({
  workers: [] as any[], inlines: [] as any[],
  behavior: { constructorFails: false, inlineFails: false },
}));
vi.mock('./perception-worker-client', () => ({
  WorkerPerception: class {
    status: { kind: 'ready'; delegate: 'GPU' } | { kind: 'unavailable'; reason: string } = { kind: 'ready', delegate: 'GPU' };
    close = vi.fn();
    init = vi.fn(async () => {});
    process = vi.fn(() => null);
    constructor() {
      if (behavior.constructorFails) throw new Error('workers unavailable');
      workers.push(this);
    }
  },
}));
vi.mock('./perception', () => ({
  MediaPipePerception: class {
    status = { kind: 'ready', delegate: 'CPU' };
    close = vi.fn();
    init = vi.fn(async () => {
      if (behavior.inlineFails) throw new Error('models unavailable');
    });
    process = vi.fn(() => null);
    constructor() { inlines.push(this); }
  },
}));
import { PerceptionPump } from './perception-pump';

afterEach(() => {
  workers.length = 0;
  inlines.length = 0;
  behavior.constructorFails = false;
  behavior.inlineFails = false;
  vi.restoreAllMocks();
});

function makePump() {
  const engine = { clear_perception: vi.fn() } as unknown as AetherEngine;
  const pump = new PerceptionPump({
    engine,
    frame: () => ({}) as HTMLVideoElement,
    mask: () => new Float32Array(0),
  });
  return { pump, engine };
}

it('switches a worker that dies after boot to inline inference once', async () => {
  const { pump, engine } = makePump();
  await pump.attach();
  expect(pump.attached).toBe(true);
  workers[0].status = { kind: 'unavailable', reason: 'worker crashed' };
  pump.pump(1000);
  await vi.waitFor(() => expect(pump.status.kind).toBe('ready'));
  expect(workers[0].close).toHaveBeenCalledOnce();
  expect(engine.clear_perception).toHaveBeenCalledOnce();
  expect(inlines).toHaveLength(1);
  pump.pump(1100);
  expect(inlines[0].process).toHaveBeenCalledOnce();
  expect(inlines).toHaveLength(1);
});

it('does not resurrect inline perception after the pump closes', async () => {
  const { pump } = makePump();
  await pump.attach();
  workers[0].status = { kind: 'unavailable', reason: 'worker crashed' };
  pump.pump(1000);
  pump.close();
  await Promise.resolve();
  expect(pump.attached).toBe(false);
  expect(inlines[0].close).toHaveBeenCalledOnce();
});
