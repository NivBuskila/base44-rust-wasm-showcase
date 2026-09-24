import { afterEach, expect, it, vi } from 'vitest';
import { WorkerPerception } from './perception-worker-client';
import type { FromWorker } from './perception-protocol';

class FakeWorker {
  onmessage: ((event: MessageEvent<FromWorker>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  emit(message: FromWorker): void {
    this.onmessage?.({ data: message } as MessageEvent<FromWorker>);
  }
}

let worker: FakeWorker;
vi.stubGlobal('Worker', class extends FakeWorker {
  constructor() {
    super();
    worker = this;
  }
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

it('rejects boot on a worker error instead of waiting forever', async () => {
  vi.stubGlobal('Worker', class extends FakeWorker {
    constructor() { super(); worker = this; }
  });
  const source = new WorkerPerception();
  const boot = source.init();
  worker.onerror?.({ message: 'module failed' } as ErrorEvent);
  await expect(boot).rejects.toThrow('module failed');
  expect(source.status.kind).toBe('unavailable');
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it('times out a silent boot and rejects when closed while loading', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('Worker', class extends FakeWorker {
    constructor() { super(); worker = this; }
  });
  const source = new WorkerPerception();
  const boot = source.init();
  const rejected = expect(boot).rejects.toThrow('timed out');
  await vi.advanceTimersByTimeAsync(60_000);
  await rejected;
  expect(worker.terminate).toHaveBeenCalledOnce();

  const other = new WorkerPerception();
  const cancelled = expect(other.init()).rejects.toThrow('closed');
  other.close();
  await cancelled;
  expect(vi.getTimerCount()).toBe(0);
});

it('marks a stalled frame unavailable so the pump can switch sources', async () => {
  vi.stubGlobal('Worker', class extends FakeWorker {
    constructor() { super(); worker = this; }
  });
  let now = 1000;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  const source = new WorkerPerception();
  const boot = source.init();
  worker.emit({ type: 'status', status: { kind: 'ready', delegate: 'GPU' } });
  await boot;
  const video = { readyState: 2, videoWidth: 640, videoHeight: 480, currentTime: 1 } as HTMLVideoElement;
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ close: vi.fn() }));
  source.process(video, now);
  await Promise.resolve();
  expect(worker.postMessage).toHaveBeenCalledTimes(2);
  // The first pass compiles shaders; a cold cache made it take ~18 s.
  now += 18_000;
  source.process(video, now);
  expect(source.status.kind).toBe('ready');
  worker.emit({ type: 'skip', maskBuf: null });

  (video as { currentTime: number }).currentTime = 2;
  source.process(video, now);
  await Promise.resolve();
  expect(worker.postMessage).toHaveBeenCalledTimes(3);
  now += 5001;
  source.process(video, now);
  expect(source.status.kind).toBe('unavailable');
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it('still gives up on a first frame that never answers', async () => {
  vi.stubGlobal('Worker', class extends FakeWorker {
    constructor() { super(); worker = this; }
  });
  let now = 1000;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  const source = new WorkerPerception();
  const boot = source.init();
  worker.emit({ type: 'status', status: { kind: 'ready', delegate: 'GPU' } });
  await boot;
  const video = { readyState: 2, videoWidth: 640, videoHeight: 480, currentTime: 1 } as HTMLVideoElement;
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ close: vi.fn() }));
  source.process(video, now);
  await Promise.resolve();
  now += 60_001;
  source.process(video, now);
  expect(source.status.kind).toBe('unavailable');
});

it('handles an unreadable worker reply after boot', async () => {
  vi.stubGlobal('Worker', class extends FakeWorker {
    constructor() { super(); worker = this; }
  });
  const source = new WorkerPerception();
  const boot = source.init();
  worker.emit({ type: 'status', status: { kind: 'ready', delegate: 'GPU' } });
  await boot;
  worker.onmessageerror?.();
  expect(source.status.kind).toBe('unavailable');
});
