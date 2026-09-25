/**
 * Test-side view of the hooks `web/src/main.ts` publishes on `window`.
 *
 * Declared structurally rather than imported from the app so the specs stay
 * decoupled from the app's module graph — Playwright compiles these files on
 * its own and does not resolve Vite's aliases or CSS imports.
 */

interface AetherDiagnostics {
  frames: number;
  fps: number;
  stepMs: number;
  renderMs: number;
  inferenceMs: number;
  cameraAvailable: boolean;
  perception:
    | { kind: 'loading' }
    | { kind: 'ready'; delegate: 'GPU' | 'CPU' }
    | { kind: 'unavailable'; reason: string };
  /** Packed engine stats; indices mirror `STAT` in `web/src/constants.ts`. */
  stats: number[];
  /** Latched spell name per hand slot. */
  spells: [string, string];
  particleCount: number;
  mode: 'aether' | 'camera' | 'debug' | 'particles';
  /** Measured rate at which inference results reach the engine, in Hz. */
  perceptionHz: number;
  /** Inline path only: the cadence the duty budget allows, in Hz. */
  perceptionBudgetHz: number;
  /** Inline path only: smoothed wall time one inference costs the loop, in ms. */
  inferenceCostMs: number;
  /** The model's own latency for the last completed result, in ms. */
  modelMs: number;
  /** The models run on the worker rather than inline. */
  perceptionWorker: boolean;
}

interface AetherTestHooks {
  diagnostics(): AetherDiagnostics;
  /** Swaps in a scripted perception source; null detaches perception. */
  setPerception(source: unknown): void;
  setParam(key: string, value: number): boolean;
  /** Resizes the particle pool, which reallocates engine buffers. */
  setParticleCount(n: number): void;
  /**
   * Mean luminance of the last rendered frame, `[0, 1]`.
   *
   * A framebuffer readback: never poll it from `waitForFunction`, which runs
   * on every animation frame. Doing so measurably slows the app down.
   */
  luminance(): number;
  forceMode(mode: 'aether' | 'camera' | 'debug' | 'particles'): void;
  reset(): void;
  /** The running app; only the parts a spec drives directly are declared. */
  app: {
    stop(): void;
    rawEngine: { step(dt: number): void };
  };
}

interface Window {
  __aether?: AetherTestHooks;
}
