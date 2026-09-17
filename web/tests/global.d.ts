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
  /** Mean luminance of the last rendered frame, `[0, 1]`. */
  luminance: number;
  particleCount: number;
  mode: 'aether' | 'camera' | 'debug' | 'particles';
}

interface AetherTestHooks {
  diagnostics(): AetherDiagnostics;
  /** Swaps in a scripted perception source; null detaches perception. */
  setPerception(source: unknown): void;
  setParam(key: string, value: number): boolean;
  /** Resizes the particle pool, which reallocates engine buffers. */
  setParticleCount(n: number): void;
  forceMode(mode: 'aether' | 'camera' | 'debug' | 'particles'): void;
  reset(): void;
}

interface Window {
  __aether?: AetherTestHooks;
}
