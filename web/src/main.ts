/**
 * Entry point: boot sequence and the test surface.
 *
 * Everything that runs per frame lives in `app.ts`; this file only brings the
 * pieces up in the right order and reports a failure to the boot screen.
 */

import './styles.css';

import { App } from './app';
import { ensureCrossOriginIsolation } from './cross-origin-isolation';
import { loadEngine } from './engine-loader';
import { mountIntro } from './landing';
import { clearQaSession, readQaSession, type QaSession } from './qa-recorder';
import { createRenderer } from './gpu';
import { assertLayout } from './constants';
import type { PerceptionSource, ViewMode } from './types';

/** Engine seed; fixed so a session is reproducible given the same input. */
const SEED = 0xa37e5eed;

/** Surface exposed on `window.__aether` for the Playwright suite. */
export interface AetherTestHooks {
  app: App;
  diagnostics(): App['diagnostics'];
  setPerception(source: PerceptionSource | null): void;
  setParam(key: string, value: number): boolean;
  setParticleCount(n: number): void;
  forceMode(mode: ViewMode): void;
  /** Framebuffer readback; do not poll this on every frame. */
  luminance(): number;
  reset(): void;
  /** The live session record from this tab. */
  qaSession(): QaSession;
  /** The last session persisted on this origin, from any tab. */
  lastQaSession(): QaSession | null;
  /** Discards the persisted record. */
  clearQaSession(): void;
}

declare global {
  interface Window {
    __aether?: AetherTestHooks;
  }
}

async function boot(): Promise<void> {
  // One console for the whole boot: it is the loading screen and, on a first
  // visit, the welcome that opens onto the running field.
  const intro = mountIntro();
  const setStatus = (text: string) => intro.status(text);

  const fail = (message: string, err?: unknown) => {
    console.error(`[aether] ${message}`, err);
    intro.fail(message);
  };

  try {
    // Must precede loadEngine: it decides the tier from `crossOriginIsolated`,
    // and this is what can still turn that true.
    await ensureCrossOriginIsolation();

    const loaded = await loadEngine(setStatus);
    const engine = new loaded.module.AetherEngine(SEED);
    assertLayout(engine.layout());
    intro.stage('engine');

    const canvas = document.getElementById('stage');
    if (!(canvas instanceof HTMLCanvasElement)) throw new Error('#stage canvas missing');

    setStatus('starting the renderer…');
    const { renderer } = await createRenderer(canvas);
    intro.stage('render');

    const app = new App(engine, loaded.memory, renderer, loaded.tier);
    window.__aether = {
      app,
      diagnostics: () => app.diagnostics,
      setPerception: (source) => app.setPerception(source),
      setParam: (key, value) => engine.set_param(key, value),
      setParticleCount: (n) => app.setParticleCount(n),
      forceMode: (mode) => app.forceMode(mode),
      luminance: () => app.luminance(),
      reset: () => engine.reset(),
      qaSession: () => app.qaSession(),
      lastQaSession: () => readQaSession(),
      clearQaSession,
    };

    // A tab is usually closed rather than idled out, and the 2 Hz writer may be
    // up to two seconds behind when that happens.
    window.addEventListener('pagehide', () => app.flushQaSession());

    await app.start(setStatus);
    intro.ready();
  } catch (err) {
    fail(
      err instanceof Error ? `Aether failed to start: ${err.message}` : 'Aether failed to start.',
      err,
    );
  }
}

void boot();
