/**
 * Picks and boots the best engine build this browser can run.
 *
 * Two modules come out of `scripts/build-wasm.sh`:
 *
 * - `wasm/`     single-threaded, SIMD128. Runs anywhere WebAssembly does.
 * - `wasm-mt/`  the same engine compiled with atomics, driving a rayon pool
 *               of Web Workers over shared linear memory. Needs the page to
 *               be cross-origin isolated (COOP + COEP headers, see
 *               `vite.config.ts`), because that is what unlocks
 *               `SharedArrayBuffer`.
 *
 * The threaded build is tried first when the page qualifies and falls back
 * silently otherwise, so the app never fails to start over a missing header
 * or an old browser. `?engine=single` forces the baseline for side-by-side
 * comparison.
 */

import type * as Baseline from './wasm/aether';

/** The engine module surface `main.ts` uses; identical across both builds. */
export type EngineModule = Pick<typeof Baseline, 'AetherEngine'>;

export type EngineTierName = 'threads' | 'single';

export interface EngineTier {
  readonly name: EngineTierName;
  /** Worker threads rayon is spreading the hot loops over; 1 when single. */
  readonly threads: number;
  /** Why this tier was chosen — surfaced in the HUD and the console. */
  readonly reason: string;
}

export interface LoadedEngine {
  readonly module: EngineModule;
  readonly memory: WebAssembly.Memory;
  readonly tier: EngineTier;
}

/** Cap on the worker pool: past this the memory bus, not the cores, is the limit. */
const MAX_THREADS = 16;

/**
 * Whether this document may allocate `SharedArrayBuffer`, which the threaded
 * build needs for its memory. Cross-origin isolation is a property of the
 * whole frame tree, so an embedded preview whose parent lacks the headers
 * reports false here even though the dev server sends them.
 */
export function canUseThreads(): { ok: boolean; reason: string } {
  if (typeof SharedArrayBuffer === 'undefined') {
    return { ok: false, reason: 'SharedArrayBuffer unavailable' };
  }
  if (!globalThis.crossOriginIsolated) {
    return { ok: false, reason: 'page is not cross-origin isolated' };
  }
  const cores = navigator.hardwareConcurrency ?? 1;
  if (cores < 2) return { ok: false, reason: 'single-core device' };
  return { ok: true, reason: `${cores} cores` };
}

function requestedTier(): EngineTierName | 'auto' {
  const v = new URLSearchParams(location.search).get('engine');
  return v === 'single' || v === 'threads' ? v : 'auto';
}

export async function loadEngine(
  setStatus: (text: string) => void = () => {},
): Promise<LoadedEngine> {
  const want = requestedTier();
  const gate = canUseThreads();

  if (want !== 'single' && gate.ok) {
    try {
      setStatus('loading the multithreaded engine…');
      const mt = await import('./wasm-mt/aether.js');
      const wasm = await mt.default();
      const cores = Math.min(MAX_THREADS, navigator.hardwareConcurrency ?? 2);
      setStatus(`spinning up ${cores} worker threads…`);
      await mt.initThreadPool(cores);
      const threads = mt.thread_count();
      console.info(`[aether] engine: ${threads} threads, simd128 (${gate.reason})`);
      return {
        module: mt,
        memory: wasm.memory,
        tier: { name: 'threads', threads, reason: gate.reason },
      };
    } catch (err) {
      console.warn('[aether] threaded engine unavailable, falling back to single-threaded', err);
    }
  }

  setStatus('loading the engine…');
  const single = await import('./wasm/aether.js');
  const wasm = await single.default();
  const reason = want === 'single' ? 'forced by ?engine=single' : gate.reason;
  console.info(`[aether] engine: single thread, simd128 (${reason})`);
  return {
    module: single,
    memory: wasm.memory,
    tier: { name: 'single', threads: 1, reason },
  };
}
