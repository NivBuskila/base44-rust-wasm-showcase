/**
 * Backend choice: WebGPU when the browser offers a hardware device, WebGL2
 * otherwise.
 *
 * The WebGL2 compositor stays the baseline — it runs everywhere and is the one
 * the test suite grades against — so this never fails the boot: any refusal
 * from `acquireGpu`, and any error setting the WebGPU chain up, falls through
 * to `render/renderer.ts` with the reason logged.
 */

import { Renderer } from '../render/renderer';
import type { SceneRenderer } from '../types';
import { acquireGpu } from './device';
import { GpuRenderer } from './renderer';

export interface SelectedRenderer {
  renderer: SceneRenderer;
  /** Why this backend was chosen — surfaced in the diagnostics and the console. */
  reason: string;
}

export async function createRenderer(canvas: HTMLCanvasElement): Promise<SelectedRenderer> {
  const { ctx, gate } = await acquireGpu();
  if (ctx) {
    try {
      const renderer = new GpuRenderer(canvas, ctx);
      console.info(`[aether] WebGPU backend on ${gate.reason}`);
      return { renderer, reason: gate.reason };
    } catch (err) {
      // A canvas that already handed out a WebGL2 context cannot be
      // reconfigured, so this has to come first — and if it throws anyway, the
      // fallback below is on a canvas that never got a working context.
      console.warn('[aether] WebGPU setup failed; falling back to WebGL2', err);
    }
  } else {
    console.info(`[aether] WebGL2 backend: ${gate.reason}`);
  }
  return { renderer: new Renderer(canvas), reason: gate.reason };
}
