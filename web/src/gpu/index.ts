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
      // A canvas only ever hands out one kind of context: once `getContext
      // ('webgpu')` ran, `getContext('webgl2')` on the same element returns
      // null. Swap in a fresh element for the fallback.
      console.warn('[aether] WebGPU setup failed; falling back to WebGL2', err);
      canvas = replaceCanvas(canvas);
    }
  } else {
    console.info(`[aether] WebGL2 backend: ${gate.reason}`);
  }
  return { renderer: new Renderer(canvas), reason: gate.reason };
}

/** A same-id, same-class clone in the old canvas's place, with no context yet. */
function replaceCanvas(old: HTMLCanvasElement): HTMLCanvasElement {
  const fresh = document.createElement('canvas');
  for (const attr of Array.from(old.attributes)) fresh.setAttribute(attr.name, attr.value);
  old.replaceWith(fresh);
  return fresh;
}
