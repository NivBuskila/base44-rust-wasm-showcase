/**
 * The perception status line's wording, as data instead of DOM writes.
 *
 * `hud.ts` paints it at 10 Hz and diffs on a signature string, so the text has
 * to be derivable from the status alone — that is the whole reason this is a
 * pure function living outside the panel.
 */

import type { PerceptionStatus } from './types';

/** The three fields the status line renders: heading, sub-line, tone flag. */
export interface StatusLine {
  head: string;
  why: string;
  tone: 'ok' | 'wait' | 'warn';
}

/** The wording for one perception status. */
export function statusLine(p: PerceptionStatus): StatusLine {
  if (p.kind === 'ready') {
    return {
      head: 'hand + body tracking live',
      why:
        p.delegate === 'GPU' ? 'GPU delegate' : 'CPU delegate — inference will cost more per frame',
      tone: 'ok',
    };
  }
  if (p.kind === 'loading') {
    return {
      head: 'loading the vision models',
      why: 'optical flow is already driving the fluid',
      tone: 'wait',
    };
  }
  return {
    head: 'optical flow only',
    why: `${p.reason} Move, and the fluid still answers.`,
    tone: 'warn',
  };
}
