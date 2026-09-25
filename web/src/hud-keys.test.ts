// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';

import { hudAction } from './hud-keys';
import { statusLine } from './hud-status';

function key(k: string, extra: Partial<KeyboardEvent> = {}) {
  return { key: k, metaKey: false, ctrlKey: false, altKey: false, target: null, ...extra };
}

describe('hudAction', () => {
  it('maps the single-letter shortcuts', () => {
    expect(hudAction(key('c'), false)).toEqual({ kind: 'camera' });
    expect(hudAction(key('O'), false)).toEqual({ kind: 'overdrive' });
    expect(hudAction(key('h'), false)).toEqual({ kind: 'hideAll' });
    expect(hudAction(key('r'), false)).toEqual({ kind: 'reset' });
    expect(hudAction(key('t'), false)).toEqual({ kind: 'tutorial' });
    expect(hudAction(key('?'), false)).toEqual({ kind: 'help' });
  });

  it('reads a view-mode key as its mode', () => {
    expect(hudAction(key('1'), false)).toEqual({ kind: 'mode', mode: 'aether' });
  });

  it('leaves a modified chord to the browser', () => {
    expect(hudAction(key('r', { metaKey: true }), false)).toBeNull();
    expect(hudAction(key('c', { ctrlKey: true }), false)).toBeNull();
  });

  it('never hijacks a key pressed inside a control', () => {
    const input = document.createElement('input');
    expect(hudAction(key('c', { target: input }), false)).toBeNull();
  });

  it('claims Escape only while the help sheet is open', () => {
    expect(hudAction(key('Escape'), false)).toBeNull();
    expect(hudAction(key('Escape'), true)).toEqual({ kind: 'closeHelp' });
  });

  it('ignores anything unmapped', () => {
    expect(hudAction(key('z'), false)).toBeNull();
  });
});

describe('statusLine', () => {
  it('warns that a CPU delegate costs more', () => {
    expect(statusLine({ kind: 'ready', delegate: 'CPU' }).why).toContain('CPU delegate');
    expect(statusLine({ kind: 'ready', delegate: 'GPU' }).tone).toBe('ok');
  });

  it('says flow is already driving while the models load', () => {
    expect(statusLine({ kind: 'loading' })).toMatchObject({ tone: 'wait' });
  });

  it('carries the failure reason into the flow-only line', () => {
    const line = statusLine({ kind: 'unavailable', reason: 'No camera.' });
    expect(line.tone).toBe('warn');
    expect(line.why).toContain('No camera.');
  });
});
