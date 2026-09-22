/**
 * The keyboard map, with no DOM state of its own.
 *
 * Kept apart from `hud.ts` because the two rules that matter here are pure
 * decisions, not side effects: a modified chord is never a shortcut, and a key
 * pressed inside a control belongs to the control — dragging a slider with the
 * arrow keys must not toggle the camera. Everything the keys *do* stays in the
 * HUD; this only says which action a key asked for.
 */

import { MODES } from './hud-spec';
import type { ViewMode } from './types';

/** What a recognised key asked for. `mode` is the only one carrying a value. */
export type HudAction =
  | { kind: 'mode'; mode: ViewMode }
  | { kind: 'camera' }
  | { kind: 'overdrive' }
  | { kind: 'hideAll' }
  | { kind: 'reset' }
  | { kind: 'tutorial' }
  | { kind: 'help' }
  | { kind: 'closeHelp' };

/** True when the event's target is somewhere text or a value is being edited. */
function inControl(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable === true
  );
}

/**
 * Maps a keydown to an action, or `null` when the HUD should keep its hands
 * off the event. `helpOpen` is needed for Escape alone: with the sheet shut,
 * Escape belongs to the browser.
 */
export function hudAction(
  e: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey' | 'target'>,
  helpOpen: boolean,
): HudAction | null {
  if (e.metaKey || e.ctrlKey || e.altKey) return null;
  if (inControl(e.target)) return null;

  const mode = MODES.find((m) => m.key === e.key);
  if (mode) return { kind: 'mode', mode: mode.mode };
  switch (e.key) {
    case 'c':
    case 'C':
      return { kind: 'camera' };
    case 'o':
    case 'O':
      return { kind: 'overdrive' };
    case 'h':
    case 'H':
      return { kind: 'hideAll' };
    case 'r':
    case 'R':
      return { kind: 'reset' };
    case 't':
    case 'T':
      return { kind: 'tutorial' };
    case '?':
    case '/':
      return { kind: 'help' };
    case 'Escape':
      return helpOpen ? { kind: 'closeHelp' } : null;
    default:
      return null;
  }
}
