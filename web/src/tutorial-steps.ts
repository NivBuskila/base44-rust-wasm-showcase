/**
 * What the tutorial teaches, in order, and how each pose is made.
 *
 * Separate from `tutorial.ts` so the lesson content can be edited (and unit
 * tested) without touching the card's DOM or its timing.
 */

import { GESTURES, type GestureSpec } from './hud-spec';

/**
 * Teaching order. Warp is two-handed and read from the time scale rather than a
 * spell name, so it stays in the reference sheet.
 *
 * The order also avoids putting two spells that open a combo back to back
 * (`attract,repel` = nova, `repel,vortex` = tempest, `attract,freeze,…` =
 * supernova). While the tutorial is open the engine is in practice mode and
 * casts no sequences at all, so this is belt and braces — but it also means the
 * order still reads sanely if practice mode is ever turned off.
 */
const ORDER: readonly string[] = [
  'attract',
  'vortex',
  'ignite',
  'shatter',
  'freeze',
  'repel',
  'release',
];

export const STEPS: readonly GestureSpec[] = ORDER.map(
  (spell) => GESTURES.find((g) => g.spell === spell)!,
);

/**
 * How to actually form the shape, one line per step. The drawing shows it; this
 * says it, because a drawing of a curled hand is ambiguous about the thumb and
 * the thumb is usually what the recogniser is unhappy about.
 */
export const HOW_TO: Record<string, string> = {
  attract: 'curl all four fingers in and lay the thumb across them',
  vortex: 'touch the tip of your thumb to the tip of your index finger',
  ignite: 'index finger straight up, the other three folded away',
  shatter: 'fist, then the thumb straight up out of it',
  freeze: 'index and middle finger up in a V, the rest folded',
  repel: 'all five fingers spread, palm flat towards the camera',
  release: 'squeeze a fist for a moment, then open the hand wide',
};
