/**
 * Animated hands for the tutorial: every pose is a little loop, not a still.
 *
 * The hands themselves are a real icon family (`hand-icons.ts`, Fluent Emoji high
 * contrast) rather than geometry generated here — two attempts at drawing hands
 * from finger declarations produced shapes nobody reads as a hand, and a real
 * hand drawing shows *which fingers are folded*, which is the whole lesson.
 *
 * This module is only the mapping and the composition. Each drawing is two
 * stacked frames — the hand at rest and the hand in the taught pose — that
 * cross-fade on a loop, so the card shows the *movement* into the pose (fingers
 * closing, the thumb coming in) instead of a shape to decode. The card draws a
 * charge ring around it while the pose is held.
 *
 * Pure strings: no DOM and no state, so the unit tests can read them.
 */

import { HAND_ICONS, ICON_BOX } from './hand-icons';

/** One glyph, filled, on the shared canvas. */
function glyph(key: string, cls: string): string {
  return `<path class="ha-hand ${cls}" d="${HAND_ICONS[key]}"/>`;
}

/** A drawing on the glyph canvas, sized by CSS. */
function svg(body: string, extraClass = ''): string {
  return (
    `<svg viewBox="0 0 ${ICON_BOX} ${ICON_BOX}" class="ha ${extraClass}" aria-hidden="true" ` +
    `focusable="false" fill="currentColor">${body}</svg>`
  );
}

/**
 * The two frames of a pose, stacked and cross-fading (CSS drives the timing, the
 * second frame simply runs half a period behind the first). `from` is where the
 * hand starts, `to` is the pose being taught, so the loop reads as the gesture.
 */
function loop(from: string, to: string): string {
  return glyph(from, 'ha-f1') + glyph(to, 'ha-f2');
}

/**
 * One drawing per tutorial step, keyed by spell name. The keys match `GESTURES`
 * in `hud-spec.ts`; {@link handArt} returns `''` for a spell with no drawing, so
 * adding a spell degrades to the plain icon rather than breaking the card.
 */
const ART: Record<string, string> = {
  attract: svg(loop('palm', 'fist'), 'ha-pull'),
  repel: svg(loop('fist', 'palm'), 'ha-push'),
  vortex: svg(loop('palm', 'pinch'), 'ha-spin'),
  ignite: svg(loop('fist', 'point'), 'ha-trail'),
  freeze: svg(loop('fist', 'peace')),
  shatter: svg(loop('fist', 'thumb'), 'ha-pop'),
  release: svg(loop('fist', 'palm'), 'ha-burst'),
  warp: svg(loop('palm', 'thumb')),
};

/** The drawing for a spell, or `''` when there is none. */
export function handArt(spell: string): string {
  return ART[spell] ?? '';
}
