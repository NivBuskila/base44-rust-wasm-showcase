/**
 * Drawn hands for the tutorial: one glyph per taught pose.
 *
 * The hands themselves are a real icon family (`hand-icons.ts`, Phosphor) rather
 * than geometry generated here — two attempts at drawing hands from finger
 * declarations produced shapes nobody reads as a hand, and an icon set gets the
 * proportions right and keeps every pose in the same visual language. This module
 * is only the mapping (which glyph teaches which spell) and the composition: a
 * single pose, or two frames with an arrow when the lesson is a *movement*.
 *
 * Pure strings: no DOM and no state, so the unit tests can read them.
 */

import { HAND_ICONS, ICON_BOX } from './hand-icons';

/** One glyph, filled, on the shared canvas. */
function glyph(key: string): string {
  return `<path class="ha-hand" d="${HAND_ICONS[key]}"/>`;
}

/** A drawing on the glyph canvas, sized by CSS. */
function svg(body: string, extraClass = ''): string {
  return (
    `<svg viewBox="0 0 ${ICON_BOX} ${ICON_BOX}" class="ha ${extraClass}" aria-hidden="true" ` +
    `focusable="false" fill="currentColor">${body}</svg>`
  );
}

/**
 * Two frames of the same hand with an arrow between them, for the poses that are
 * a movement rather than a shape. Both frames are scaled into the one canvas so
 * the tile stays square.
 */
function motion(from: string, to: string): string {
  return (
    `<g transform="translate(-6 38) scale(0.46)">${glyph(from)}</g>` +
    '<g class="ha-arrow" fill="none" stroke="currentColor" stroke-width="14" ' +
    'stroke-linecap="round" stroke-linejoin="round" opacity="0.8">' +
    '<path d="M116 128h22"/><path d="M130 116l12 12-12 12"/></g>' +
    `<g transform="translate(146 38) scale(0.46)">${glyph(to)}</g>`
  );
}

/** A small mark the pose is *about*, drawn beside the hand. */
const SPARK = '<circle class="ha-spark" cx="60" cy="56" r="13" opacity="0.9"/>';

/**
 * One drawing per tutorial step, keyed by spell name. The keys match `GESTURES`
 * in `hud-spec.ts`; {@link handArt} returns `''` for a spell with no drawing, so
 * adding a spell degrades to the plain icon rather than breaking the card.
 */
const ART: Record<string, string> = {
  attract: svg(glyph('fist'), 'ha-pull'),
  repel: svg(glyph('palm'), 'ha-push'),
  vortex: svg(glyph('grab') + SPARK, 'ha-spin'),
  ignite: svg(glyph('point'), 'ha-trail'),
  freeze: svg(glyph('peace')),
  shatter: svg(glyph('thumb'), 'ha-pop'),
  release: svg(motion('fist', 'palm'), 'ha-burst'),
  warp: svg(motion('palm', 'thumb')),
};

/** The drawing for a spell, or `''` when there is none. */
export function handArt(spell: string): string {
  return ART[spell] ?? '';
}
