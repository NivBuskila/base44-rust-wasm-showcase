/**
 * Drawn hands for the tutorial: a sketched right hand, posed per spell.
 *
 * The look is a hand-drawn diagram — a dashed outline traced around the hand,
 * the way a printed instruction card draws a gesture — not a filled icon. The
 * contour itself comes from `hand-outlines.ts`, so a pose stays a small
 * declaration (which fingers are out, where the thumb goes) instead of a wall of
 * path data nobody can safely edit. Everything lives in one 120x150 box with a
 * fixed finger layout, which is what makes two poses read as *the same hand*
 * doing something different — the thing a static icon cannot show.
 *
 * Pure strings: no DOM and no state, so the unit tests can read them.
 */

import { type Digit, digitOutline, handOutline } from './hand-outlines';

/** Where a finger sits on the knuckle line, how far it reaches, how thick. */
interface Finger {
  readonly x: number;
  readonly base: number;
  readonly tip: number;
  readonly tipX: number;
  readonly hw: number;
}

/** Index, middle, ring, little. The hand faces the viewer, thumb on the left. */
const FINGERS: readonly Finger[] = [
  { x: 48, base: 72, tip: 22, tipX: 46, hw: 6.5 },
  { x: 63, base: 68, tip: 14, tipX: 63, hw: 6.8 },
  { x: 77, base: 70, tip: 20, tipX: 80, hw: 6.2 },
  { x: 90, base: 76, tip: 38, tipX: 95, hw: 5.4 },
];

export type ThumbPose = 'curl' | 'side' | 'up' | 'pinch';

export interface HandPose {
  /** Extended state of index, middle, ring, little. */
  readonly out: readonly [boolean, boolean, boolean, boolean];
  readonly thumb: ThumbPose;
  /** Optional extra marks: the accent the pose is *about*. */
  readonly accent?: string;
}

/** Thumbs, by pose. `pinch` is drawn on its own so the tip gap survives. */
const THUMBS: Record<ThumbPose, Digit> = {
  side: { baseX: 40, baseY: 102, tipX: 13, tipY: 73, hw: 7.5 },
  curl: { baseX: 39, baseY: 102, tipX: 23, tipY: 71, hw: 7.5 },
  up: { baseX: 38, baseY: 100, tipX: 25, tipY: 32, hw: 8 },
  pinch: { baseX: 38, baseY: 100, tipX: 41, tipY: 54, hw: 7.5 },
};

/** A folded finger leaves its knuckle above the palm; an extended one reaches. */
function digit(f: Finger, out: boolean): Digit {
  return out
    ? { baseX: f.x, baseY: f.base, tipX: f.tipX, tipY: f.tip, hw: f.hw }
    : { baseX: f.x, baseY: f.base, tipX: f.x, tipY: f.base - 11, hw: f.hw };
}

/** The palm creases, drawn lighter: what tells a sketch from a cut-out. */
const CREASES =
  '<path class="ha-crease" d="M42 96q16 7 34 3"/>' +
  '<path class="ha-crease" d="M46 112q12 5 26 1"/>';

function pose(p: HandPose): string {
  const thumb = THUMBS[p.thumb];
  const fingers = FINGERS.map((f, i) =>
    p.thumb === 'pinch' && i === 0
      ? // The pinching index bends down to meet the thumb tip.
        { baseX: f.x, baseY: f.base, tipX: 44, tipY: 50, hw: f.hw }
      : digit(f, p.out[i]),
  );
  const contour =
    p.thumb === 'pinch'
      ? `<path d="${handOutline(fingers)}"/><path d="${digitOutline(thumb)}"/>`
      : `<path d="${handOutline([thumb, ...fingers])}"/>`;
  return `<g class="ha-ink">${contour}${CREASES}</g>${p.accent ?? ''}`;
}

/** A 120x150 drawing, sized by CSS. The dash pattern is the sketch. */
function svg(body: string, extraClass = ''): string {
  return (
    `<svg viewBox="0 0 120 150" class="ha ${extraClass}" aria-hidden="true" focusable="false" ` +
    'fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" ' +
    `stroke-linejoin="round">${body}</svg>`
  );
}

const FIST: HandPose = { out: [false, false, false, false], thumb: 'curl' };
const OPEN: HandPose = { out: [true, true, true, true], thumb: 'side' };

/**
 * Two frames of the same hand with a drawn arrow between them, for the poses
 * that are a *movement* rather than a shape.
 */
function motion(from: HandPose, to: HandPose): string {
  return (
    `<g transform="translate(-16 24) scale(0.6)">${pose(from)}</g>` +
    '<g class="ha-arrow" opacity="0.8">' +
    '<path d="M50 84q8-5 16 0"/><path d="M62 80l6 4-6 4"/></g>' +
    `<g transform="translate(50 24) scale(0.6)">${pose(to)}</g>`
  );
}

/**
 * One drawing per tutorial step, keyed by spell name. The keys match `GESTURES`
 * in `hud-spec.ts`; {@link handArt} returns `''` for a spell with no drawing, so
 * adding a spell degrades to the plain icon rather than breaking the card.
 */
const ART: Record<string, string> = {
  attract: svg(pose(FIST), 'ha-pull'),
  repel: svg(pose(OPEN), 'ha-push'),
  vortex: svg(
    pose({
      out: [false, true, true, true],
      thumb: 'pinch',
      accent: '<circle class="ha-spark" cx="42" cy="47" r="4" opacity="0.9"/>',
    }),
    'ha-spin',
  ),
  ignite: svg(pose({ out: [true, false, false, false], thumb: 'curl' }), 'ha-trail'),
  freeze: svg(pose({ out: [true, true, false, false], thumb: 'curl' })),
  shatter: svg(pose({ out: [false, false, false, false], thumb: 'up' }), 'ha-pop'),
  release: svg(motion(FIST, OPEN), 'ha-burst'),
  warp: svg(motion(OPEN, { out: [true, true, true, true], thumb: 'up' })),
};

/** The drawing for a spell, or `''` when there is none. */
export function handArt(spell: string): string {
  return ART[spell] ?? '';
}
