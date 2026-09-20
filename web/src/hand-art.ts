/**
 * Drawn hands for the tutorial: a schematic right hand, posed per spell.
 *
 * The drawings are generated rather than hand-authored so a pose is a small
 * declaration (which fingers are out, where the thumb goes) instead of a wall
 * of path data nobody can safely edit. Everything lives in one 120x150 box with
 * a fixed finger layout, which is what makes two poses read as *the same hand*
 * doing something different — the thing a static icon cannot show.
 *
 * Pure strings: no DOM and no state, so the unit tests can read them.
 */

/** Where a finger sits along the knuckle line and how long it is when out. */
interface Finger {
  /** Knuckle x. */
  readonly x: number;
  /** Knuckle y — the middle finger's knuckle is the highest. */
  readonly base: number;
  /** Tip y when extended. */
  readonly tip: number;
}

/** Index, middle, ring, little. The hand faces the viewer, thumb on the left. */
const FINGERS: readonly Finger[] = [
  { x: 42, base: 66, tip: 20 },
  { x: 57, base: 62, tip: 12 },
  { x: 72, base: 64, tip: 18 },
  { x: 86, base: 70, tip: 32 },
];

export type ThumbPose = 'curl' | 'side' | 'up' | 'pinch';

export interface HandPose {
  /** Extended state of index, middle, ring, little. */
  readonly out: readonly [boolean, boolean, boolean, boolean];
  readonly thumb: ThumbPose;
  /** Optional extra marks: the accent the pose is *about*. */
  readonly accent?: string;
}

/** The palm, same in every pose: the anchor that makes the poses comparable. */
const PALM =
  '<path d="M34 72c0-6 4-10 10-10h44c6 0 10 4 10 10v30c0 16-12 28-28 28h-8' +
  'c-16 0-28-12-28-28z"/>' +
  '<path d="M38 96h52" opacity="0.35"/>';

/** An extended finger: a straight capsule with a soft knuckle bend. */
function extended(f: Finger): string {
  const mid = (f.base + f.tip) / 2;
  return `<path d="M${f.x} ${f.base}V${mid}Q${f.x} ${f.tip} ${f.x} ${f.tip}"/>`;
}

/**
 * A curled finger: over the palm and folded back on itself, which is what makes
 * a fist read as a fist rather than as a hand with no fingers.
 */
function curled(f: Finger): string {
  const knee = f.base - 16;
  return (
    `<path d="M${f.x} ${f.base}V${knee}` +
    `Q${f.x} ${knee - 8} ${f.x - 6} ${knee - 4}` +
    `Q${f.x - 11} ${knee} ${f.x - 6} ${knee + 8}"/>`
  );
}

function thumbPath(pose: ThumbPose): string {
  switch (pose) {
    // Tucked across the palm: the fist.
    case 'curl':
      return '<path d="M34 92q-10 2-12 12t8 12"/>';
    // Out to the side, in the plane of the palm: the open hand.
    case 'side':
      return '<path d="M34 90q-12-2-19 4t-7 13"/>';
    // Straight up, fist below it: thumb up.
    case 'up':
      return '<path d="M34 86q-8-6-9-18t3-20"/>';
    // Reaching the index tip: the pinch.
    case 'pinch':
      return '<path d="M34 90q-8-8-5-18t14-12"/>';
  }
}

function pose(p: HandPose): string {
  const fingers = FINGERS.map((f, i) => {
    if (p.thumb === 'pinch' && i === 0) {
      // The pinching index comes down to meet the thumb instead of curling
      // under the palm: the gap between the two tips *is* the gesture.
      return `<path d="M${f.x} ${f.base}V46q0-10-8-12"/>`;
    }
    return p.out[i] ? extended(f) : curled(f);
  }).join('');
  return PALM + fingers + thumbPath(p.thumb) + (p.accent ?? '');
}

/** A 120x150 drawing, sized by CSS. */
function svg(body: string, extraClass = ''): string {
  return (
    `<svg viewBox="0 0 120 150" class="ha ${extraClass}" aria-hidden="true" focusable="false" ` +
    'fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" ' +
    `stroke-linejoin="round">${body}</svg>`
  );
}

const FIST: HandPose = { out: [false, false, false, false], thumb: 'curl' };
const OPEN: HandPose = { out: [true, true, true, true], thumb: 'side' };

/**
 * Two frames of the same hand with an arrow between them, for the poses that
 * are a *movement* rather than a shape.
 */
function motion(from: HandPose, to: HandPose): string {
  return (
    `<g transform="translate(-18 18) scale(0.66)">${pose(from)}</g>` +
    '<g class="ha-arrow" stroke-width="3" opacity="0.75">' +
    '<path d="M52 86h13"/><path d="M61 81l5 5-5 5"/></g>' +
    `<g transform="translate(46 18) scale(0.66)">${pose(to)}</g>`
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
      accent: '<circle class="ha-spark" cx="39" cy="38" r="4.5" opacity="0.9"/>',
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
