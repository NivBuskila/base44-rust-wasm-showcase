/**
 * Drawn hands for the tutorial: a silhouette right hand, posed per spell.
 *
 * The drawings are generated rather than hand-authored so a pose is a small
 * declaration (which fingers are out, where the thumb goes) instead of a wall
 * of path data nobody can safely edit. Everything lives in one 120x150 box with
 * a fixed finger layout, which is what makes two poses read as *the same hand*
 * doing something different — the thing a static icon cannot show.
 *
 * Anatomy comes from volume, not outline: fingers are round-capped capsules
 * with real thickness and a knuckle bend, the palm carries a thumb-side bulge,
 * and the whole hand renders as one flat silhouette so overlapping parts never
 * show a seam. Pure strings: no DOM and no state, so the unit tests can read
 * them.
 */

/** Where a finger sits along the knuckle line, how long it is, how thick. */
interface Finger {
  /** Knuckle x. */
  readonly x: number;
  /** Knuckle y — the middle finger's knuckle is the highest. */
  readonly base: number;
  /** Tip y when extended. */
  readonly tip: number;
  /** Capsule width: index and middle are the thickest, little the thinnest. */
  readonly w: number;
  /** Sideways drift of the tip, so the fingers splay instead of running parallel. */
  readonly splay: number;
}

/** Index, middle, ring, little. The hand faces the viewer, thumb on the left. */
const FINGERS: readonly Finger[] = [
  { x: 48, base: 70, tip: 24, w: 13, splay: -3 },
  { x: 63, base: 66, tip: 16, w: 13.5, splay: 0 },
  { x: 77, base: 68, tip: 22, w: 12.5, splay: 3 },
  { x: 89, base: 74, tip: 38, w: 10.5, splay: 6 },
];

export type ThumbPose = 'curl' | 'side' | 'up' | 'pinch';

export interface HandPose {
  /** Extended state of index, middle, ring, little. */
  readonly out: readonly [boolean, boolean, boolean, boolean];
  readonly thumb: ThumbPose;
  /** Optional extra marks: the accent the pose is *about*. */
  readonly accent?: string;
}

/**
 * The palm, same in every pose: the anchor that makes the poses comparable.
 * The left edge swells at the thumb base (the thenar pad) and the wrist tapers.
 */
const PALM =
  '<path d="M36 80c0-12 6-18 14-19h40c8 1 14 7 14 19v22' +
  'c0 17-5 27-13 30h-38c-9-3-15-11-16-25' +
  'c-6-5-9-13-8-19 1-5 4-8 7-8z"/>';

/** A capsule: a round-capped stroke of the finger's own width. */
function capsule(d: string, w: number): string {
  return `<path d="${d}" stroke-width="${w}" fill="none"/>`;
}

/** An extended finger: a capsule with a slight knuckle bend and outward splay. */
function extended(f: Finger): string {
  const mid = (f.base + f.tip) / 2;
  const tipX = f.x + f.splay;
  return capsule(`M${f.x} ${f.base}Q${f.x} ${mid} ${tipX} ${f.tip}`, f.w);
}

/**
 * A curled finger: the knuckle bump left above the palm when the finger folds
 * in, which is what makes a fist read as a fist rather than a hand with no
 * fingers.
 */
function curled(f: Finger): string {
  const top = f.base - 11;
  return capsule(`M${f.x} ${f.base}V${top}`, f.w);
}

function thumbPath(pose: ThumbPose): string {
  switch (pose) {
    // Folded across the front of the fingers: the fist.
    case 'curl':
      return capsule('M40 96Q30 92 34 82Q38 73 50 72', 15);
    // Out to the side, in the plane of the palm: the open hand.
    case 'side':
      return capsule('M38 98Q24 96 16 88Q9 81 12 72', 15);
    // Straight up alongside the fist: thumb up.
    case 'up':
      return capsule('M38 94Q28 88 26 72Q25 54 28 40', 15);
    // Reaching up to meet the index tip: the pinch.
    case 'pinch':
      return capsule('M38 96Q26 86 28 68Q30 54 38 48', 15);
  }
}

function pose(p: HandPose): string {
  const fingers = FINGERS.map((f, i) => {
    if (p.thumb === 'pinch' && i === 0) {
      // The pinching index comes down to meet the thumb instead of folding
      // into the palm: the gap between the two tips *is* the gesture.
      return capsule(`M${f.x} ${f.base}Q${f.x} 48 ${f.x - 8} 42`, f.w);
    }
    return p.out[i] ? extended(f) : curled(f);
  }).join('');
  // One flat silhouette: the group's own opacity hides every internal overlap.
  return (
    '<g class="ha-body" fill="currentColor" stroke="currentColor" ' +
    `stroke-linecap="round" stroke-linejoin="round">${fingers}${PALM}${thumbPath(p.thumb)}</g>` +
    (p.accent ?? '')
  );
}

/** A 120x150 drawing, sized by CSS. */
function svg(body: string, extraClass = ''): string {
  return (
    `<svg viewBox="0 0 120 150" class="ha ${extraClass}" aria-hidden="true" focusable="false" ` +
    `fill="none" stroke="none">${body}</svg>`
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
    `<g transform="translate(-14 22) scale(0.62)">${pose(from)}</g>` +
    '<g class="ha-arrow" fill="none" stroke="currentColor" stroke-width="3" ' +
    'stroke-linecap="round" stroke-linejoin="round" opacity="0.75">' +
    '<path d="M52 86h13"/><path d="M61 81l5 5-5 5"/></g>' +
    `<g transform="translate(48 22) scale(0.62)">${pose(to)}</g>`
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
      accent:
        '<circle class="ha-spark" cx="38" cy="44" r="4.5" fill="currentColor" opacity="0.9"/>',
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
