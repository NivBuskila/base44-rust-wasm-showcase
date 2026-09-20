/**
 * The taught poses as landmarks, so the target hand is drawn by the same
 * skeleton renderer as the caster's own hand.
 *
 * Each finger is declared as a direction plus a curl **per joint** (and a length
 * scale, because a curled finger is foreshortened in projection), then expanded
 * into MediaPipe's 21 points. Angles are degrees from straight up, positive to
 * the right; the hand is drawn mirrored like the stage, so the thumb sits on the
 * left and a thumb laid across the fist points right.
 *
 * It is schematic on purpose — a target to compare a live skeleton against, not
 * an anatomical model. The vendored glyphs stay for the reference panel.
 */

/** Wrist, then the base of each finger: thumb, index, middle, ring, pinky. */
const WRIST: readonly [number, number] = [0.5, 0.95];
const BASES: readonly (readonly [number, number])[] = [
  [0.35, 0.78],
  [0.4, 0.55],
  [0.5, 0.52],
  [0.6, 0.54],
  [0.68, 0.6],
];
/** Segment lengths per finger, base → tip. */
const LENGTHS: readonly (readonly number[])[] = [
  [0.1, 0.08, 0.06],
  [0.13, 0.09, 0.06],
  [0.14, 0.1, 0.07],
  [0.13, 0.09, 0.06],
  [0.11, 0.07, 0.05],
];

interface Finger {
  /** Direction of the first segment. */
  dir: number;
  /** Bend added at each joint, base → tip. */
  curl: readonly [number, number, number];
  /**
   * Per-segment length multiplier. Seen from the front a folded phalanx points
   * at the camera, so it must be drawn *short*, not swung out sideways —
   * without this a fist grew hooks off the side of the hand.
   */
  seg?: readonly [number, number, number];
}

/** A pose is five fingers, thumb first. */
type Pose = readonly [Finger, Finger, Finger, Finger, Finger];

/** Extended: nearly straight, with the natural slight bend. */
const up = (dir: number): Finger => ({ dir, curl: [5, 6, 8] });
/**
 * Curled into the palm: the knuckle rides up, then the finger folds straight
 * back down so the tip sits just under its own knuckle — a compact stub, the
 * way a fist reads from the front.
 */
const folded = (dir: number): Finger => ({
  dir,
  curl: [170, 15, 10],
  seg: [0.85, 0.3, 0.5],
});
/** Thumb tucked along the side of a curled hand. */
const THUMB_TUCKED: Finger = { dir: 34, curl: [16, 14, 8], seg: [0.9, 0.9, 0.9] };

/**
 * Some poses are not a front-facing hand at all — a fist and a thumbs-up are
 * seen from the side, so the knuckles stack instead of arching and the fingers
 * fold sideways. Those two are traced straight from real captured landmarks
 * (21 points, wrist first) rather than generated, because the finger-chain
 * model only knows the front view.
 */
const TRACED: Record<string, readonly (readonly number[])[]> = {
  // closed fist, seen from the thumb side, thumb wrapped over the fingers
  attract: [
    [0.33, 0.88],
    [0.36, 0.76],
    [0.4, 0.66],
    [0.46, 0.6],
    [0.5, 0.62],
    [0.4, 0.58],
    [0.44, 0.44],
    [0.38, 0.42],
    [0.36, 0.5],
    [0.46, 0.57],
    [0.5, 0.42],
    [0.44, 0.4],
    [0.42, 0.49],
    [0.51, 0.58],
    [0.56, 0.46],
    [0.5, 0.45],
    [0.48, 0.53],
    [0.55, 0.62],
    [0.6, 0.52],
    [0.55, 0.51],
    [0.53, 0.58],
  ],
  // thumbs up: hand turned sideways, thumb a long vertical, fingers folded right
  shatter: [
    [0.38, 0.92],
    [0.4, 0.8],
    [0.42, 0.6],
    [0.44, 0.4],
    [0.45, 0.24],
    [0.5, 0.62],
    [0.7, 0.57],
    [0.62, 0.54],
    [0.52, 0.56],
    [0.5, 0.7],
    [0.72, 0.67],
    [0.63, 0.64],
    [0.53, 0.66],
    [0.5, 0.78],
    [0.7, 0.76],
    [0.62, 0.73],
    [0.53, 0.75],
    [0.49, 0.86],
    [0.66, 0.85],
    [0.59, 0.82],
    [0.51, 0.84],
  ],
};

const POSES: Record<string, Pose> = {
  // pinch: the index curls right back onto the thumb tip
  vortex: [
    { dir: -16, curl: [-6, -8, -6] },
    { dir: -24, curl: [-78, -86, -80], seg: [1, 0.95, 0.95] },
    up(2),
    up(14),
    up(26),
  ],
  // index straight up, the rest folded away
  ignite: [THUMB_TUCKED, up(-8), folded(2), folded(10), folded(16)],
  // index and middle up in a V
  freeze: [THUMB_TUCKED, up(-24), up(6), folded(10), folded(16)],
  // all five spread, palm flat to the camera
  repel: [{ dir: -48, curl: [6, 10, 8] }, up(-18), up(-2), up(12), up(28)],
  // open hand, wider still
  release: [{ dir: -56, curl: [4, 8, 6] }, up(-24), up(-4), up(16), up(34)],
};

/** Expands one finger declaration into its four points from `base`. */
function chain(
  base: readonly [number, number],
  finger: Finger,
  lengths: readonly number[],
): number[][] {
  const seg = finger.seg ?? [1, 1, 1];
  const pts: number[][] = [[base[0], base[1]]];
  let angle = finger.dir;
  lengths.forEach((len, i) => {
    const a = (angle * Math.PI) / 180;
    const [x, y] = pts[pts.length - 1];
    const l = len * seg[i];
    pts.push([x + Math.sin(a) * l, y - Math.cos(a) * l]);
    angle += finger.curl[i];
  });
  return pts;
}

/** The 21 landmarks of a taught pose, or `null` when the spell has no pose. */
export function poseLandmarks(spell: string): number[][] | null {
  const traced = TRACED[spell];
  if (traced) return traced.map(([x, y]) => [x, y]);
  const pose = POSES[spell];
  if (!pose) return null;
  const pts: number[][] = [[WRIST[0], WRIST[1]]];
  pose.forEach((finger, i) => {
    for (const p of chain(BASES[i], finger, LENGTHS[i])) pts.push(p);
  });
  return pts;
}
