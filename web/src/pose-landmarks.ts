/**
 * The taught poses as landmarks, so the target hand is drawn by the same
 * skeleton renderer as the caster's own hand.
 *
 * Each pose is declared per finger as a direction and a curl, and expanded into
 * MediaPipe's 21 points. This is deliberately schematic — it is a target to
 * compare a skeleton against, not an anatomical model — and it replaces the
 * vendored glyphs on the tutorial card only (the glyphs stay for the panel).
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

/** One finger: `dir` in degrees from straight up (positive = to the right), `curl` per joint. */
interface Finger {
  dir: number;
  curl: number;
}

/** A pose is five fingers, thumb first. */
type Pose = readonly [Finger, Finger, Finger, Finger, Finger];

const up = (dir: number): Finger => ({ dir, curl: 8 });
const folded = (dir: number): Finger => ({ dir, curl: 62 });

/** Fist: four fingers curled, thumb laid across them. */
const FIST: Pose = [
  { dir: -75, curl: 12 },
  folded(-10),
  folded(0),
  folded(10),
  folded(20),
];

const POSES: Record<string, Pose> = {
  // squeeze a fist
  attract: FIST,
  // thumb tip on index tip
  vortex: [
    { dir: -48, curl: -32 },
    { dir: -4, curl: 40 },
    up(2),
    up(14),
    up(26),
  ],
  // index straight up
  ignite: [{ dir: -70, curl: 14 }, up(-6), folded(0), folded(10), folded(20)],
  // fist with the thumb straight up
  shatter: [{ dir: -12, curl: -4 }, folded(-10), folded(0), folded(10), folded(20)],
  // index and middle up in a V
  freeze: [{ dir: -70, curl: 14 }, up(-20), up(8), folded(10), folded(20)],
  // all five spread
  repel: [{ dir: -52, curl: 4 }, up(-16), up(-2), up(12), up(26)],
  release: [{ dir: -58, curl: 2 }, up(-22), up(-4), up(16), up(32)],
};

/** Expands one finger declaration into its four points from `base`. */
function chain(base: readonly [number, number], finger: Finger, lengths: readonly number[]): number[][] {
  const pts: number[][] = [[base[0], base[1]]];
  let angle = finger.dir;
  for (const len of lengths) {
    const a = (angle * Math.PI) / 180;
    const [x, y] = pts[pts.length - 1];
    pts.push([x + Math.sin(a) * len, y - Math.cos(a) * len]);
    angle += finger.curl;
  }
  return pts;
}

/** The 21 landmarks of a taught pose, or `null` when the spell has no pose. */
export function poseLandmarks(spell: string): number[][] | null {
  const pose = POSES[spell];
  if (!pose) return null;
  const pts: number[][] = [[WRIST[0], WRIST[1]]];
  pose.forEach((finger, i) => {
    for (const p of chain(BASES[i], finger, LENGTHS[i])) pts.push(p);
  });
  return pts;
}
