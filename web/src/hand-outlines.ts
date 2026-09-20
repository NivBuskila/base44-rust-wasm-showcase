/**
 * Hand contours: the geometry behind the tutorial's drawn hands.
 *
 * A pose is declared as a few digits (where each one starts, where it points,
 * how thick it is) and this module traces ONE closed outline around them — up
 * the left side of a digit, around the tip, down the right side, into the web,
 * then on to the next, and back along the palm. A single continuous contour is
 * what makes the sketch style possible: a dashed stroke only reads as a drawn
 * hand when it follows the hand's silhouette instead of a pile of overlapping
 * capsules.
 *
 * Pure geometry and strings: no DOM, no state.
 */

/** One digit: a straight segment of a given thickness, base to tip. */
export interface Digit {
  readonly baseX: number;
  readonly baseY: number;
  readonly tipX: number;
  readonly tipY: number;
  /** Half width — also the tip's rounding radius. */
  readonly hw: number;
}

interface Pt {
  readonly x: number;
  readonly y: number;
}

interface Geom {
  readonly hw: number;
  /** Base and tip on the left flank, then on the right flank. */
  readonly lb: Pt;
  readonly lt: Pt;
  readonly rt: Pt;
  readonly rb: Pt;
  /** Control points that give each flank a slight outward bulge. */
  readonly cpL: Pt;
  readonly cpR: Pt;
}

function n(v: number): string {
  return (Math.round(v * 10) / 10).toString();
}

function p(pt: Pt): string {
  return `${n(pt.x)} ${n(pt.y)}`;
}

/**
 * Flank offsets for one digit. The normal is taken as `(dy, -dx)`, which for a
 * digit pointing up is the left side — the direction the contour is traced in.
 */
function geom(d: Digit): Geom {
  const dx = d.tipX - d.baseX;
  const dy = d.tipY - d.baseY;
  const len = Math.hypot(dx, dy) || 1;
  const nx = dy / len;
  const ny = -dx / len;
  const base = { x: d.baseX, y: d.baseY };
  const tip = { x: d.tipX, y: d.tipY };
  const mid = { x: (base.x + tip.x) / 2, y: (base.y + tip.y) / 2 };
  const off = (s: number, at: Pt): Pt => ({ x: at.x + nx * s, y: at.y + ny * s });
  // A little extra width at mid-length: real fingers are not straight tubes.
  const bulge = 1.6;
  return {
    hw: d.hw,
    lb: off(d.hw, base),
    lt: off(d.hw, tip),
    rt: off(-d.hw, tip),
    rb: off(-d.hw, base),
    cpL: off(d.hw + bulge, mid),
    cpR: off(-(d.hw + bulge), mid),
  };
}

/** The digit's own flanks and tip, traced left flank → tip → right flank. */
function flanks(g: Geom): string {
  return (
    `Q${p(g.cpL)} ${p(g.lt)}` +
    `A${n(g.hw)} ${n(g.hw)} 0 0 1 ${p(g.rt)}` +
    `Q${p(g.cpR)} ${p(g.rb)}`
  );
}

/**
 * One closed outline around the given digits (ordered left to right) plus the
 * palm and wrist. The palm is fixed in the 120x150 box so every pose shares it.
 */
export function handOutline(digits: readonly Digit[]): string {
  const gs = digits.map(geom);
  const first = gs[0];
  const last = gs[gs.length - 1];
  let d = `M${p(first.lb)}`;
  gs.forEach((g, i) => {
    d += flanks(g);
    const next = gs[i + 1];
    if (next) {
      // The web between two digits dips below both knuckles.
      const web = { x: (g.rb.x + next.lb.x) / 2, y: Math.max(g.rb.y, next.lb.y) + 5 };
      d += `Q${p(web)} ${p(next.lb)}`;
    }
  });
  // Little-finger edge down, across the wrist, and back up the thumb edge.
  d +=
    `Q${n(last.rb.x + 7)} ${n(last.rb.y + 14)} 101 106` +
    'Q99 131 77 135L57 135' +
    `Q34 130 33 105Q33 95 ${p(first.lb)}Z`;
  return d;
}

/**
 * A standalone digit outline, for a thumb that must stay clear of the fingers —
 * the pinch, where the gap between the two tips *is* the gesture.
 */
export function digitOutline(digit: Digit): string {
  const g = geom(digit);
  return `M${p(g.lb)}${flanks(g)}A${n(g.hw)} ${n(g.hw)} 0 0 1 ${p(g.lb)}Z`;
}
