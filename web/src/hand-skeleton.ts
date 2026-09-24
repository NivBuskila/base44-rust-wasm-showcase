/**
 * The one hand drawing used by the tutorial.
 *
 * A skeleton — MediaPipe's 21 landmarks as polylines — drawn the same way
 * whether the points come from the camera ({@link HandMirror}) or from the
 * taught pose ({@link posePoints}). One renderer, so "your hand" and "the pose"
 * can never drift into two different graphic languages.
 *
 * Optionally carries the hold timer: a second copy of the same polylines in the
 * accent colour, revealed from the wrist up by a clip rect that rises with the
 * charge.
 */

/** The drawing's square canvas; every skeleton is fitted into it. */
export const ICON_BOX = 32;

/** MediaPipe hand topology, as polylines: one line per finger plus the palm arch. */
export const CHAINS: readonly (readonly number[])[] = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
  [5, 9, 13, 17],
];

/** Fraction of the box left as margin, so the skeleton never touches the ring. */
const PAD = 0.12;

/**
 * Fits landmarks into the glyph box, preserving aspect: the drawing must keep
 * the hand's proportions or a spread palm reads as a fist.
 */
export function fit(pts: readonly (readonly number[])[]): number[][] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-4);
  const scale = (ICON_BOX * (1 - 2 * PAD)) / span;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const half = ICON_BOX / 2;
  return pts.map(([x, y]) => [half + (x - cx) * scale, half + (y - cy) * scale]);
}

const NS = 'http://www.w3.org/2000/svg';

/** One skeleton drawing inside a round tile. */
export class HandSkeleton {
  readonly el: SVGSVGElement;
  private readonly lines: SVGPolylineElement[];
  private readonly fills: SVGPolylineElement[];
  private readonly clip: SVGRectElement | null = null;
  private readonly wrist: SVGCircleElement;
  private charge = -1;

  constructor(parent: HTMLElement, opts: { charge?: boolean; className?: string } = {}) {
    this.el = document.createElementNS(NS, 'svg');
    this.el.setAttribute('class', opts.className ?? 'tut-skeleton');
    this.el.setAttribute('viewBox', `0 0 ${ICON_BOX} ${ICON_BOX}`);
    this.el.setAttribute('aria-hidden', 'true');
    this.el.setAttribute('focusable', 'false');
    this.lines = CHAINS.map(() => this.line('tut-mirror-line', this.el));
    if (opts.charge) {
      const clipId = `tut-clip-${Math.random().toString(36).slice(2, 8)}`;
      const defs = document.createElementNS(NS, 'defs');
      const clipPath = document.createElementNS(NS, 'clipPath');
      clipPath.setAttribute('id', clipId);
      this.clip = document.createElementNS(NS, 'rect');
      this.clip.setAttribute('x', '0');
      this.clip.setAttribute('width', String(ICON_BOX));
      clipPath.appendChild(this.clip);
      defs.appendChild(clipPath);
      this.el.appendChild(defs);
      const group = document.createElementNS(NS, 'g');
      group.setAttribute('clip-path', `url(#${clipId})`);
      this.fills = CHAINS.map(() => this.line('tut-mirror-fill', group));
      this.el.appendChild(group);
      this.setCharge(0);
    } else {
      this.fills = [];
    }
    // The wrist anchors the reading: it says which way the hand is turned.
    this.wrist = document.createElementNS(NS, 'circle');
    this.wrist.setAttribute('class', 'tut-mirror-wrist');
    this.wrist.setAttribute('r', '3.5');
    this.el.appendChild(this.wrist);
    parent.appendChild(this.el);
  }

  private line(cls: string, parent: SVGElement): SVGPolylineElement {
    const line = document.createElementNS(NS, 'polyline');
    line.setAttribute('class', cls);
    parent.appendChild(line);
    return line;
  }

  /** Draws 21 already-fitted points. */
  draw(pts: readonly (readonly number[])[]): void {
    CHAINS.forEach((chain, i) => {
      const points = chain.map((n) => `${pts[n][0].toFixed(1)},${pts[n][1].toFixed(1)}`).join(' ');
      this.lines[i].setAttribute('points', points);
      this.fills[i]?.setAttribute('points', points);
    });
    this.wrist.setAttribute('cx', pts[0][0].toFixed(1));
    this.wrist.setAttribute('cy', pts[0][1].toFixed(1));
  }

  /** Charge 0..1: how much of the hand is filled, from the wrist upwards. */
  setCharge(charge: number): void {
    if (!this.clip) return;
    const v = Math.min(1, Math.max(0, charge));
    if (v === this.charge) return;
    this.charge = v;
    const h = ICON_BOX * v;
    this.clip.setAttribute('y', (ICON_BOX - h).toFixed(2));
    this.clip.setAttribute('height', h.toFixed(2));
  }
}
