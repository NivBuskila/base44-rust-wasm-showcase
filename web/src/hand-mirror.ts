/**
 * Live pose mirror for the tutorial.
 *
 * Draws the caster's own hand skeleton, from the packed landmark buffer the
 * perception layer already produces, inside the charge ring and on top of the
 * taught glyph — so the lesson becomes a comparison ("my little finger is still
 * out") instead of a shape to decode.
 *
 * It is a view: it reads landmarks and writes SVG points, never the other way
 * round. Geometry is normalised into the glyph box ({@link ICON_BOX}) so the
 * skeleton sits at the size of the drawn hand however far from the camera the
 * caster is, and x is flipped because the stage shows a mirrored camera.
 */

import { HAND_LANDMARKS, HAND_STRIDE } from './constants';
import { ICON_BOX } from './hand-icons';

/** MediaPipe hand topology, as polylines: one line per finger plus the palm arch. */
const CHAINS: readonly (readonly number[])[] = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
  [5, 9, 13, 17],
];

/** Fraction of the box left as margin, so the skeleton never touches the ring. */
const PAD = 0.12;

/** Landmark x/y for one hand slot, or null when that slot is empty. */
function slotPoints(hands: Float32Array | null, slot = 0): number[][] | null {
  if (!hands) return null;
  const base = slot * HAND_STRIDE;
  if (hands.length < base + HAND_STRIDE || hands[base] <= 0) return null;
  const pts: number[][] = [];
  for (let i = 0; i < HAND_LANDMARKS; i++) {
    const o = base + 4 + i * 3;
    // Mirrored: the stage shows the camera flipped, so the skeleton must match.
    pts.push([1 - hands[o], hands[o + 1]]);
  }
  return pts;
}

/**
 * Fits landmarks into the glyph box, preserving aspect: the drawing must keep
 * the hand's proportions or a spread palm reads as a fist.
 */
function fit(pts: number[][]): number[][] {
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

/** The skeleton overlay inside the tutorial's ring. */
export class HandMirror {
  private readonly el: SVGSVGElement;
  private readonly lines: SVGPolylineElement[];
  private readonly fills: SVGPolylineElement[];
  private readonly clip: SVGRectElement;
  private readonly wrist: SVGCircleElement;
  private live = false;
  private charge = -1;

  constructor(parent: HTMLElement) {
    const ns = 'http://www.w3.org/2000/svg';
    this.el = document.createElementNS(ns, 'svg');
    this.el.setAttribute('class', 'tut-mirror');
    this.el.setAttribute('viewBox', `0 0 ${ICON_BOX} ${ICON_BOX}`);
    this.el.setAttribute('aria-hidden', 'true');
    this.el.setAttribute('focusable', 'false');
    // The hold timer *is* the hand: a second copy of the same skeleton, in the
    // accent, revealed from the wrist up by a clip that rises with the charge.
    const clipId = `tut-mirror-clip-${Math.random().toString(36).slice(2, 8)}`;
    const defs = document.createElementNS(ns, 'defs');
    const clipPath = document.createElementNS(ns, 'clipPath');
    clipPath.setAttribute('id', clipId);
    this.clip = document.createElementNS(ns, 'rect');
    this.clip.setAttribute('x', '0');
    this.clip.setAttribute('width', String(ICON_BOX));
    clipPath.appendChild(this.clip);
    defs.appendChild(clipPath);
    this.el.appendChild(defs);
    this.lines = CHAINS.map(() => {
      const line = document.createElementNS(ns, 'polyline');
      line.setAttribute('class', 'tut-mirror-line');
      this.el.appendChild(line);
      return line;
    });
    const fillGroup = document.createElementNS(ns, 'g');
    fillGroup.setAttribute('clip-path', `url(#${clipId})`);
    this.fills = CHAINS.map(() => {
      const line = document.createElementNS(ns, 'polyline');
      line.setAttribute('class', 'tut-mirror-fill');
      fillGroup.appendChild(line);
      return line;
    });
    this.el.appendChild(fillGroup);
    this.setCharge(0);
    // The wrist anchors the reading: it says which way the hand is turned.
    this.wrist = document.createElementNS(ns, 'circle');
    this.wrist.setAttribute('class', 'tut-mirror-wrist');
    this.wrist.setAttribute('r', '3.5');
    this.el.appendChild(this.wrist);
    parent.appendChild(this.el);
  }

  /** Feeds one frame of landmarks; `null` fades the skeleton out. */
  update(hands: Float32Array | null): void {
    const raw = slotPoints(hands);
    if (!raw) {
      if (this.live) {
        this.live = false;
        this.el.classList.remove('is-live');
      }
      return;
    }
    const pts = fit(raw);
    CHAINS.forEach((chain, i) => {
      const points = chain.map((n) => `${pts[n][0].toFixed(1)},${pts[n][1].toFixed(1)}`).join(' ');
      this.lines[i].setAttribute('points', points);
      this.fills[i].setAttribute('points', points);
    });
    this.wrist.setAttribute('cx', pts[0][0].toFixed(1));
    this.wrist.setAttribute('cy', pts[0][1].toFixed(1));
    if (!this.live) {
      this.live = true;
      this.el.classList.add('is-live');
    }
  }

  /** Charge 0..1: how much of the hand is filled, from the wrist upwards. */
  setCharge(charge: number): void {
    const v = Math.min(1, Math.max(0, charge));
    if (v === this.charge) return;
    this.charge = v;
    // Generous overshoot at the top: fingertips sit inside the padded box.
    const h = ICON_BOX * v;
    this.clip.setAttribute('y', (ICON_BOX - h).toFixed(2));
    this.clip.setAttribute('height', h.toFixed(2));
  }
}
