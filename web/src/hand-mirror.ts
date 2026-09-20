/**
 * Live pose mirror for the tutorial.
 *
 * Draws the caster's own hand skeleton, from the packed landmark buffer the
 * perception layer already produces, inside the charge ring — and carries the
 * hold timer, filling from the wrist up as the pose is held.
 *
 * It is a view: it reads landmarks and writes SVG points, never the other way
 * round. Drawing lives in `hand-skeleton.ts`, shared with the taught pose beside
 * it, and x is flipped here because the stage shows a mirrored camera.
 */

import { HAND_LANDMARKS, HAND_STRIDE } from './constants';
import { fit, HandSkeleton } from './hand-skeleton';

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

/** The skeleton overlay inside the tutorial's ring. */
export class HandMirror {
  private readonly skeleton: HandSkeleton;
  private live = false;

  constructor(parent: HTMLElement) {
    this.skeleton = new HandSkeleton(parent, { charge: true, className: 'tut-mirror' });
  }

  /** Feeds one frame of landmarks; `null` fades the skeleton out. */
  update(hands: Float32Array | null): void {
    const raw = slotPoints(hands);
    if (!raw) {
      if (this.live) {
        this.live = false;
        this.skeleton.el.classList.remove('is-live');
      }
      return;
    }
    this.skeleton.draw(fit(raw));
    if (!this.live) {
      this.live = true;
      this.skeleton.el.classList.add('is-live');
    }
  }

  /** Charge 0..1: how much of the hand is filled, from the wrist upwards. */
  setCharge(charge: number): void {
    this.skeleton.setCharge(charge);
  }
}
