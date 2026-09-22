/**
 * "Raise a hand to enter": the welcome's button charges while the camera sees
 * a hand and opens the field when full, so the first gesture a new user makes
 * is the one that lets them in. Losing the hand drains the charge at twice the
 * rate it fills, so a hand passing through the frame never opens it.
 */

/** How long a hand must be held up to enter. */
const HOLD_MS = 1200;
const TICK_MS = 50;

export interface HandWatch {
  stop(): void;
}

/**
 * Polls `hands` and reports the charge (0..1) through `onCharge` whenever it
 * moves; calls `onFull` once and stops when it reaches 1.
 */
export function watchHand(
  hands: () => number,
  onCharge: (charge: number, seen: boolean) => void,
  onFull: () => void,
): HandWatch {
  let charge = 0;
  const timer = window.setInterval(() => {
    const seen = hands() > 0;
    const next = Math.min(1, Math.max(0, charge + (seen ? 1 : -2) * (TICK_MS / HOLD_MS)));
    if (next === charge) return;
    charge = next;
    onCharge(charge, seen);
    if (charge === 1) {
      window.clearInterval(timer);
      onFull();
    }
  }, TICK_MS);
  return { stop: () => window.clearInterval(timer) };
}
