/**
 * Manual energy shots: click, tap or drag on the stage to fire a bolt into the
 * simulation at that point.
 *
 * Every other effect needs a camera and a hand in frame; this is the one input
 * that always works, so it is bound directly to the canvas rather than to a HUD
 * control. Coordinates are the canvas's own client box normalised to `[0, 1]`,
 * which is the space the engine's `shoot` expects — the same view space the dye
 * texture is drawn in, so a bolt lands exactly under the pointer.
 */

/** Minimum milliseconds between shots while dragging, so a swipe is a volley
 *  rather than one bolt per pointer sample. */
const SHOT_INTERVAL_MS = 90;

export interface EnergyShotTarget {
  shoot(x: number, y: number): void;
}

/**
 * Binds pointer input on `canvas` to `target.shoot`. Returns a disposer.
 */
export function bindEnergyShots(canvas: HTMLCanvasElement, target: EnergyShotTarget): () => void {
  let last = 0;
  let firing = false;

  const fire = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    target.shoot(x, y);
    last = event.timeStamp;
  };

  const onDown = (event: PointerEvent) => {
    firing = true;
    fire(event);
  };

  const onMove = (event: PointerEvent) => {
    if (!firing || event.timeStamp - last < SHOT_INTERVAL_MS) return;
    fire(event);
  };

  const stop = () => {
    firing = false;
  };

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', stop);
  window.addEventListener('pointercancel', stop);

  return () => {
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', stop);
    window.removeEventListener('pointercancel', stop);
  };
}
