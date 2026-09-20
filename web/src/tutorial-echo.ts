/**
 * Live visual echo for the gesture tutorial.
 *
 * Pure presentation: a soft light ring over the stage whose brightness and
 * breath follow the step's charge (`--vn`), so holding a pose visibly pushes
 * energy into the scene, plus a one-shot bloom when the spell latches. It never
 * touches the engine — the fluid keeps being driven only by real gestures.
 */
export class TutorialEcho {
  private readonly el: HTMLElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'tut-echo';
    this.el.hidden = true;
    this.el.setAttribute('aria-hidden', 'true');
    parent.appendChild(this.el);
  }

  show(on: boolean): void {
    this.el.hidden = !on;
    if (!on) this.el.classList.remove('is-burst');
  }

  /** Charge 0..1 through the current step. */
  setCharge(charge: number): void {
    this.el.style.setProperty('--vn', `${charge}`);
  }

  /** One-shot bloom on a latched spell. */
  burst(): void {
    this.el.classList.remove('is-burst');
    void this.el.offsetWidth;
    this.el.classList.add('is-burst');
  }
}
