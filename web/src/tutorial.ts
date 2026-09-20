/**
 * Interactive gesture tutorial.
 *
 * Walks through the single-hand spells one at a time and advances only when
 * the engine itself reports the spell latched — the card teaches what the
 * recogniser accepts, not what a drawing suggests. Each step is passed by
 * holding the spell for {@link HOLD_MS}; `release` is a one-frame event in
 * `spells.rs`, so it passes on sight.
 *
 * Starts on its own the first time a hand is seen in a session that has never
 * finished it (remembered in `localStorage`), and can be reopened from the
 * panel or `T` at any time. Everything here is presentation: the spell names
 * come straight from `AetherEngine.spell_name`.
 */

import { GESTURES, type GestureSpec } from './hud-spec';

/** How long a spell must stay latched before the step is passed. */
export const HOLD_MS = 700;
/** The "passed" flash before the next step slides in. */
const PASS_MS = 650;
/** How long the completion card lingers before closing itself. */
const DONE_MS = 6000;
const STORAGE_KEY = 'aether.tutorial.done';

/** Warp is two-handed and read from the time scale, not a spell name. */
export const STEPS: readonly GestureSpec[] = GESTURES.filter((g) => g.spell !== 'warp');

function svg(paths: string): string {
  return (
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    `stroke-linejoin="round">${paths}</svg>`
  );
}

function remembered(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function remember(): void {
  try {
    localStorage.setItem(STORAGE_KEY, '1');
  } catch {
    /* private mode: the tutorial simply offers itself again next time */
  }
}

/**
 * Pure step logic, separate from the DOM so it can be unit-tested: feeds the
 * current spell names and returns the charge through the current step.
 */
export class TutorialProgress {
  index = 0;
  /** `performance.now()` when the current spell was first seen; null when not held. */
  private heldSince: number | null = null;

  constructor(private readonly steps: readonly GestureSpec[] = STEPS) {}

  get step(): GestureSpec | undefined {
    return this.steps[this.index];
  }

  get done(): boolean {
    return this.index >= this.steps.length;
  }

  /** Charge 0..1; `1` means the step just passed and `index` has advanced. */
  feed(spells: readonly string[], now: number): number {
    const step = this.step;
    if (!step) return 1;
    const latched = spells.includes(step.spell);
    if (!latched) {
      this.heldSince = null;
      return 0;
    }
    if (step.spell === 'release') return this.pass();
    if (this.heldSince === null) this.heldSince = now;
    const t = (now - this.heldSince) / HOLD_MS;
    return t >= 1 ? this.pass() : t;
  }

  skip(): void {
    this.heldSince = null;
    this.index++;
  }

  private pass(): number {
    this.skip();
    return 1;
  }
}

export class GestureTutorial {
  private readonly el: HTMLElement;
  private readonly count: HTMLElement;
  private readonly icon: HTMLElement;
  private readonly hand: HTMLElement;
  private readonly effect: HTMLElement;
  private readonly fill: HTMLElement;
  private readonly note: HTMLElement;
  private readonly next: HTMLButtonElement;

  private progress = new TutorialProgress();
  private on = false;
  /** Set while the "passed" flash is showing; the next step paints after it. */
  private passUntil = 0;
  private doneAt = 0;
  private lastPct = '';
  private autoStarted = remembered();

  constructor(parent: HTMLElement) {
    this.el = document.createElement('aside');
    this.el.className = 'tut';
    this.el.hidden = true;
    this.el.setAttribute('aria-live', 'polite');
    this.el.innerHTML = `
      <header class="tut-head">
        <span class="tut-tag">tutorial</span>
        <span class="tut-count" data-tut-count></span>
        <button type="button" class="tut-btn" data-tut="skip">skip all</button>
      </header>
      <div class="tut-body">
        <span class="tut-icon" data-tut-icon></span>
        <div class="tut-text">
          <b data-tut-hand></b>
          <span data-tut-effect></span>
        </div>
        <button type="button" class="tut-btn" data-tut="next">next ›</button>
      </div>
      <div class="tut-track"><i class="tut-fill" data-tut-fill></i></div>
      <span class="tut-note" data-tut-note></span>`;
    parent.appendChild(this.el);
    this.count = this.el.querySelector('[data-tut-count]')!;
    this.icon = this.el.querySelector('[data-tut-icon]')!;
    this.hand = this.el.querySelector('[data-tut-hand]')!;
    this.effect = this.el.querySelector('[data-tut-effect]')!;
    this.fill = this.el.querySelector('[data-tut-fill]')!;
    this.note = this.el.querySelector('[data-tut-note]')!;
    this.next = this.el.querySelector('[data-tut="next"]')!;

    this.el.querySelector('[data-tut="skip"]')!.addEventListener('click', () => this.stop());
    this.next.addEventListener('click', () => {
      if (this.progress.done) {
        this.stop();
        return;
      }
      this.progress.skip();
      if (this.progress.done) this.paintDone(performance.now());
      else this.paintStep();
    });
  }

  get active(): boolean {
    return this.on;
  }

  start(): void {
    this.progress = new TutorialProgress();
    this.passUntil = 0;
    this.doneAt = 0;
    this.on = true;
    this.autoStarted = true;
    this.el.hidden = false;
    this.el.classList.remove('is-in');
    void this.el.offsetWidth;
    this.el.classList.add('is-in');
    this.paintStep();
  }

  /** Closes the card. Finishing or dismissing both count as "seen". */
  stop(): void {
    this.on = false;
    this.el.hidden = true;
    remember();
  }

  toggle(): void {
    if (this.on) this.stop();
    else this.start();
  }

  /**
   * Once per frame. `hands` is the engine's `HANDS_PRESENT` stat, used only to
   * start the tutorial the first time someone actually shows a hand.
   */
  update(spells: readonly string[], hands: number, now: number): void {
    if (!this.on) {
      if (!this.autoStarted && hands > 0) this.start();
      return;
    }
    if (this.progress.done) {
      if (this.doneAt > 0 && now - this.doneAt > DONE_MS) this.stop();
      return;
    }
    if (this.passUntil > 0) {
      if (now < this.passUntil) return;
      this.passUntil = 0;
      this.paintStep();
    }
    const charge = this.progress.feed(spells, now);
    if (charge >= 1) {
      this.el.classList.add('is-pass');
      this.setPct('100%');
      if (this.progress.done) this.paintDone(now + PASS_MS);
      else this.passUntil = now + PASS_MS;
      return;
    }
    this.setPct(`${Math.round(charge * 100)}%`);
  }

  private paintStep(): void {
    const step = this.progress.step;
    if (!step) return;
    this.el.classList.remove('is-pass', 'is-done');
    this.count.textContent = `${this.progress.index + 1} / ${STEPS.length}`;
    this.icon.innerHTML = svg(step.icon);
    this.hand.textContent = step.hand;
    this.effect.textContent = `${step.spell} · ${step.effect}`;
    this.note.textContent =
      step.spell === 'release'
        ? 'make a fist, hold it a moment, then open it'
        : 'show one hand to the camera and hold the pose';
    this.next.textContent = 'next ›';
    this.setPct('0%');
  }

  private paintDone(at: number): void {
    this.doneAt = at;
    this.el.classList.add('is-done');
    this.count.textContent = `${STEPS.length} / ${STEPS.length}`;
    this.hand.textContent = 'you know the spells';
    this.effect.textContent = 'chain them into combos — the book is in the panel';
    this.note.textContent = 'press ? for two-hand duets, throws and the finger gun';
    this.next.textContent = 'done';
    this.setPct('100%');
  }

  private setPct(pct: string): void {
    if (pct === this.lastPct) return;
    this.lastPct = pct;
    this.fill.style.setProperty('--v', pct);
  }
}
