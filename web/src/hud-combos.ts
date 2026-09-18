/**
 * The spellbook overlay: which gesture sequences exist, and how far the caster
 * has got through one right now.
 *
 * Its own component rather than more rows in {@link Hud} for two reasons. It is
 * the only part of the HUD that must stay visible while the panel is collapsed —
 * a combo you cannot see the progress of is a combo nobody lands — and its
 * content is defined by the engine, not by this file: the book arrives from
 * `AetherEngine.combo_book()` as `name:step,step;...`, so adding a combo in Rust
 * makes it appear here with no TypeScript change.
 *
 * Everything it touches is per-frame, so updates are diffed against the last
 * painted state: writing the same class 60 times a second is how a HUD ends up
 * costing more than the simulation it describes.
 */

/** One combo, as parsed out of the engine's book string. */
interface Combo {
  readonly name: string;
  readonly steps: readonly string[];
}

/** Progress as packed by `AetherEngine.combo_progress`. */
export interface ComboState {
  /** Combo being cast, or -1. */
  readonly active: number;
  readonly matched: number;
  /** 0..1 through the step currently being held. */
  readonly charge: number;
  /** Combo that just completed, or -1. */
  readonly fired: number;
}

/** Short hand shape for each spell, so the row reads as instructions. */
const HANDS: Record<string, string> = {
  attract: 'fist',
  repel: 'palm',
  vortex: 'pinch',
  ignite: 'point',
  freeze: 'victory',
  shatter: 'thumb',
  release: 'open fist',
};

export function parseBook(book: string): Combo[] {
  return book
    .split(';')
    .map((entry) => entry.split(':'))
    .filter((parts): parts is [string, string] => parts.length === 2 && parts[0].length > 0)
    .map(([name, steps]) => ({ name, steps: steps.split(',').filter((s) => s.length > 0) }))
    .filter((c) => c.steps.length > 1);
}

/** Reads {@link ComboState} out of the engine's packed progress array. */
export function comboState(packed: ArrayLike<number> | undefined): ComboState {
  const at = (i: number, fallback: number): number => {
    const v = packed?.[i];
    return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  };
  return {
    active: at(0, -1),
    matched: at(1, 0),
    charge: Math.min(1, Math.max(0, at(2, 0))),
    fired: at(3, -1),
  };
}

export class ComboBook {
  private readonly el: HTMLElement;
  private combos: Combo[] = [];
  /** Row element and its step pips, in book order. */
  private rows: { root: HTMLElement; pips: HTMLElement[] }[] = [];
  private painted = '';

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'hud-combos';
    this.el.hidden = true;
    this.el.setAttribute('aria-hidden', 'true');
    parent.appendChild(this.el);
  }

  /**
   * Rebuilds the rows from the engine's book. Cheap to call repeatedly: an
   * unchanged book is ignored, because the caller has it per frame and the DOM
   * cost of rebuilding it would be charged to every one of them.
   */
  setBook(book: string): void {
    if (book === this.painted) return;
    this.painted = book;
    this.combos = parseBook(book);
    this.rows = [];
    this.el.textContent = '';
    if (this.combos.length === 0) {
      this.el.hidden = true;
      return;
    }

    const title = document.createElement('b');
    title.className = 'cb-title';
    title.textContent = 'sequences';
    this.el.appendChild(title);

    for (const combo of this.combos) {
      const row = document.createElement('div');
      row.className = 'cb-row';
      row.dataset.combo = combo.name;

      const name = document.createElement('span');
      name.className = 'cb-name';
      name.textContent = combo.name;
      row.appendChild(name);

      const pips = document.createElement('span');
      pips.className = 'cb-pips';
      const pipEls: HTMLElement[] = [];
      combo.steps.forEach((step, i) => {
        if (i > 0) {
          const arrow = document.createElement('i');
          arrow.className = 'cb-arrow';
          arrow.setAttribute('aria-hidden', 'true');
          pips.appendChild(arrow);
        }
        const pip = document.createElement('i');
        pip.className = 'cb-pip';
        // The hand shape, not the spell: the caster performs shapes.
        pip.textContent = HANDS[step] ?? step;
        pips.appendChild(pip);
        pipEls.push(pip);
      });
      row.appendChild(pips);
      this.el.appendChild(row);
      this.rows.push({ root: row, pips: pipEls });
    }
    this.el.hidden = false;
  }

  /** Paints one frame of progress. */
  update(state: ComboState): void {
    for (let i = 0; i < this.rows.length; i++) {
      const { root, pips } = this.rows[i];
      const active = i === state.active;
      const fired = i === state.fired;
      const rowState = fired ? 'fired' : active ? 'casting' : 'idle';
      if (root.dataset.state !== rowState) root.dataset.state = rowState;

      for (let p = 0; p < pips.length; p++) {
        // A fired combo lights every step: the reward has to be legible in the
        // same glance as the gesture that completed it.
        const done = fired || (active && p < state.matched);
        const holding = active && !fired && p === state.matched;
        const pipState = done ? 'done' : holding ? 'hold' : 'off';
        const pip = pips[p];
        if (pip.dataset.state !== pipState) pip.dataset.state = pipState;
        // Only the pip being held carries a charge, so the others never write.
        const charge = holding ? `${Math.round(state.charge * 100)}%` : '';
        if (pip.style.getPropertyValue('--charge') !== charge) {
          pip.style.setProperty('--charge', charge);
        }
      }
    }
  }
}
