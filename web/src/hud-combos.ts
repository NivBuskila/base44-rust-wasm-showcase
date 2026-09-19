/**
 * The spellbook overlay: which gesture sequences and two-hand duets exist, and
 * how far the caster has got through one right now.
 *
 * Its own component rather than more rows in {@link Hud} for two reasons. It is
 * the only part of the HUD that must stay visible while the panel is collapsed —
 * a combo you cannot see the progress of is a combo nobody lands — and its
 * content is defined by the engine, not by this file: the books arrive from
 * `AetherEngine.combo_book()` / `duet_book()` as `name:step,step;...`, so adding
 * a combo or duet in Rust makes it appear here with no TypeScript change.
 *
 * Everything it touches is per-frame, so updates are diffed against the last
 * painted state: writing the same class 60 times a second is how a HUD ends up
 * costing more than the simulation it describes.
 */

/** One combo or duet, as parsed out of an engine book string. */
interface Entry {
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

/** Progress as packed by `AetherEngine.duet_progress`. */
export interface DuetState {
  /** Duet being wound up, or -1. */
  readonly active: number;
  /** 0..1 through the wind-up. */
  readonly charge: number;
  /** Duet that just fired, or -1. */
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

/**
 * A step token as the caster performs it. A duet step names both hands at
 * once as `spell+spell`; anything the engine does not name as a spell (a
 * motion word like `clap`) is shown verbatim.
 */
function stepLabel(step: string): string {
  return step
    .split('+')
    .map((s) => HANDS[s] ?? s)
    .join(' + ');
}

function parseEntries(book: string, minSteps: number): Entry[] {
  return book
    .split(';')
    .map((entry) => entry.split(':'))
    .filter((parts): parts is [string, string] => parts.length === 2 && parts[0].length > 0)
    .map(([name, steps]) => ({ name, steps: steps.split(',').filter((s) => s.length > 0) }))
    .filter((c) => c.steps.length >= minSteps);
}

export function parseBook(book: string): Entry[] {
  // A one-step combo is just a gesture.
  return parseEntries(book, 2);
}

export function parseDuetBook(book: string): Entry[] {
  // A one-step duet is legitimate: a held pair of hands.
  return parseEntries(book, 1);
}

function at(packed: ArrayLike<number> | undefined, i: number, fallback: number): number {
  const v = packed?.[i];
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** Reads {@link ComboState} out of the engine's packed progress array. */
export function comboState(packed: ArrayLike<number> | undefined): ComboState {
  return {
    active: at(packed, 0, -1),
    matched: at(packed, 1, 0),
    charge: Math.min(1, Math.max(0, at(packed, 2, 0))),
    fired: at(packed, 3, -1),
  };
}

/** Reads {@link DuetState} out of the engine's packed progress array. */
export function duetState(packed: ArrayLike<number> | undefined): DuetState {
  return {
    active: at(packed, 0, -1),
    charge: Math.min(1, Math.max(0, at(packed, 1, 0))),
    fired: at(packed, 2, -1),
  };
}

interface Row {
  readonly root: HTMLElement;
  readonly pips: HTMLElement[];
}

/** One titled group of rows, built once from a book and painted per frame. */
class Section {
  readonly el: HTMLElement;
  private rows: Row[] = [];
  private painted = '';

  constructor(
    private readonly title: string,
    private readonly parse: (book: string) => Entry[],
  ) {
    this.el = document.createElement('div');
    this.el.className = 'cb-section';
    this.el.hidden = true;
  }

  /** True when the section has at least one row. */
  get present(): boolean {
    return this.rows.length > 0;
  }

  /** Number of steps in row `i`, 0 when there is no such row. */
  stepCount(i: number): number {
    return this.rows[i]?.pips.length ?? 0;
  }

  setBook(book: string): boolean {
    if (book === this.painted) return false;
    this.painted = book;
    const entries = this.parse(book);
    this.rows = [];
    this.el.textContent = '';
    if (entries.length === 0) {
      this.el.hidden = true;
      return true;
    }

    const title = document.createElement('b');
    title.className = 'cb-title';
    title.textContent = this.title;
    this.el.appendChild(title);

    for (const entry of entries) {
      const row = document.createElement('div');
      row.className = 'cb-row';
      row.dataset.combo = entry.name;

      const name = document.createElement('span');
      name.className = 'cb-name';
      name.textContent = entry.name;
      row.appendChild(name);

      const pips = document.createElement('span');
      pips.className = 'cb-pips';
      const pipEls: HTMLElement[] = [];
      entry.steps.forEach((step, i) => {
        if (i > 0) {
          const arrow = document.createElement('i');
          arrow.className = 'cb-arrow';
          arrow.setAttribute('aria-hidden', 'true');
          pips.appendChild(arrow);
        }
        const pip = document.createElement('i');
        pip.className = 'cb-pip';
        // The hand shape, not the spell: the caster performs shapes.
        pip.textContent = stepLabel(step);
        pips.appendChild(pip);
        pipEls.push(pip);
      });
      row.appendChild(pips);
      this.el.appendChild(row);
      this.rows.push({ root: row, pips: pipEls });
    }
    this.el.hidden = false;
    return true;
  }

  /**
   * Paints one frame. `matched` is how many steps of the active row are done;
   * the step at that index is the one being held and carries `charge`.
   */
  paint(active: number, matched: number, charge: number, fired: number): void {
    for (let i = 0; i < this.rows.length; i++) {
      const { root, pips } = this.rows[i];
      const isActive = i === active;
      const isFired = i === fired;
      const rowState = isFired ? 'fired' : isActive ? 'casting' : 'idle';
      if (root.dataset.state !== rowState) root.dataset.state = rowState;

      for (let p = 0; p < pips.length; p++) {
        // A fired entry lights every step: the reward has to be legible in the
        // same glance as the gesture that completed it.
        const done = isFired || (isActive && p < matched);
        const holding = isActive && !isFired && p === matched;
        const pipState = done ? 'done' : holding ? 'hold' : 'off';
        const pip = pips[p];
        if (pip.dataset.state !== pipState) pip.dataset.state = pipState;
        // Only the pip being held carries a charge, so the others never write.
        const fill = holding ? `${Math.round(charge * 100)}%` : '';
        if (pip.style.getPropertyValue('--charge') !== fill) {
          pip.style.setProperty('--charge', fill);
        }
      }
    }
  }
}

/**
 * The projectiles. These are recognised from motion and finger geometry rather
 * than latched spells, so the engine has no book for them; the instructions
 * are static and are never painted with progress.
 */
const SHOTS_BOOK = 'throw:palm,flick;gun:finger gun,thumb down';

export class ComboBook {
  private readonly el: HTMLElement;
  private readonly sequences = new Section('sequences', parseBook);
  private readonly duets = new Section('duets', parseDuetBook);
  private readonly shots = new Section('shots', parseDuetBook);

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'hud-combos';
    this.el.hidden = true;
    this.el.setAttribute('aria-hidden', 'true');
    this.el.appendChild(this.sequences.el);
    this.el.appendChild(this.duets.el);
    this.el.appendChild(this.shots.el);
    this.shots.setBook(SHOTS_BOOK);
    parent.appendChild(this.el);
    this.refreshVisibility();
  }

  /**
   * Rebuilds the sequence rows from the engine's book. Cheap to call
   * repeatedly: an unchanged book is ignored, because the caller has it per
   * frame and the DOM cost of rebuilding it would be charged to every one of
   * them.
   */
  setBook(book: string): void {
    if (this.sequences.setBook(book)) this.refreshVisibility();
  }

  /** Same as {@link setBook}, for the two-hand duets. */
  setDuetBook(book: string): void {
    if (this.duets.setBook(book)) this.refreshVisibility();
  }

  private refreshVisibility(): void {
    this.el.hidden = !(this.sequences.present || this.duets.present || this.shots.present);
  }

  /** Paints one frame of progress. */
  update(combos: ComboState, duets: DuetState = duetState(undefined)): void {
    this.sequences.paint(combos.active, combos.matched, combos.charge, combos.fired);
    // A duet has no step counter: the whole wind-up is the charge on its last
    // step, and every step before it is shown done, because the caster is
    // already holding both hands in the required shape when it starts.
    const held = Math.max(0, this.duets.stepCount(duets.active) - 1);
    this.duets.paint(duets.active, held, duets.charge, duets.fired);
  }
}
