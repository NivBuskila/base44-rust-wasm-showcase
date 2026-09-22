/**
 * The intro console: the boot screen and the first-run welcome as one surface.
 *
 * It mounts over `#boot` before anything loads as the compact loader (the
 * wordmark and a live boot log). Once the engine runs the backdrop turns
 * translucent and, on a first visit, the console unfolds into the welcome —
 * the capability lines check in one by one and "Enter the field" irises it
 * open from the button.
 *
 * First visit only (`aether.landing.seen` in `localStorage` — clear it to see
 * it again): later visits stay compact and open on their own when boot
 * finishes. After it opens, `#boot` stays attached as
 * `.done` (hidden, no pointer events) so it can never eat a gesture.
 */

const SEEN_KEY = 'aether.landing.seen';
/** Must match the iris animation's duration in `landing.css`. */
const EXIT_MS = 1150;

interface Capability {
  title: string;
  body: string;
  tag: 'ONLINE' | 'READY';
}

/** What a new user needs to know before their first gesture. */
const CAPABILITIES: Capability[] = [
  {
    title: 'Rust + WebAssembly fluid',
    body: 'A grid fluid solver and a particle pool step in WebAssembly, multithreaded where the browser allows it.',
    tag: 'ONLINE',
  },
  {
    title: 'Hands and body as input',
    body: 'Webcam hand and pose tracking drives the field directly — no mouse, no controls to learn.',
    tag: 'READY',
  },
  {
    title: 'Spells, combos and duets',
    body: 'Poses cast attract, push, shatter and release; chains and two-hand moves unlock bigger effects.',
    tag: 'READY',
  },
  {
    title: 'WebGL2 or WebGPU',
    body: 'The renderer picks the best backend available and adapts quality live to hold a smooth frame rate.',
    tag: 'ONLINE',
  },
];

export interface Intro {
  status(text: string): void;
  /** The engine is running: unfold the welcome (or open, on a return visit). */
  ready(): void;
  fail(message: string): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function isFirstRun(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) !== '1';
  } catch {
    return true;
  }
}

function remember(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    /* Private mode: showing it again is a better failure than not starting. */
  }
}

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Decodes `text` into `node` through a short glyph scramble. */
function scramble(node: HTMLElement, text: string, ms = 520): void {
  if (reducedMotion()) {
    node.textContent = text;
    return;
  }
  const glyphs = '▚▞░▒▓<>/\\=+*#01';
  // A timer, not rAF: a hidden tab pauses frames, and the label must still land.
  const start = performance.now();
  const timer = window.setInterval(() => {
    const p = Math.min(1, (performance.now() - start) / ms);
    const locked = Math.floor(p * text.length);
    let out = text.slice(0, locked);
    for (let i = locked; i < text.length; i++) {
      out += text[i] === ' ' ? ' ' : glyphs[(Math.random() * glyphs.length) | 0];
    }
    node.textContent = out;
    if (p === 1) window.clearInterval(timer);
  }, 30);
}

/** The field lens: the glow and ripples track the pointer, and the key leans toward it. */
function magnetise(button: HTMLButtonElement): void {
  button.addEventListener('pointermove', (event) => {
    const r = button.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const x = (event.clientX - r.left) / r.width;
    const y = (event.clientY - r.top) / r.height;
    button.style.setProperty('--mx', `${x * 100}%`);
    button.style.setProperty('--my', `${y * 100}%`);
    if (reducedMotion()) return;
    button.style.setProperty('--tx', `${(x - 0.5) * 14}px`);
    button.style.setProperty('--ty', `${(y - 0.5) * 10}px`);
  });
  button.addEventListener('pointerleave', () => {
    for (const key of ['--mx', '--my', '--tx', '--ty']) button.style.removeProperty(key);
  });
}

/** Builds the console into `root` and returns the nodes the intro drives. */
function build(root: HTMLElement, full: boolean) {
  root.replaceChildren();
  // Everyone waits on the compact loader; the welcome unfolds on `ready`.
  root.classList.add('compact');

  const frame = el('div', 'landing-frame');

  const head = el('div', 'landing-head');
  const titleRow = el('div', 'landing-title');
  titleRow.append(el('span', 'landing-dot'), el('h1', '', 'AETHER'));
  const state = el('span', 'landing-state', 'BOOTING');
  head.append(titleRow, state);
  frame.append(head);

  const tags: HTMLElement[] = [];
  if (full) {
    frame.append(
      el(
        'p',
        'landing-lede',
        'A gesture-driven fluid reality engine, running entirely in this browser. It is coming alive behind this panel.',
      ),
    );
    const grid = el('div', 'landing-grid');
    for (const cap of CAPABILITIES) {
      const item = el('div', 'landing-item');
      const line = el('div', 'landing-line');
      const name = el('div', 'landing-name');
      name.append(el('span', 'landing-check', '✓'), el('span', '', cap.title));
      const tag = el('span', 'landing-tag', 'WAIT');
      tag.dataset.on = cap.tag;
      tags.push(tag);
      line.append(name, tag);
      item.append(line, el('p', '', cap.body));
      grid.append(item);
    }
    frame.append(grid);
  }

  const foot = el('div', 'landing-foot');
  const log = el('p', 'landing-log');
  const statusEl = el('span', '', 'waking the engine…');
  statusEl.id = 'boot-status';
  log.append(el('span', 'landing-prompt', '>'), statusEl, el('span', 'landing-caret'));
  foot.append(log);

  let enter: HTMLButtonElement | null = null;
  let label: HTMLElement | null = null;
  if (full) {
    enter = el('button', 'landing-enter');
    enter.type = 'button';
    enter.disabled = true;
    label = el('span', 'landing-enter-label', 'INITIALISING');
    const arrow = el('span', 'landing-enter-arrow');
    arrow.append(el('span', '', '›'), el('span', '', '›'), el('span', '', '›'));
    enter.append(label, arrow, el('span', 'landing-enter-sheen'));
    foot.append(
      enter,
      el(
        'p',
        'landing-note',
        'Allow camera access to cast with your hands — without it Aether runs an ambient simulation. Press H any time for controls, T for the gesture tutorial.',
      ),
    );
  }
  frame.append(foot);
  root.append(frame);

  return { state, tags, statusEl, enter, label };
}

/** Mounts the intro console into `#boot`, replacing its static fallback. */
export function mountIntro(): Intro {
  const root = document.getElementById('boot') ?? document.body.appendChild(el('div', ''));
  root.id = 'boot';
  const full = isFirstRun();
  const ui = build(root, full);
  let opened = false;

  /** Irises the console open from (x, y), surging the stage behind it. */
  const open = (x: number, y: number) => {
    if (opened) return;
    opened = true;
    remember();
    if (reducedMotion()) {
      root.classList.add('done');
      return;
    }
    root.style.setProperty('--ex', `${x}px`);
    root.style.setProperty('--ey', `${y}px`);
    const shock = el('div', 'landing-shock');
    shock.style.setProperty('--ex', `${x}px`);
    shock.style.setProperty('--ey', `${y}px`);
    shock.append(el('span', ''), el('span', ''));
    document.body.append(shock);
    const stage = document.getElementById('stage');
    stage?.classList.add('surge');
    root.classList.add('entering');
    window.setTimeout(() => {
      root.classList.add('done');
      root.classList.remove('entering');
      shock.remove();
      stage?.classList.remove('surge');
    }, EXIT_MS);
  };

  const openFromButton = () => {
    if (!ui.enter || ui.enter.disabled) return;
    const r = ui.enter.getBoundingClientRect();
    open(r.left + r.width / 2, r.top + r.height / 2);
  };

  if (ui.enter) magnetise(ui.enter);
  ui.enter?.addEventListener('click', openFromButton);
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && root.classList.contains('ready')) openFromButton();
  });

  return {
    status(text) {
      ui.statusEl.textContent = text;
    },
    ready() {
      root.classList.add('ready');
      ui.state.textContent = 'ONLINE';
      ui.statusEl.textContent = 'field stable — all systems nominal';
      if (ui.enter && ui.label) {
        root.classList.remove('compact');
        root.classList.add('revealed');
        // Each line checks in just after its card lands (see the CSS delays).
        ui.tags.forEach((tag, i) => {
          window.setTimeout(() => {
            tag.classList.add('on');
            tag.closest('.landing-item')?.classList.add('on');
            scramble(tag, tag.dataset.on ?? 'ONLINE', 360);
          }, 700 + i * 150);
        });
        ui.enter.disabled = false;
        ui.enter.focus();
        scramble(ui.label, 'ENTER THE FIELD');
      } else {
        window.setTimeout(() => open(innerWidth / 2, innerHeight / 2), 450);
      }
    },
    fail(message) {
      root.classList.add('failed');
      ui.state.textContent = 'FAILED';
      ui.statusEl.textContent = message;
      if (ui.label) ui.label.textContent = 'OFFLINE';
    },
  };
}
