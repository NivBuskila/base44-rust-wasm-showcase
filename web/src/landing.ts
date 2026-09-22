/**
 * The first-run welcome overlay.
 *
 * It sits over the live simulation rather than in front of it: the engine is
 * already running and visible behind the panel, so the introduction shows the
 * thing it describes. Dismissal is remembered in `localStorage`
 * (`aether.landing.seen` — clear it to see the overlay again), and the overlay
 * removes itself from the DOM so it can never eat a pointer event that belongs
 * to the gesture surface.
 */

const SEEN_KEY = 'aether.landing.seen';

interface Capability {
  title: string;
  body: string;
  /** The status tag on the capability's boot-log line. */
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

/** An element with a class and optional text. */
function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** True when this browser has not dismissed the overlay before. */
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

function build(): HTMLElement {
  const root = document.createElement('div');
  root.id = 'landing';
  root.setAttribute('role', 'dialog');
  root.setAttribute('aria-label', 'Welcome to Aether');

  // Header: the wordmark behind a pulsing status dot, and the engine state.
  const head = el('div', 'landing-head');
  const titleRow = el('div', 'landing-title');
  titleRow.append(el('span', 'landing-dot'), el('h1', '', 'AETHER'));
  head.append(titleRow, el('span', 'landing-state', 'ONLINE'));

  const lede = el(
    'p',
    'landing-lede',
    'A gesture-driven fluid reality engine, running entirely in this browser. It is already simulating behind this panel.',
  );

  // Each capability is one checked line of the boot log.
  const grid = el('div', 'landing-grid');
  for (const cap of CAPABILITIES) {
    const item = el('div', 'landing-item');
    const line = el('div', 'landing-line');
    const name = el('div', 'landing-name');
    name.append(el('span', 'landing-check', '✓'), el('span', '', cap.title));
    line.append(name, el('span', 'landing-tag', cap.tag));
    item.append(line, el('p', '', cap.body));
    grid.append(item);
  }

  const foot = el('div', 'landing-foot');
  const enter = el('button', 'landing-enter', 'Enter the field');
  enter.type = 'button';
  foot.append(
    enter,
    el(
      'p',
      'landing-note',
      'Allow camera access to cast with your hands — without it Aether runs an ambient simulation. Press H any time for controls, T for the gesture tutorial.',
    ),
  );

  root.append(head, lede, grid, foot);
  return root;
}

/**
 * Shows the welcome overlay on a first visit; a later visit skips it unless
 * `force` is true.
 */
export function showLanding(force = false): void {
  if (!force && !isFirstRun()) return;

  const root = build();
  const dismiss = () => {
    remember();
    root.classList.add('leaving');
    root.addEventListener('transitionend', () => root.remove(), { once: true });
    // Belt and braces: a skipped transition must not leave the overlay behind.
    window.setTimeout(() => root.remove(), 600);
  };

  const enter = root.querySelector<HTMLButtonElement>('.landing-enter');
  enter?.addEventListener('click', dismiss);
  root.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') dismiss();
  });

  document.body.append(root);
  enter?.focus();
}
