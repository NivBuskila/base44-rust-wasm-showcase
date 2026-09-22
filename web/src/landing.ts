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
}

/** What a new user needs to know before their first gesture. */
const CAPABILITIES: Capability[] = [
  {
    title: 'Rust + WebAssembly fluid',
    body: 'A grid fluid solver and a particle pool step in WebAssembly, multithreaded where the browser allows it.',
  },
  {
    title: 'Hands and body as input',
    body: 'Webcam hand and pose tracking drives the field directly — no mouse, no controls to learn.',
  },
  {
    title: 'Spells, combos and duets',
    body: 'Poses cast attract, push, shatter and release; chains and two-hand moves unlock bigger effects.',
  },
  {
    title: 'WebGL2 or WebGPU',
    body: 'The renderer picks the best backend available and adapts quality live to hold a smooth frame rate.',
  },
];

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

  const card = document.createElement('div');
  card.className = 'landing-card';

  const title = document.createElement('h1');
  title.textContent = 'AETHER';

  const lede = document.createElement('p');
  lede.className = 'landing-lede';
  lede.textContent =
    'A gesture-driven fluid reality engine, running entirely in this browser. It is already simulating behind this panel.';

  const grid = document.createElement('div');
  grid.className = 'landing-grid';
  for (const cap of CAPABILITIES) {
    const item = document.createElement('div');
    item.className = 'landing-item';
    const heading = document.createElement('h2');
    heading.textContent = cap.title;
    const body = document.createElement('p');
    body.textContent = cap.body;
    item.append(heading, body);
    grid.append(item);
  }

  const enter = document.createElement('button');
  enter.type = 'button';
  enter.className = 'landing-enter';
  enter.textContent = 'Enter the field';

  const note = document.createElement('p');
  note.className = 'landing-note';
  note.textContent =
    'Allow camera access to cast with your hands — without it Aether runs an ambient simulation. Press H any time for controls, T for the gesture tutorial.';

  card.append(title, lede, grid, enter, note);
  root.append(card);
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
