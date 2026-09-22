/**
 * The intro console: the boot screen and the first-run welcome as one surface.
 *
 * It mounts over `#boot` before anything loads as the compact loader (the
 * wordmark and a live boot log). Once the engine runs the backdrop turns
 * translucent so the field wakes behind it and, on a first visit, a single
 * line and "Enter the field" unfold under the centred header; the button
 * irises it open. With a camera running, holding a hand up fills the
 * button and opens it too (`landing-hand.ts`). The HUD's first-run moments
 * wait for `onOpen`, so nothing plays unseen behind the console.
 *
 * First visit only (`aether.landing.seen` in `localStorage` — clear it to see
 * it again): later visits stay compact and open on their own when boot
 * finishes. After it opens, `#boot` stays attached as
 * `.done` (hidden, no pointer events) so it can never eat a gesture.
 */

import { watchHand, type HandWatch } from './landing-hand';

const SEEN_KEY = 'aether.landing.seen';
/** Must match the iris animation's duration in `landing.css`. */
const EXIT_MS = 1150;

export interface ReadyOptions {
  /** Hands in view, when a camera is running: lets a raised hand enter. */
  hands?: () => number;
  /** The console has fully opened onto the stage. */
  onOpen?: () => void;
}

export interface Intro {
  status(text: string): void;
  /** The engine is running: unfold the welcome (or open, on a return visit). */
  ready(options?: ReadyOptions): void;
  fail(message: string): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function isFirstRun(): boolean {
  // `?welcome` replays the full welcome without clearing storage.
  if (new URLSearchParams(location.search).has('welcome')) return true;
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

/**
 * The field lens: the glow and ripples track the pointer, and the key leans
 * toward it. The rect is measured once on entry (the lean itself would skew
 * a live read) and writes are batched to one per frame.
 */
function magnetise(button: HTMLButtonElement): void {
  let rect: DOMRect | null = null;
  let frame = 0;
  let x = 0.5;
  let y = 0.5;

  const write = () => {
    frame = 0;
    button.style.setProperty('--mx', `${x * 100}%`);
    button.style.setProperty('--my', `${y * 100}%`);
    if (reducedMotion()) return;
    button.style.setProperty('--tx', `${(x - 0.5) * 14}px`);
    button.style.setProperty('--ty', `${(y - 0.5) * 10}px`);
  };

  button.addEventListener('pointerenter', () => {
    rect = button.getBoundingClientRect();
  });
  button.addEventListener('pointermove', (event) => {
    if (!rect || !rect.width || !rect.height) return;
    x = (event.clientX - rect.left) / rect.width;
    y = (event.clientY - rect.top) / rect.height;
    if (!frame) frame = requestAnimationFrame(write);
  });
  button.addEventListener('pointerleave', () => {
    rect = null;
    cancelAnimationFrame(frame);
    frame = 0;
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

  if (full) {
    frame.append(el('p', 'landing-lede', 'Your hands shape a living fluid. It is waking up behind this panel.'));
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
    enter.append(
      el('span', 'landing-enter-charge'),
      el('span', 'landing-enter-ripples'),
      label,
      arrow,
      el('span', 'landing-enter-sheen'),
    );
    const cta = el('div', 'landing-cta');
    cta.append(enter, el('span', 'landing-hand', 'or raise a hand to the camera'));
    foot.append(cta, el('p', 'landing-note', 'camera optional · H controls · T tutorial'));
  }
  frame.append(foot);
  root.append(frame);

  return { state, statusEl, enter, label, anchors: [titleRow, state, log] };
}

/**
 * Runs `change` and glides each node from where it was to where it lands, so
 * the loader's header and log move into the welcome instead of the console
 * appearing a second time. Transform only, on the compositor.
 */
function glide(nodes: HTMLElement[], change: () => void): void {
  const before = nodes.map((node) => node.getBoundingClientRect());
  change();
  if (reducedMotion()) return;
  nodes.forEach((node, i) => {
    const after = node.getBoundingClientRect();
    const dx = before[i].left - after.left;
    const dy = before[i].top - after.top;
    if (!dx && !dy) return;
    node.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], {
      duration: 700,
      easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
    });
  });
}

/** Mounts the intro console into `#boot`, replacing its static fallback. */
export function mountIntro(resumed = false): Intro {
  const root = document.getElementById('boot') ?? document.body.appendChild(el('div', ''));
  root.id = 'boot';
  // The prerendered loader (index.html) already played the entrance, and after
  // the isolation reload it must not play again: never replay it here.
  root.classList.toggle('resumed', resumed || root.childElementCount > 0);
  const full = isFirstRun();
  const ui = build(root, full);
  let opened = false;
  let onOpen: (() => void) | undefined;
  let watch: HandWatch | null = null;

  /** Irises the console open from (x, y), surging the stage behind it. */
  const open = (x: number, y: number) => {
    if (opened) return;
    opened = true;
    watch?.stop();
    remember();
    if (reducedMotion()) {
      root.classList.add('done');
      onOpen?.();
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
      onOpen?.();
    }, EXIT_MS);
  };

  const openFromButton = () => {
    if (!ui.enter || ui.enter.disabled) return;
    const r = ui.enter.getBoundingClientRect();
    open(r.left + r.width / 2, r.top + r.height / 2);
  };

  /** A held hand fills the button and opens it — the first gesture is the way in. */
  const watchForHand = (hands: () => number) => {
    const fill = root.querySelector<HTMLElement>('.landing-enter-charge');
    const hint = root.querySelector<HTMLElement>('.landing-hand');
    if (!fill || !hint) return;
    root.classList.add('hand-ready');
    watch = watchHand(
      hands,
      (charge, seen) => {
        fill.style.setProperty('--charge', String(charge));
        root.classList.toggle('hand-seen', seen);
        hint.textContent = seen ? 'hand seen — hold it there…' : 'or raise a hand to the camera';
      },
      openFromButton,
    );
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
    ready(options = {}) {
      onOpen = options.onOpen;
      root.classList.add('ready');
      ui.state.textContent = 'ONLINE';
      ui.statusEl.textContent = 'field stable — all systems nominal';
      if (ui.enter && ui.label) {
        // The console stays centred: the header and log only glide up to make
        // room for the lede and the button, while the field wakes behind.
        glide(ui.anchors, () => root.classList.add('revealed'));
        ui.enter.disabled = false;
        ui.enter.focus();
        scramble(ui.label, 'ENTER THE FIELD');
        if (options.hands) watchForHand(options.hands);
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
