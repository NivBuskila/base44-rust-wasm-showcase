/**
 * The HUD's one-shot markup. Built exactly once, in the `Hud` constructor;
 * every later update writes `textContent`, `data-*` flags and CSS custom
 * properties into the elements this produces.
 */

import { FLUID_H, FLUID_W } from './constants';
import { PRESETS } from './hud-presets';
import {
  CELLS,
  GESTURES,
  METERS,
  MODES,
  PARAMS,
  PARTICLE_DETENTS,
  SHORTCUTS,
  type CellSpec,
  type GestureSpec,
  type MeterSpec,
  type ParamSpec,
} from './hud-spec';

/** Escapes text destined for the one-shot markup build. */
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function meterRow(m: MeterSpec): string {
  return `
    <div class="m-row" data-meter="${m.id}">
      <span class="m-key">${esc(m.label)}</span>
      <span class="trk"><i class="trk-fill" data-tone="ok"></i></span>
      <span class="m-val">0.00</span>
      <span class="m-unit">ms</span>
    </div>`;
}

function cellBox(c: CellSpec): string {
  return `
    <div class="c-box" data-cell="${c.id}">
      <span>${esc(c.label)}</span>
      <b>—</b>
    </div>`;
}

function handCard(slot: number, name: string): string {
  return `
    <div class="h-card" data-hand="${slot}" data-state="off">
      <span>${esc(name)}</span>
      <b>no hand</b>
    </div>`;
}

/**
 * The legend appears twice — in the panel and in the help sheet. Only the panel
 * copy carries `data-spell`, so the live-highlight selectors stay unambiguous.
 */
function legendRow(g: GestureSpec, live: boolean): string {
  const id = live ? `data-spell="${g.spell}" data-on=""` : '';
  return `
    <li class="g-row" ${id}>
      <span class="g-ico">${svg(g.icon)}</span>
      <span class="g-txt"><b>${esc(g.effect)}</b><i>${esc(g.hand)}</i></span>
    </li>`;
}

function paramRow(p: ParamSpec): string {
  return `
    <label class="p-row">
      <span class="p-key">${esc(p.label)}</span>
      <input
        type="range"
        data-param="${p.key}"
        min="${p.min}"
        max="${p.max}"
        step="${p.step}"
        value="${p.value}"
        aria-label="${esc(p.label)}"
      />
      <span class="p-val" data-param-value="${p.key}">${esc(p.fmt(p.value))}</span>
    </label>`;
}

function svg(paths: string): string {
  return (
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ' +
    `stroke-linejoin="round">${paths}</svg>`
  );
}

/** The whole panel, built once. */
export function markup(): string {
  const presetBtns = PRESETS.map(
    (p) => `
      <button
        type="button"
        class="seg-btn"
        data-preset="${p.id}"
        aria-pressed="${p.id === 'default'}"
        title="${esc(p.hint)}"
      >${esc(p.label)}</button>`,
  ).join('');
  const modeBtns = MODES.map(
    (m) => `
      <button
        type="button"
        class="seg-btn"
        data-mode="${m.mode}"
        aria-pressed="${m.mode === 'aether' ? 'true' : 'false'}"
        title="${esc(m.hint)} (${m.key})"
      >${esc(m.mode)}<span class="seg-key">${m.key}</span></button>`,
  ).join('');

  const keyRows = SHORTCUTS.map(
    ([k, what]) => `<div class="k-row"><kbd>${esc(k)}</kbd><span>${esc(what)}</span></div>`,
  ).join('');

  const chips = GESTURES.map(
    (g) => `<span class="chip">${svg(g.icon)}${esc(g.effect)}</span>`,
  ).join('');

  return `
  <aside class="hud-panel" aria-label="Aether telemetry and controls">
    <header class="hud-head">
      <div class="hud-mark">
        <b>AETHER</b>
        <span>fluid reality engine</span>
      </div>
      <div class="hud-fps">
        <b data-fps data-tone="ok">—</b><span>fps</span>
      </div>
      <div class="hud-acts">
        <button type="button" class="ico-btn" data-act="help" aria-label="Help and keyboard shortcuts" title="Help (?)">
          ${svg('<circle cx="12" cy="12" r="9"/><path d="M9.6 9.2a2.5 2.5 0 1 1 3.6 2.3c-.8.5-1.2 1-1.2 2"/><path d="M12 17.2h.01"/>')}
        </button>
        <button type="button" class="ico-btn" data-act="collapse" aria-expanded="false" aria-label="Expand the panel" title="Collapse or expand">
          ${svg('<path class="chev-d" d="M6 9.5l6 5.5 6-5.5"/><path class="chev-u" d="M6 14.5l6-5.5 6 5.5"/>')}
        </button>
      </div>
      <span class="hud-badge" data-ambient hidden>ambient</span>
      <span class="hud-badge is-engine" data-engine hidden></span>
    </header>

    <div class="hud-body">
      <section class="hud-sec">
        <h2>frame budget<span>16.7 ms @ 60 hz</span></h2>
        ${METERS.map(meterRow).join('')}
        <canvas class="hud-spark" aria-hidden="true"></canvas>
      </section>

      <section class="hud-sec">
        <h2>field</h2>
        <div class="c-grid">${CELLS.map(cellBox).join('')}</div>
      </section>

      <section class="hud-sec">
        <h2>view</h2>
        <div class="seg" role="group" aria-label="View mode">${modeBtns}</div>
        <button type="button" class="row-btn" data-act="camera" aria-pressed="true">
          <span>camera feed</span><b data-camera-value>on</b>
        </button>
        <label class="p-row">
          <span class="p-key">particles</span>
          <input type="range" data-particles min="0" max="${PARTICLE_DETENTS}" step="1" value="0" aria-label="Particle count" />
          <span class="p-val" data-particles-value>120k</span>
        </label>
        <button type="button" class="row-btn is-overdrive" data-act="overdrive" aria-pressed="false">
          <span>overdrive · 1M particles</span><b>O</b>
        </button>
        <button type="button" class="row-btn is-action" data-act="reset">
          <span>reset the field</span><b>R</b>
        </button>
      </section>

      <section class="hud-sec">
        <h2>spells<span>latched per hand</span></h2>
        <div class="h-grid">${handCard(0, 'hand i')}${handCard(1, 'hand ii')}</div>
        <div class="m-row m-warp">
          <span class="m-key">time</span>
          <span class="trk"><i class="trk-fill" data-warp-fill data-tone="off"></i></span>
          <span class="m-val" data-warp-value>1.00×</span>
          <span class="m-unit"></span>
        </div>
        <ul class="g-list">${GESTURES.map((g) => legendRow(g, true)).join('')}</ul>
      </section>

      <details class="hud-sec hud-adv">
        <summary>engine parameters</summary>
        <div class="seg is-presets" role="group" aria-label="Parameter presets">${presetBtns}</div>
        ${PARAMS.map(paramRow).join('')}
      </details>
    </div>

    <p class="hud-percep" data-tone="wait">
      <i class="dot" aria-hidden="true"></i>
      <b data-percep-head>starting up</b>
      <span data-percep-why>optical flow is already driving the fluid</span>
    </p>
  </aside>

  <div class="hud-hint" aria-hidden="true">
    <b>show your hands</b>
    <div class="chips">${chips}</div>
    <span class="hint-keys">? gesture guide &middot; H hide</span>
  </div>

  <div class="hud-help" role="dialog" aria-modal="false" aria-label="Aether reference" aria-hidden="true">
    <div class="help-sheet">
      <header>
        <b>AETHER</b><span>reference</span>
        <button type="button" class="ico-btn" data-act="help-close" aria-label="Close help" title="Close (Esc)">
          ${svg('<path d="M6 6l12 12M18 6 6 18"/>')}
        </button>
      </header>
      <div class="help-cols">
        <div>
          <h3>gestures</h3>
          <ul class="g-list is-static">${GESTURES.map((g) => legendRow(g, false)).join('')}</ul>
        </div>
        <div>
          <h3>keys</h3>
          <div class="k-list">${keyRows}</div>
          <h3>how it works</h3>
          <p class="help-note">
            Hand and body landmarks steer a ${FLUID_W}×${FLUID_H} fluid solver
            compiled to WebAssembly. With no camera or no models, optical
            flow — or the engine's own ambient drive — keeps the field alive.
          </p>
        </div>
      </div>
    </div>
  </div>`;
}
