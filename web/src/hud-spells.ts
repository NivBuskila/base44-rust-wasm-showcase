/**
 * The spell island: the two hand cards and the gesture legend.
 *
 * Both read the same two slots, so they belong together: a card says what that
 * hand is casting (or that there is no hand there), and a legend row lights when
 * either hand is casting it — `warp` being the exception, since time warp is a
 * field state rather than a per-hand spell. Keeping the slot inference
 * (`hud-readouts.ts`) next to the elements it decides means `hud.ts` hands over
 * three values per paint and keeps no element references of its own.
 *
 * `hud-params.ts` and `hud-pool.ts` are the same shape for the controls; this is
 * the readout. Every write diffs first, as everywhere in the panel.
 */

import { presence, spellAt } from "./hud-readouts";
import { GESTURES } from "./hud-spec";
import type { HudStats } from "./types";

export class SpellReadout {
  private readonly cards: HTMLElement[] = [];
  private readonly legend = new Map<string, HTMLElement>();

  constructor(root: HTMLElement) {
    const q = <T extends HTMLElement>(sel: string): T => {
      const el = root.querySelector(sel);
      if (!el) throw new Error(`hud: missing element ${sel}`);
      return el as T;
    };
    this.cards.push(q('[data-hand="0"] b'), q('[data-hand="1"] b'));
    for (const g of GESTURES)
      this.legend.set(g.spell, q(`[data-spell="${g.spell}"]`));
  }

  /** `hands` is the engine's hand count; `warping` lights the `warp` row. */
  paint(
    hands: number,
    spells: HudStats["spells"] | undefined,
    warping: boolean,
  ): void {
    const held = presence(hands, spells);
    for (let i = 0; i < this.cards.length; i++) {
      const el = this.cards[i];
      if (!el) continue;
      const spell = spellAt(spells, i);
      const present = held[i] === true;
      setText(el, !present ? "no hand" : spell);
      const card = el.parentElement;
      if (card) {
        setData(
          card,
          "state",
          !present ? "off" : spell === "idle" ? "idle" : "cast",
        );
      }
    }

    for (const [spell, el] of this.legend) {
      const on =
        spell === "warp"
          ? warping
          : spells?.[0] === spell || spells?.[1] === spell;
      setData(el, "on", on ? "1" : "");
    }
  }
}

const setText = (el: HTMLElement, text: string): void => {
  if (el.textContent !== text) el.textContent = text;
};

const setData = (el: HTMLElement, key: string, value: string): void => {
  if (el.dataset[key] !== value) el.dataset[key] = value;
};
