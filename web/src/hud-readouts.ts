/**
 * Pure readings the panel prints: which hand slots hold a hand, what each one
 * is casting, and the small-number format for the divergence column. No DOM and
 * no engine calls, so they can be reasoned about (and tested) on their own.
 */

import type { HudStats } from './types';

/** A latched spell name, or `idle` for anything the engine did not send. */
export function spellAt(spells: HudStats['spells'] | undefined, slot: number): string {
  const s = spells?.[slot];
  return typeof s === 'string' && s.length > 0 ? s : 'idle';
}

/**
 * Which hand slots hold a hand, from the only two things the engine reports:
 * a *count* and the per-slot latched spell.
 *
 * The count alone is not enough, because the slots are independent — a single
 * right hand lands in slot 1 with slot 0 empty (`packHands` in
 * `perception/packing.ts` keys the slot off handedness). Treating
 * `count > slot` as presence therefore reads a lone right hand as "hand i idle,
 * hand ii absent", which is both cards wrong at exactly the moment one is
 * casting. A non-idle spell pins a slot down: `spells.rs` forces `Spell::Idle`
 * for any slot whose hand is missing, so a named spell can only come from a
 * hand that is there.
 *
 * One case stays genuinely ambiguous: one hand present, nothing latched.
 * Nothing in `HudStats` says which slot it is, so slot 0 is assumed.
 */
export function presence(
  hands: number,
  spells: HudStats['spells'] | undefined,
): [boolean, boolean] {
  const cast0 = spellAt(spells, 0) !== 'idle';
  const cast1 = spellAt(spells, 1) !== 'idle';
  if (cast0 && cast1) return [true, true];
  if (hands <= 0) return [false, false];
  if (hands >= 2) return [true, true];
  if (cast1 && !cast0) return [false, true];
  return [true, false];
}

/** `0.00042` -> `4.2e−4`; keeps the divergence column from reading `0.0000`. */
export function fmtSmall(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 0.001) return v.toExponential(1).replace('e-', 'e−');
  return v.toFixed(4);
}
