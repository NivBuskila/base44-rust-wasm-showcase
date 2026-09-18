/**
 * Named starting points for the engine parameters.
 *
 * The nine sliders interact — vorticity fights the dissipations, spawn rate
 * only reads as density against particle lifetime — so finding a look by
 * dragging one at a time means passing through a lot of combinations that just
 * look broken. These are the combinations worth seeing: each one moves the
 * whole set at once, and any slider can still be dragged afterwards, since a
 * preset is a write, not a mode.
 *
 * Only keys that differ from `Params::default` / `ParticleConfig::default` are
 * listed; the HUD fills the rest from each slider's own default, so applying a
 * preset lands on a complete state rather than a diff on whatever came before.
 */

/** A preset is a partial map from `ParamSpec.key` to value. */
export interface ParamPreset {
  readonly id: string;
  readonly label: string;
  /** One line, shown as the button's tooltip. */
  readonly hint: string;
  readonly values: Readonly<Record<string, number>>;
}

export const PRESETS: readonly ParamPreset[] = [
  {
    id: 'default',
    label: 'default',
    hint: 'The engine defaults — the balance everything else is a departure from.',
    values: {},
  },
  {
    id: 'silk',
    label: 'silk',
    hint: 'Slow, continuous and smooth: low curl, high density, gentle time.',
    values: {
      vorticity: 9,
      dye_dissipation: 0.35,
      velocity_dissipation: 0.1,
      hand_force: 1.6,
      flow_force: 0.45,
      curl_influence: 1.2,
      time_scale: 0.85,
      spawn_rate: 60_000,
    },
  },
  {
    id: 'storm',
    label: 'storm',
    hint: 'Hard vorticity and fast decay: sharp filaments that tear apart quickly.',
    values: {
      vorticity: 42,
      dye_dissipation: 1.4,
      velocity_dissipation: 0.5,
      hand_force: 4.5,
      flow_force: 1.2,
      curl_influence: 8,
      time_scale: 1.35,
      spawn_rate: 90_000,
    },
  },
  {
    id: 'ink',
    label: 'ink',
    hint: 'Almost no decay: dye keeps every stroke, like ink dropped into water.',
    values: {
      vorticity: 6,
      dye_dissipation: 0.06,
      velocity_dissipation: 0.04,
      hand_force: 1.2,
      flow_force: 0.3,
      curl_influence: 0.8,
      time_scale: 0.6,
      spawn_rate: 40_000,
    },
  },
  {
    id: 'ember',
    label: 'ember',
    hint: 'Sparse and swirling: few particles, heavy curl, long-lived velocity.',
    values: {
      vorticity: 22,
      dye_dissipation: 0.9,
      velocity_dissipation: 0.08,
      hand_force: 2.4,
      flow_force: 0.6,
      curl_influence: 14,
      time_scale: 1,
      spawn_rate: 18_000,
    },
  },
];
