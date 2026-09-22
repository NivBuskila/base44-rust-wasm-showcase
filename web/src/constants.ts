/**
 * Mirror of `aether_core::config`. Keep in sync with
 * `crates/aether-core/src/config.rs`.
 *
 * `assertLayout` below is called once at boot against the engine's own
 * `layout()`, so a drift between the two files fails immediately and loudly
 * rather than showing up as a skewed texture.
 */

/** Fluid / dye / obstacle grid width, in cells. */
export const FLUID_W = 256;
/** Fluid / dye / obstacle grid height, in cells. */
export const FLUID_H = 144;
/** Optical-flow working width; the camera luma plane is downscaled to this. */
export const FLOW_W = 128;
/** Optical-flow working height. */
export const FLOW_H = 72;
/** Hard particle ceiling in the engine. */
export const MAX_PARTICLES = 1_000_000;
/** Floats per particle in the render buffer: x, y, heat, life. */
export const PARTICLE_STRIDE = 4;

/**
 * Particle op log, `aether_core::particles::offload`. Floats per record and
 * records per frame; `assertOpLayout` checks them at boot when the GPU pool is
 * in use.
 */
export const PARTICLE_OP_STRIDE = 12;
export const MAX_PARTICLE_OPS = 64;
/**
 * Op kinds, slot 0 of each record. Declaration order is the protocol order:
 * `aether_core::particles::OP_KINDS` publishes the same list through
 * `particle_op_layout()`, `assertOpLayout` compares the two entry by entry, and
 * the WGSL replay constants are generated from this object — so a kind exists in
 * exactly one place per language and a renumber cannot pass unnoticed.
 */
export const PARTICLE_OP = {
  BURST: 1,
  RING: 2,
  IMPULSE: 3,
  DAMP: 4,
  GRIP: 5,
} as const;

/** MediaPipe hand landmark count. */
export const HAND_LANDMARKS = 21;
/** MediaPipe pose landmark count. */
export const POSE_LANDMARKS = 33;
/** Hands the engine tracks at once. */
export const HANDS = 2;

/**
 * Floats per hand in the packed buffer handed to `push_hands`:
 * `[present, handedness, gestureId, gestureScore, ...21 * (x, y, z)]`.
 */
export const HAND_STRIDE = 4 + HAND_LANDMARKS * 3;
/** Total length of the packed hand buffer. */
export const HAND_BUFFER = HAND_STRIDE * HANDS;
/**
 * Floats in the packed pose buffer handed to `push_pose`:
 * `[present, ...33 * (x, y, z, visibility)]`.
 */
export const POSE_STRIDE = 1 + POSE_LANDMARKS * 4;

/** Largest segmentation mask dimension the engine's staging buffer accepts. */
export const MASK_IN_MAX_DIM = 320;

/** Hand landmark indices, MediaPipe ordering. */
export const LM = {
  WRIST: 0,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_TIP: 12,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_TIP: 20,
} as const;

/**
 * Indices into the packed array returned by `AetherEngine.stats()`.
 * Mirrors the doc comment on `aether_wasm::AetherEngine::stats`.
 */
export const STAT = {
  FLUID_ENERGY: 0,
  FLUID_MAX_SPEED: 1,
  FLUID_DIVERGENCE: 2,
  MOTION_ENERGY: 3,
  FLOW_DX: 4,
  FLOW_DY: 5,
  PARTICLES_ALIVE: 6,
  MASK_COVERAGE: 7,
  MASK_PRESENT: 8,
  HANDS_PRESENT: 9,
  POSE_PRESENT: 10,
  TIME_SCALE: 11,
  BURSTS: 12,
  NAN_REPAIRS: 13,
  FRAME: 14,
  AMBIENT: 15,
} as const;

/** Length of the packed stats array. */
export const STATS_LEN = 16;

/**
 * MediaPipe canned gesture labels, in the classifier index order that
 * `aether_core::gesture::CannedGesture::from_id` expects. Index 0 is "None",
 * so a label the recogniser does not emit maps to 0 and means "no gesture".
 */
export const CANNED_GESTURES = [
  'None',
  'Closed_Fist',
  'Open_Palm',
  'Pointing_Up',
  'Thumb_Down',
  'Thumb_Up',
  'Victory',
  'ILoveYou',
] as const;

/** Maps a MediaPipe gesture category name to the id the engine expects. */
export function gestureId(categoryName: string): number {
  const i = (CANNED_GESTURES as readonly string[]).indexOf(categoryName);
  return i < 0 ? 0 : i;
}

/**
 * Fails loudly when the op protocol here drifts from the Rust one.
 *
 * `layout` is `AetherEngine.particle_op_layout()`:
 * `[opStride, maxOps, ...one value per kind in PARTICLE_OP declaration order]`.
 */
export function assertOpLayout(layout: Uint32Array | number[]): void {
  const kinds = Object.entries(PARTICLE_OP);
  const expected: [string, number][] = [
    ['PARTICLE_OP_STRIDE', PARTICLE_OP_STRIDE],
    ['MAX_PARTICLE_OPS', MAX_PARTICLE_OPS],
    ...kinds.map(([name, value]): [string, number] => [`PARTICLE_OP.${name}`, value]),
  ];
  if (layout.length !== expected.length) {
    throw new Error(
      `constants.ts is out of sync with aether-core: the engine publishes ` +
        `${layout.length} op layout entries, TS expects ${expected.length}. ` +
        `Update web/src/constants.ts to match particles/offload.rs.`,
    );
  }
  for (let i = 0; i < expected.length; i++) {
    const [name, value] = expected[i];
    if (layout[i] !== value) {
      throw new Error(
        `constants.ts is out of sync with aether-core: ${name} is ${value} in TS ` +
          `but ${layout[i]} in the engine. ` +
          `Update web/src/constants.ts to match particles/offload.rs.`,
      );
    }
  }
}

export function assertLayout(layout: Uint32Array | number[]): void {
  const expected = [FLUID_W, FLUID_H, FLOW_W, FLOW_H, MAX_PARTICLES, PARTICLE_STRIDE];
  const names = ['FLUID_W', 'FLUID_H', 'FLOW_W', 'FLOW_H', 'MAX_PARTICLES', 'PARTICLE_STRIDE'];
  for (let i = 0; i < expected.length; i++) {
    if (layout[i] !== expected[i]) {
      throw new Error(
        `constants.ts is out of sync with aether-core: ${names[i]} is ` +
          `${expected[i]} in TS but ${layout[i]} in the engine. ` +
          `Update web/src/constants.ts to match crates/aether-core/src/config.rs.`,
      );
    }
  }
}
