/**
 * Per-view-mode look, shared by both backends.
 *
 * Split out of `renderer.ts` so the WebGL2 compositor and the WebGPU one grade
 * a frame identically: the two chains differ in API, never in look.
 */

import type { ViewMode } from '../types';

/** Per-view-mode look. Everything that differs between modes lives here. */
export interface ModeStyle {
  /** Background gradient and nebula. */
  bg: number;
  /** Dye layer gain. */
  dye: number;
  /** Camera body tint; also decides whether the camera layer runs at all. */
  camTint: readonly [number, number, number];
  /** Camera silhouette rim colour. */
  camEdge: readonly [number, number, number];
  /** Camera toe strength; 1 buries the room, 0 leaves the feed linear. */
  camToe: number;
  /** 1 shows the feed as a plain colour camera, bypassing tint, toe and rim. */
  camRaw: number;
  /** 1 applies the filmic tone map and grade in the composite; 0 passes through. */
  grade: number;
  particles: number;
  /** Hand skeleton overlay alpha. */
  overlay: number;
  bloom: number;
  /** Radiance above which a highlight blooms. */
  threshold: number;
  exposure: number;
  aberration: number;
  vignette: number;
}

export const STYLES: Record<ViewMode, ModeStyle> = {
  // The full composite: the camera is crushed to a cold suggestion of a room
  // and the fluid owns the frame.
  aether: {
    bg: 1,
    dye: 1,
    // Linear radiance, and the sRGB transfer at the end lifts it hard: these
    // put a brightly lit subject at roughly 12% display grey and the
    // silhouette rim at 35%, which is as much room as the fluid can share.
    camTint: [0.0065, 0.0105, 0.0215],
    camEdge: [0.018, 0.048, 0.088],
    camToe: 1,
    camRaw: 0,
    grade: 1,
    particles: 1,
    overlay: 0.35,
    bloom: 0.95,
    threshold: 0.62,
    exposure: 1.0,
    aberration: 0.55,
    vignette: 0.52,
  },
  // A regular camera: the feed in its own colour with no tint, toe, grade,
  // fringe or vignette. The fluid and particles are drawn at full strength on
  // top of it, since an additive layer over a bright feed reads far dimmer
  // than over black. The bloom threshold sits above the feed's own range so
  // only the fluid's dense cores glow, never the room.
  camera: {
    bg: 0,
    dye: 1.1,
    camTint: [1, 1, 1],
    camEdge: [0, 0, 0],
    camToe: 0,
    camRaw: 1,
    grade: 0,
    particles: 1.2,
    overlay: 1.0,
    bloom: 0.6,
    threshold: 1.05,
    exposure: 1.0,
    aberration: 0,
    vignette: 0,
  },
  // Camera and aether together: the feed in its own colour, held down to a
  // little under half brightness so the fluid stays the subject, with the
  // full filmic grade, bloom and fringe of the aether view laid over it.
  blend: {
    bg: 0.3,
    dye: 1.05,
    camTint: [0.42, 0.42, 0.42],
    camEdge: [0, 0, 0],
    camToe: 0,
    camRaw: 1,
    grade: 1,
    particles: 1.1,
    overlay: 0.6,
    bloom: 0.9,
    threshold: 0.78,
    exposure: 1.0,
    aberration: 0.35,
    vignette: 0.42,
  },
  // Particles on black, with the bloom pushed: this is the mode where the
  // point cloud has to carry the whole image on its own.
  particles: {
    bg: 0,
    dye: 0,
    camTint: [0, 0, 0],
    camEdge: [0, 0, 0],
    camToe: 1,
    camRaw: 0,
    grade: 1,
    particles: 1.3,
    overlay: 0,
    bloom: 1.25,
    threshold: 0.42,
    exposure: 1.05,
    aberration: 0.6,
    vignette: 0.34,
  },
  // Unused: the debug view bypasses the whole chain. Present so the table is
  // total over ViewMode and a new mode cannot silently fall through.
  debug: {
    bg: 0,
    dye: 0,
    camTint: [0, 0, 0],
    camEdge: [0, 0, 0],
    camToe: 1,
    camRaw: 0,
    grade: 1,
    particles: 0,
    overlay: 0,
    bloom: 0,
    threshold: 1,
    exposure: 1,
    aberration: 0,
    vignette: 0,
  },
};

