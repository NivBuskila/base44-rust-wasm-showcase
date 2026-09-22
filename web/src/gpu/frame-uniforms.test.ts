import { describe, expect, it } from "vitest";

import { STYLES } from "../render/styles";
import { packCompositeUniform, packSceneUniform } from "./frame-uniforms";
import { COMPOSITE_UNIFORM_FLOATS } from "./shaders/composite";
import { SCENE_UNIFORM_FLOATS } from "./shaders/scene";

/** Slot names of the WGSL `Scene` struct, in order. */
const SCENE = {
  gridW: 0,
  gridH: 1,
  invGridW: 2,
  invGridH: 3,
  camScaleX: 4,
  camScaleY: 5,
  invTexW: 6,
  invTexH: 7,
  dye: 16,
  bg: 17,
  hasVideo: 18,
  intensity: 19,
  time: 20,
} as const;

/** Slot names of the WGSL `Composite` struct, in order. */
const COMP = {
  bloom: 0,
  exposure: 1,
  aberration: 2,
  vignette: 3,
  grade: 4,
  frame: 5,
  rushHue: 6,
  rushPower: 7,
  rushX: 8,
  rushY: 9,
} as const;

const style = STYLES.blend;

const scene = (over: Record<string, unknown> = {}): Float32Array => {
  const f = new Float32Array(SCENE_UNIFORM_FLOATS);
  packSceneUniform(f, {
    style,
    fluidW: 200,
    fluidH: 120,
    camScaleX: 1,
    camScaleY: 0.5,
    videoTexW: 640,
    videoTexH: 480,
    hasVideo: true,
    intensity: 0.25,
    time: 3,
    ...over,
  } as Parameters<typeof packSceneUniform>[1]);
  return f;
};

const composite = (
  rush: Float32Array | null,
  intensity = 0.5,
): Float32Array => {
  const c = new Float32Array(COMPOSITE_UNIFORM_FLOATS);
  packCompositeUniform(c, { style, intensity, frameIndex: 11, rush });
  return c;
};

describe("packSceneUniform", () => {
  it("writes the slots the WGSL Scene struct reads", () => {
    const f = scene();
    expect(f[SCENE.gridW]).toBe(200);
    expect(f[SCENE.gridH]).toBe(120);
    expect(f[SCENE.invGridW]).toBeCloseTo(1 / 200, 8);
    expect(f[SCENE.invGridH]).toBeCloseTo(1 / 120, 8);
    expect(f[SCENE.camScaleX]).toBe(1);
    expect(f[SCENE.camScaleY]).toBe(0.5);
    expect(f[SCENE.invTexW]).toBeCloseTo(1 / 640, 8);
    expect(f[SCENE.invTexH]).toBeCloseTo(1 / 480, 8);
    expect(f[SCENE.dye]).toBeCloseTo(style.dye, 6);
    expect(f[SCENE.bg]).toBeCloseTo(style.bg, 6);
    expect(f[SCENE.hasVideo]).toBe(1);
    expect(f[SCENE.intensity]).toBeCloseTo(0.25, 6);
    expect(f[SCENE.time]).toBe(3);
  });

  it("keeps the camera reciprocals finite before a texture exists", () => {
    const f = scene({ videoTexW: 0, videoTexH: 0, hasVideo: false });
    expect(f[SCENE.invTexW]).toBe(1);
    expect(f[SCENE.invTexH]).toBe(1);
    expect(f[SCENE.hasVideo]).toBe(0);
  });

  it("rejects a scratch block smaller than the uniform", () => {
    expect(() =>
      packSceneUniform(new Float32Array(4), {
        style,
        fluidW: 1,
        fluidH: 1,
        camScaleX: 1,
        camScaleY: 1,
        videoTexW: 1,
        videoTexH: 1,
        hasVideo: false,
        intensity: 0,
        time: 0,
      }),
    ).toThrow(/too small/);
    expect(() =>
      packCompositeUniform(new Float32Array(2), {
        style,
        intensity: 0,
        frameIndex: 0,
        rush: null,
      }),
    ).toThrow(/too small/);
  });
});

describe("packCompositeUniform", () => {
  it("writes the slots the WGSL Composite struct reads", () => {
    const c = composite(null);
    expect(c[COMP.aberration]).toBeCloseTo(style.aberration, 6);
    expect(c[COMP.vignette]).toBeCloseTo(style.vignette, 6);
    expect(c[COMP.grade]).toBeCloseTo(style.grade, 6);
    expect(c[COMP.frame]).toBe(11);
  });

  it("folds bloom and exposure against intensity", () => {
    const c = composite(null, 0.5);
    expect(c[COMP.bloom]).toBeCloseTo(style.bloom * (0.72 + 0.75 * 0.5), 6);
    const gain = style.grade > 0 ? 0.96 + 0.22 * 0.5 : 1;
    expect(c[COMP.exposure]).toBeCloseTo(style.exposure * gain, 6);
  });

  it("centres the rush focus while no rush is staged", () => {
    const c = composite(new Float32Array([0.2, 0.3, 0.7, 0]));
    expect(c[COMP.rushPower]).toBe(0);
    expect(c[COMP.rushHue]).toBe(0);
    expect(c[COMP.rushX]).toBe(0.5);
    expect(c[COMP.rushY]).toBe(0.5);
  });

  it("passes the rush focus through once its power is positive", () => {
    const c = composite(new Float32Array([0.2, 0.3, 0.7, 1.5]));
    expect(c[COMP.rushPower]).toBeCloseTo(1.5, 6);
    expect(c[COMP.rushHue]).toBeCloseTo(0.7, 6);
    expect(c[COMP.rushX]).toBeCloseTo(0.2, 6);
    expect(c[COMP.rushY]).toBeCloseTo(0.3, 6);
  });

  it("ignores a non-finite rush power instead of poisoning the block", () => {
    const c = composite(new Float32Array([0.2, 0.3, 0.7, Number.NaN]));
    expect(c[COMP.rushPower]).toBe(0);
    expect(c[COMP.rushX]).toBe(0.5);
  });
});
