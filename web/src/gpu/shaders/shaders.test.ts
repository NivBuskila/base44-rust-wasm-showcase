/**
 * WGSL contract tests, run without a GPU.
 *
 * `wgsl_reflect` parses every shader the WebGPU chain compiles and reports
 * struct layouts, bindings and entry points. That covers the three ways a
 * shader edit breaks the renderer on hardware the editor does not have:
 *
 * - a syntax slip: `createShaderModule` only reports it asynchronously, and
 *   only on a machine that passed the hardware-adapter gate;
 * - a uniform struct that no longer matches the byte count the TypeScript side
 *   packs (`*_UNIFORM_FLOATS`), which the driver may pad silently or reject;
 * - a binding, entry point or workgroup size that drifted from the explicit
 *   layouts in `particle-sim.ts` / `renderer.ts`.
 *
 * Arithmetic inside the kernels is pinned elsewhere: `sim-uniform.test.ts`
 * fixes the `Sim` slot order and per-frame folds, `op-protocol.test.ts` the op
 * kinds the compute shader replays.
 */

import { describe, expect, it } from 'vitest';
import { WgslReflect } from 'wgsl_reflect';
import { PARTICLE_OP } from '../../constants';
import { BLOOM_DOWN_WGSL, BLOOM_UNIFORM_FLOATS, BLOOM_UP_WGSL } from './bloom';
import { BLIT_WGSL, COMPOSITE_UNIFORM_FLOATS, COMPOSITE_WGSL } from './composite';
import { OVERLAY_LINES_WGSL, OVERLAY_POINTS_WGSL, OVERLAY_UNIFORM_FLOATS } from './overlay';
import {
  DRAW_UNIFORM_FLOATS,
  PARTICLE_BYTES,
  PARTICLE_COMPUTE_WGSL,
  PARTICLE_DRAW_WGSL,
  SIM_UNIFORM_FLOATS,
  STEP_WORKGROUP,
} from './particles';
import { SCENE_UNIFORM_FLOATS, SCENE_WGSL } from './scene';

/** Every module the renderer and the sim hand to `createShaderModule`. */
const MODULES: Record<string, string> = {
  scene: SCENE_WGSL,
  'bloom-down': BLOOM_DOWN_WGSL,
  'bloom-up': BLOOM_UP_WGSL,
  composite: COMPOSITE_WGSL,
  blit: BLIT_WGSL,
  'particles-draw': PARTICLE_DRAW_WGSL,
  'overlay-lines': OVERLAY_LINES_WGSL,
  'overlay-points': OVERLAY_POINTS_WGSL,
  'particles-step': PARTICLE_COMPUTE_WGSL,
};

const reflect = (code: string): WgslReflect => new WgslReflect(code);

/** The single `@group(0) @binding(0)` uniform block a pass declares. */
const uniformBytes = (code: string): number => {
  const r = reflect(code);
  expect(r.uniforms).toHaveLength(1);
  expect(r.uniforms[0].group).toBe(0);
  expect(r.uniforms[0].binding).toBe(0);
  return r.uniforms[0].size;
};

const struct = (code: string, name: string) => {
  const s = reflect(code).structs.find((s) => s.name === name);
  expect(s, `struct ${name}`).toBeDefined();
  return s!;
};

describe('every WebGPU shader', () => {
  for (const [label, code] of Object.entries(MODULES)) {
    it(`${label} parses`, () => {
      expect(() => reflect(code)).not.toThrow();
    });
  }

  it('fullscreen and draw passes expose vs_main + fs_main', () => {
    for (const [label, code] of Object.entries(MODULES)) {
      if (label === 'particles-step') continue;
      const r = reflect(code);
      expect(r.entry.vertex.map((e) => e.name), label).toEqual(['vs_main']);
      expect(r.entry.fragment.map((e) => e.name), label).toEqual(['fs_main']);
      expect(r.entry.compute, label).toHaveLength(0);
    }
  });
});

describe('uniform blocks match the floats the browser packs', () => {
  const cases: [string, string, number][] = [
    ['scene', SCENE_WGSL, SCENE_UNIFORM_FLOATS],
    ['composite', COMPOSITE_WGSL, COMPOSITE_UNIFORM_FLOATS],
    ['bloom-down', BLOOM_DOWN_WGSL, BLOOM_UNIFORM_FLOATS],
    ['bloom-up', BLOOM_UP_WGSL, BLOOM_UNIFORM_FLOATS],
    ['overlay-lines', OVERLAY_LINES_WGSL, OVERLAY_UNIFORM_FLOATS],
    ['overlay-points', OVERLAY_POINTS_WGSL, OVERLAY_UNIFORM_FLOATS],
    ['particles-draw', PARTICLE_DRAW_WGSL, DRAW_UNIFORM_FLOATS],
    ['particles-step', PARTICLE_COMPUTE_WGSL, SIM_UNIFORM_FLOATS],
  ];
  for (const [label, code, floats] of cases) {
    it(`${label}: ${floats} floats`, () => {
      expect(uniformBytes(code)).toBe(floats * 4);
    });
  }

  it('blit has no uniform at all', () => {
    expect(reflect(BLIT_WGSL).uniforms).toHaveLength(0);
  });
});

describe('particle compute kernel', () => {
  const r = reflect(PARTICLE_COMPUTE_WGSL);

  it('exposes step_main and seed_main at the dispatch workgroup size', () => {
    const entries = r.entry.compute.map((e) => [
      e.name,
      e.attributes?.find((a) => a.name === 'workgroup_size')?.value,
    ]);
    expect(entries).toEqual([
      ['step_main', String(STEP_WORKGROUP)],
      ['seed_main', String(STEP_WORKGROUP)],
    ]);
  });

  it('binds the seven slots of the explicit bind group layout, in order', () => {
    // Mirrors `ParticleSim`'s `createBindGroupLayout` — one uniform, then six
    // storage buffers whose access must match the layout's `type`.
    const storage = r.storage
      .map((s) => ({ binding: s.binding, name: s.name, access: s.access }))
      .sort((a, b) => a.binding - b.binding);
    expect(r.uniforms.map((u) => u.binding)).toEqual([0]);
    expect(storage).toEqual([
      { binding: 1, name: 'P', access: 'read_write' },
      { binding: 2, name: 'vel_u', access: 'read' },
      { binding: 3, name: 'vel_v', access: 'read' },
      { binding: 4, name: 'obstacle', access: 'read' },
      { binding: 5, name: 'ops', access: 'read' },
      { binding: 6, name: 'C', access: 'read_write' },
    ]);
    for (const s of r.storage) expect(s.group, s.name).toBe(0);
  });

  it('lays Sim out in the slot order sim-uniform.ts writes', () => {
    const sim = struct(PARTICLE_COMPUTE_WGSL, 'Sim');
    expect(sim.members.map((m) => [m.name, m.offset])).toEqual([
      ['grid', 0],
      ['rates', 16],
      ['misc', 32],
      ['life_ob', 48],
      ['counts', 64],
      ['dims', 80],
    ]);
    expect(sim.size).toBe(SIM_UNIFORM_FLOATS * 4);
  });

  it('declares one OP_* constant per protocol kind', () => {
    for (const [name, value] of Object.entries(PARTICLE_OP)) {
      expect(PARTICLE_COMPUTE_WGSL).toContain(`const OP_${name}: f32 = ${value.toFixed(1)};`);
    }
  });
});

describe('particle pool struct', () => {
  it('is PARTICLE_BYTES wide in both the kernel and the draw shader', () => {
    const compute = struct(PARTICLE_COMPUTE_WGSL, 'Particle');
    const draw = struct(PARTICLE_DRAW_WGSL, 'Particle');
    expect(compute.size).toBe(PARTICLE_BYTES);
    expect(draw.size).toBe(PARTICLE_BYTES);
  });

  it('has the same member layout on the write and read sides', () => {
    // The draw pass reads the buffer the kernel writes; a reordered field
    // would draw velocities as positions and never raise an error.
    const shape = (code: string) =>
      struct(code, 'Particle').members.map((m) => [m.name, m.offset, m.type.name]);
    expect(shape(PARTICLE_DRAW_WGSL)).toEqual(shape(PARTICLE_COMPUTE_WGSL));
  });

  it('is read-only from the draw pass', () => {
    const r = reflect(PARTICLE_DRAW_WGSL);
    expect(r.storage.map((s) => [s.name, s.binding, s.access])).toEqual([['P', 1, 'read']]);
  });
});

describe('sampled passes', () => {
  it('scene binds dye, video, noise and both samplers', () => {
    const r = reflect(SCENE_WGSL);
    expect(r.textures.map((t) => [t.name, t.binding])).toEqual([
      ['u_dye', 1],
      ['u_video', 2],
      ['u_noise', 3],
    ]);
    expect(r.samplers.map((s) => [s.name, s.binding])).toEqual([
      ['s_clamp', 4],
      ['s_repeat', 5],
    ]);
  });

  it('composite binds scene, bloom and noise', () => {
    const r = reflect(COMPOSITE_WGSL);
    expect(r.textures.map((t) => [t.name, t.binding])).toEqual([
      ['u_scene', 1],
      ['u_bloom', 2],
      ['u_noise', 3],
    ]);
    expect(r.samplers.map((s) => [s.name, s.binding])).toEqual([['s_clamp', 4]]);
  });

  it('bloom levels and blit sample one source each', () => {
    for (const code of [BLOOM_DOWN_WGSL, BLOOM_UP_WGSL]) {
      const r = reflect(code);
      expect(r.textures.map((t) => [t.name, t.binding])).toEqual([['u_src', 1]]);
      expect(r.samplers.map((s) => [s.name, s.binding])).toEqual([['s_clamp', 2]]);
    }
    const blit = reflect(BLIT_WGSL);
    expect(blit.textures.map((t) => [t.name, t.binding])).toEqual([['u_src', 0]]);
    expect(blit.samplers.map((s) => [s.name, s.binding])).toEqual([['s_nearest', 1]]);
  });
});
