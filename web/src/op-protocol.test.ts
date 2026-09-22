/**
 * The op log protocol has to say the same thing in three languages: Rust
 * records it, `constants.ts` mirrors it, WGSL replays it. Rust is the source —
 * `assertOpLayout` rejects a drift at boot — and these tests pin the two links
 * that a boot assertion cannot see: that the assertion really compares every
 * kind, and that the shader constants are generated from the mirror.
 */

import { describe, expect, it } from 'vitest';
import { MAX_PARTICLE_OPS, PARTICLE_OP, PARTICLE_OP_STRIDE, assertOpLayout } from './constants';
import { OP_KINDS_WGSL, PARTICLE_COMPUTE_WGSL } from './gpu/shaders/particles';

/** What a matching engine publishes from `particle_op_layout()`. */
const engineLayout = (): number[] => [
  PARTICLE_OP_STRIDE,
  MAX_PARTICLE_OPS,
  ...Object.values(PARTICLE_OP),
];

describe('assertOpLayout', () => {
  it('accepts a layout that matches the mirror', () => {
    expect(() => assertOpLayout(engineLayout())).not.toThrow();
  });

  it('rejects a changed stride or op ceiling', () => {
    const stride = engineLayout();
    stride[0] += 1;
    expect(() => assertOpLayout(stride)).toThrow(/PARTICLE_OP_STRIDE/);

    const max = engineLayout();
    max[1] += 1;
    expect(() => assertOpLayout(max)).toThrow(/MAX_PARTICLE_OPS/);
  });

  it('rejects a renumbered kind, naming it', () => {
    const renumbered = engineLayout();
    renumbered[2 + Object.keys(PARTICLE_OP).indexOf('GRIP')] = 9;
    expect(() => assertOpLayout(renumbered)).toThrow(/PARTICLE_OP\.GRIP/);
  });

  it('rejects a kind added or dropped on the Rust side', () => {
    expect(() => assertOpLayout([...engineLayout(), 6])).toThrow(/op layout entries/);
    expect(() => assertOpLayout(engineLayout().slice(0, -1))).toThrow(/op layout entries/);
  });
});

describe('WGSL op constants', () => {
  it('are generated from the mirror, one per kind', () => {
    for (const [name, value] of Object.entries(PARTICLE_OP)) {
      expect(OP_KINDS_WGSL).toContain(`const OP_${name}: f32 = ${value}.0;`);
    }
    expect(OP_KINDS_WGSL.split('\n')).toHaveLength(Object.keys(PARTICLE_OP).length);
  });

  it('are spliced into the compute shader, with no literal copy left', () => {
    expect(PARTICLE_COMPUTE_WGSL).toContain(OP_KINDS_WGSL);
    // Each kind is declared exactly once and read at least once by the replay.
    for (const name of Object.keys(PARTICLE_OP)) {
      const declarations = PARTICLE_COMPUTE_WGSL.match(
        new RegExp(`const OP_${name}: f32`, 'g'),
      );
      expect(declarations).toHaveLength(1);
      expect(PARTICLE_COMPUTE_WGSL).toContain(`kind == OP_${name}`);
    }
  });
});
