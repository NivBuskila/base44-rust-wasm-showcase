/**
 * The one noise source the whole chain shares: a 128x128 RGBA8 tile, built
 * once on the CPU and sampled with REPEAT.
 *
 * Every shader here wants noise — the dye needs a domain warp and sub-cell
 * grain, the background a slow nebula, the final pass a dither. Doing that
 * procedurally costs ~20 hash rounds per pixel, which is real money at 4K and
 * catastrophic on SwiftShader. Three texture fetches replace all of it.
 *
 * Channel assignment, relied on by the shaders:
 * - `R`, `G`: two independent tileable fBm fields — smooth, for the warp and
 *   the nebula.
 * - `B`, `A`: white noise — for sub-cell grain and for the triangular-PDF
 *   dither, both of which want to be uncorrelated pixel to pixel.
 */

/** Edge length of the tile. Mirrored by `NOISE_SIZE` in the shaders. */
export const NOISE_SIZE = 128;

/**
 * Deterministic 32-bit LCG. The tile must be identical between sessions or a
 * screenshot diff of the renderer becomes noise-dependent and useless.
 */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** Smooth, exactly tileable value noise on a `period`-cell lattice. */
function valueNoise(period: number, rand: () => number): Float32Array {
  const lattice = new Float32Array(period * period);
  for (let i = 0; i < lattice.length; i++) lattice[i] = rand();

  const out = new Float32Array(NOISE_SIZE * NOISE_SIZE);
  const scale = period / NOISE_SIZE;
  for (let y = 0; y < NOISE_SIZE; y++) {
    const fy = y * scale;
    const y0 = Math.floor(fy) % period;
    const y1 = (y0 + 1) % period;
    const ty = fy - Math.floor(fy);
    // Quintic rather than cubic: the second derivative is continuous too, so
    // a warp driven by this field has no visible lattice creases.
    const wy = ty * ty * ty * (ty * (ty * 6 - 15) + 10);
    for (let x = 0; x < NOISE_SIZE; x++) {
      const fx = x * scale;
      const x0 = Math.floor(fx) % period;
      const x1 = (x0 + 1) % period;
      const tx = fx - Math.floor(fx);
      const wx = tx * tx * tx * (tx * (tx * 6 - 15) + 10);
      const a = lattice[y0 * period + x0];
      const b = lattice[y0 * period + x1];
      const c = lattice[y1 * period + x0];
      const d = lattice[y1 * period + x1];
      out[y * NOISE_SIZE + x] = (a + (b - a) * wx) * (1 - wy) + (c + (d - c) * wx) * wy;
    }
  }
  return out;
}

/** Three octaves of tileable value noise, normalised to `[0, 1]`. */
function fbm(rand: () => number): Float32Array {
  const octaves: Array<[number, number]> = [
    [4, 0.55],
    [8, 0.28],
    [16, 0.17],
  ];
  const out = new Float32Array(NOISE_SIZE * NOISE_SIZE);
  for (const [period, weight] of octaves) {
    const layer = valueNoise(period, rand);
    for (let i = 0; i < out.length; i++) out[i] += layer[i] * weight;
  }
  return out;
}

/** Builds the packed RGBA8 tile. */
export function buildNoiseTile(seed = 0x9e3779b9): Uint8Array {
  const rand = lcg(seed);
  const r = fbm(rand);
  const g = fbm(rand);
  const data = new Uint8Array(NOISE_SIZE * NOISE_SIZE * 4);
  for (let i = 0; i < NOISE_SIZE * NOISE_SIZE; i++) {
    const o = i * 4;
    data[o] = Math.max(0, Math.min(255, Math.round(r[i] * 255)));
    data[o + 1] = Math.max(0, Math.min(255, Math.round(g[i] * 255)));
    data[o + 2] = Math.floor(rand() * 256) & 255;
    data[o + 3] = Math.floor(rand() * 256) & 255;
  }
  return data;
}
