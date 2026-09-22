/**
 * WebGL2 bloom chain: a 13-tap downsample ladder and a 9-tap tent fold-back.
 *
 * The twin of `gpu/bloom.ts`, and split out for the same reason: the ladder
 * owns its own targets and programs, so the compositor only tells it the scene
 * size (`resize`), hands it the scene target for level -1 (`record`) and reads
 * `output` for the composite. Everything here dies with the GL context and is
 * rebuilt with the rest of `Resources`.
 */

import { Program, RenderTarget } from './gl';
import type { TextureFormat } from './gl';
import { BLOOM_DOWN_FRAG, BLOOM_UP_FRAG } from './shaders/bloom';
import type { ShaderEnv } from './shaders/common';
import { FULLSCREEN_VERT, buildShader } from './shaders/common';
import type { ModeStyle } from './styles';

/** Deepest bloom mip. Five halvings is a glow radius of ~1/16 of the frame. */
const BLOOM_LEVELS = 5;

/** Tent filter radius for the bloom fold-back, in source texels. */
const BLOOM_TENT = 1.1;

export class BloomChain {
  private readonly levels: RenderTarget[] = [];
  private readonly downPass: Program;
  private readonly upPass: Program;
  private count = 1;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    env: ShaderEnv,
    format: TextureFormat,
  ) {
    for (let i = 0; i < BLOOM_LEVELS; i++) this.levels.push(new RenderTarget(gl, format));
    const vert = buildShader(FULLSCREEN_VERT, env);
    this.downPass = new Program(gl, vert, buildShader(BLOOM_DOWN_FRAG, env), 'bloom-down');
    this.upPass = new Program(gl, vert, buildShader(BLOOM_UP_FRAG, env), 'bloom-up');
  }

  /** The level the composite samples. */
  get output(): WebGLTexture {
    return this.levels[0].texture;
  }

  /**
   * Halves down from the scene size. The chain stops halving once a level would
   * be too small for the 13-tap kernel to mean anything; on a phone in portrait
   * that is three levels, on a 4K canvas the full five.
   */
  resize(sceneW: number, sceneH: number): void {
    let lw = sceneW;
    let lh = sceneH;
    let levels = 0;
    while (levels < BLOOM_LEVELS) {
      const nw = Math.max(1, lw >> 1);
      const nh = Math.max(1, lh >> 1);
      if (nw < 8 || nh < 8) break;
      this.levels[levels].resize(nw, nh);
      lw = nw;
      lh = nh;
      levels++;
    }
    this.count = Math.max(1, levels);
  }

  /** Draws the ladder. The caller has the fullscreen VAO bound. */
  record(scene: RenderTarget, style: ModeStyle): void {
    const gl = this.gl;
    const down = this.downPass;
    down.use();
    down.f1('u_knee', Math.max(0.05, style.threshold * 0.7));
    for (let i = 0; i < this.count; i++) {
      const src = i === 0 ? scene : this.levels[i - 1];
      down.tex('u_src', 0, src.texture);
      down.f2('u_texel', 1 / src.width, 1 / src.height);
      // Only the first level thresholds; below that everything in the chain is
      // already bright by construction.
      down.f1('u_threshold', i === 0 ? style.threshold : 0);
      down.f1('u_fromScene', i === 0 ? 1 : 0);
      this.levels[i].bind();
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const up = this.upPass;
    up.use();
    up.f1('u_amount', 1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = this.count - 1; i > 0; i--) {
      const src = this.levels[i];
      up.tex('u_src', 0, src.texture);
      up.f2('u_texel', BLOOM_TENT / src.width, BLOOM_TENT / src.height);
      this.levels[i - 1].bind();
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.disable(gl.BLEND);
  }

  dispose(): void {
    for (const level of this.levels) level.dispose();
    this.downPass.dispose();
    this.upPass.dispose();
  }
}
