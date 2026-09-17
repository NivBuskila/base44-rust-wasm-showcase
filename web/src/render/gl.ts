/**
 * WebGL2 plumbing: program linking with cached uniform lookups, render targets
 * that can be resized in place, and the capability probe that decides whether
 * the HDR chain runs on float or 8-bit targets.
 *
 * Split out of `renderer.ts` so that file reads as a list of passes rather than
 * as resource bookkeeping. Nothing here knows what Aether looks like.
 */

export class RendererError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RendererError';
  }
}

/** A texture's storage triple, as `texImage2D` wants it. */
export interface TextureFormat {
  internal: number;
  format: number;
  type: number;
}

/**
 * A linked program plus its uniform locations.
 *
 * Locations are looked up lazily and cached: `getUniformLocation` is a string
 * lookup into the driver and doing it per uniform per frame shows up in a
 * profile once there are eight programs in the chain.
 */
export class Program {
  readonly handle: WebGLProgram;
  private readonly locs = new Map<string, WebGLUniformLocation | null>();

  constructor(
    private readonly gl: WebGL2RenderingContext,
    vert: string,
    frag: string,
    readonly label: string,
  ) {
    this.handle = link(gl, vert, frag, label);
  }

  use(): void {
    this.gl.useProgram(this.handle);
  }

  /**
   * `null` for a uniform the linker dropped, which every `gl.uniform*` call
   * accepts as a no-op — so callers never branch on it.
   */
  private loc(name: string): WebGLUniformLocation | null {
    let cached = this.locs.get(name);
    if (cached === undefined) {
      cached = this.gl.getUniformLocation(this.handle, name);
      this.locs.set(name, cached);
    }
    return cached;
  }

  f1(name: string, x: number): void {
    this.gl.uniform1f(this.loc(name), x);
  }

  f2(name: string, x: number, y: number): void {
    this.gl.uniform2f(this.loc(name), x, y);
  }

  f3(name: string, x: number, y: number, z: number): void {
    this.gl.uniform3f(this.loc(name), x, y, z);
  }

  /** Binds `texture` to `unit` and points the sampler at it. */
  tex(name: string, unit: number, texture: WebGLTexture | null): void {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.uniform1i(this.loc(name), unit);
  }

  dispose(): void {
    this.gl.deleteProgram(this.handle);
  }
}

function link(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string, label: string): WebGLProgram {
  const compile = (type: number, src: string): WebGLShader => {
    const shader = gl.createShader(type);
    if (!shader) throw new RendererError(`could not create a shader for ${label}`);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) ?? '';
      gl.deleteShader(shader);
      const kind = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
      throw new RendererError(`${label}: ${kind} shader failed to compile: ${log}`);
    }
    return shader;
  };

  const vert = compile(gl.VERTEX_SHADER, vertSrc);
  const frag = compile(gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram();
  if (!program) throw new RendererError(`could not create the program for ${label}`);
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  // Safe the moment the link is issued: the program holds its own reference,
  // and deleting here means an early return cannot leak the shader objects.
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? '';
    gl.deleteProgram(program);
    throw new RendererError(`${label}: program failed to link: ${log}`);
  }
  return program;
}

/** Allocates an immutable-parameter 2D texture with no initial contents. */
export function createTexture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  fmt: TextureFormat,
  filter: number,
  wrap: number = gl.CLAMP_TO_EDGE,
): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new RendererError('out of texture handles');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internal, Math.max(1, w), Math.max(1, h), 0, fmt.format, fmt.type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  return tex;
}

/**
 * Colour-only framebuffer. `resize` reallocates level 0 in place; the FBO
 * attachment points at the texture object, not at its storage, so it survives.
 */
export class RenderTarget {
  readonly texture: WebGLTexture;
  readonly fbo: WebGLFramebuffer;
  width = 0;
  height = 0;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly fmt: TextureFormat,
    filter: number = gl.LINEAR,
  ) {
    this.texture = createTexture(gl, 1, 1, fmt, filter);
    const fbo = gl.createFramebuffer();
    if (!fbo) throw new RendererError('out of framebuffer handles');
    this.fbo = fbo;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.width = 1;
    this.height = 1;
  }

  resize(w: number, h: number): void {
    const width = Math.max(1, Math.floor(w));
    const height = Math.max(1, Math.floor(h));
    if (width === this.width && height === this.height) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.fmt.internal, width, height, 0, this.fmt.format, this.fmt.type, null);
    this.width = width;
    this.height = height;
  }

  /** Binds for drawing and sets the viewport to match. */
  bind(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.width, this.height);
  }

  complete(): boolean {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return status === gl.FRAMEBUFFER_COMPLETE;
  }

  dispose(): void {
    this.gl.deleteFramebuffer(this.fbo);
    this.gl.deleteTexture(this.texture);
  }
}

/**
 * What the intermediate targets can hold.
 *
 * `range` is the radiance that maps to 1.0 when the chain is stuck on 8-bit
 * targets: every pass then works in `radiance / range`, which keeps additive
 * blending linear (a pure scale) while still representing highlights above
 * display white. On float targets `range` is 1 and every encode is identity.
 */
export interface HdrCaps {
  format: TextureFormat;
  float: boolean;
  range: number;
}

/** Radiance that saturates an 8-bit intermediate target. */
const SDR_RANGE = 6.0;

/**
 * True when this context is backed by a CPU rasteriser.
 *
 * SwiftShader (headless Chromium, and Chrome's fallback when the GPU is
 * blocklisted) and llvmpipe run two to three orders of magnitude slower per
 * pixel than any real GPU, and the difference is entirely in fill rate — a
 * chain that costs a GPU 1.5 ms at 1440p costs SwiftShader a quarter of a
 * second. Knowing this up front lets the renderer trade internal resolution
 * for frame rate deterministically, which is much better than a dynamic
 * controller here: the frame period this renderer can observe also contains
 * the simulation step and, in this app, a synchronous inference call, so a
 * feedback loop would degrade the image in response to costs that are not
 * the renderer's and would make screenshots non-reproducible.
 */
export function isSoftwareRasteriser(gl: WebGL2RenderingContext): boolean {
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  if (!info) return false;
  const name = String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL) ?? '');
  return /swiftshader|software|llvmpipe|basic render/i.test(name);
}

/**
 * Picks the intermediate format.
 *
 * The extension query alone is not enough: SwiftShader and some mobile drivers
 * advertise float colour buffers and then hand back an incomplete framebuffer,
 * so this allocates a scratch target and checks it before committing.
 */
export function probeHdr(gl: WebGL2RenderingContext, forceSdr = false): HdrCaps {
  const rgba8: TextureFormat = { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE };
  if (forceSdr) return { format: rgba8, float: false, range: SDR_RANGE };

  const hasFloat = gl.getExtension('EXT_color_buffer_float') !== null;
  const hasHalf = gl.getExtension('EXT_color_buffer_half_float') !== null;
  if (hasFloat || hasHalf) {
    const half: TextureFormat = { internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT };
    // Drain anything queued by earlier setup so the check below is about this
    // allocation only.
    while (gl.getError() !== gl.NO_ERROR) {
      /* flush */
    }
    const probe = new RenderTarget(gl, half);
    probe.resize(8, 8);
    const ok = probe.complete() && gl.getError() === gl.NO_ERROR;
    probe.dispose();
    if (ok) return { format: half, float: true, range: 1.0 };
  }
  return { format: rgba8, float: false, range: SDR_RANGE };
}
