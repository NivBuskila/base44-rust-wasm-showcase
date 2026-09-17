/**
 * WebGL2 renderer.
 *
 * MINIMAL BASELINE — this draws the dye field and nothing else. It exists so
 * the app is verifiable end to end from the first commit; the full compositor
 * (camera feed, instanced particles, bloom, colour grading, landmark overlay)
 * replaces the body of `render` and the shaders while keeping this public
 * surface intact.
 */

import { FLUID_H, FLUID_W } from '../constants';
import type { RenderFrame } from '../types';

const VERT = /* glsl */ `#version 300 es
// Fullscreen triangle strip from gl_VertexID: no vertex buffer needed.
out vec2 v_uv;
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 1), float(gl_VertexID & 2)) - 1.0;
  v_uv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_dye;
uniform float u_intensity;
out vec4 o_color;
void main() {
  // The dye texture's row 0 is the top of the simulation, but GL samples with
  // y up, so the V coordinate is flipped here rather than in the upload.
  vec3 dye = texture(u_dye, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
  vec3 lifted = dye * (0.85 + 0.6 * u_intensity);
  o_color = vec4(lifted, 1.0);
}`;

/**
 * `sampleLuminance` reduces the frame to an NxN block read from the centre.
 * A single pixel is not enough: the fluid is sparse, so the centre pixel is
 * legitimately black much of the time and a one-pixel probe cannot tell that
 * apart from a dead render path.
 */
const LUMA_SAMPLE_GRID = 16;

export class RendererError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RendererError';
  }
}

export class Renderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly dyeTexture: WebGLTexture;
  private readonly uDye: WebGLUniformLocation | null;
  private readonly uIntensity: WebGLUniformLocation | null;
  private video: HTMLVideoElement | null = null;
  private lostContext = false;
  /** One-row scratch for `sampleLuminance`, grown to the canvas width. */
  private sampleRow = new Uint8Array(4);

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      // Needed so `sampleLuminance` and Playwright screenshots can read the
      // framebuffer after the frame is drawn.
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new RendererError('WebGL2 is unavailable in this browser.');
    this.gl = gl;

    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.lostContext = true;
      console.error('[aether] WebGL context lost');
    });
    canvas.addEventListener('webglcontextrestored', () => {
      this.lostContext = false;
    });

    this.program = this.buildProgram(VERT, FRAG);
    this.uDye = gl.getUniformLocation(this.program, 'u_dye');
    this.uIntensity = gl.getUniformLocation(this.program, 'u_intensity');

    this.dyeTexture = this.createTexture(FLUID_W, FLUID_H);
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  private buildProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string): WebGLShader => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new RendererError(`shader compile failed: ${log}`);
      }
      return shader;
    };

    const vert = compile(gl.VERTEX_SHADER, vertSrc);
    const frag = compile(gl.FRAGMENT_SHADER, fragSrc);
    const program = gl.createProgram()!;
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    // Shaders can be deleted immediately after a successful link; the program
    // holds its own reference.
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new RendererError(`program link failed: ${log}`);
    }
    return program;
  }

  private createTexture(w: number, h: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    // LINEAR so the 256x144 dye field upscales smoothly to a 4K canvas;
    // CLAMP_TO_EDGE so the border does not wrap colour across the screen.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Matches the drawing buffer to the CSS size and device pixel ratio. */
  resize(): void {
    // Capped at 2: beyond that the fill cost buys nothing visible for a field
    // this soft, and it halves the frame rate on phones.
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
  }

  get hasVideo(): boolean {
    return this.video !== null;
  }

  render(frame: RenderFrame): void {
    if (this.lostContext) return;
    const gl = this.gl;
    this.resize();

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.02, 0.02, 0.04, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const source = frame.mode === 'debug' && frame.debug ? frame.debug : frame.dye;
    gl.bindTexture(gl.TEXTURE_2D, this.dyeTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      FLUID_W,
      FLUID_H,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      source,
    );

    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dyeTexture);
    if (this.uDye) gl.uniform1i(this.uDye, 0);
    if (this.uIntensity) gl.uniform1f(this.uIntensity, frame.intensity);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /**
   * Mean luminance over a centre block of the rendered frame, `[0, 1]`.
   *
   * Used by the headless tests to tell "something is being drawn" from "the
   * render path is dead". Reads back `LUMA_SAMPLE_GRID^2` pixels stretched
   * across the frame rather than a contiguous block, so a bright region
   * anywhere registers.
   */
  sampleLuminance(): number {
    if (this.lostContext) return 0;
    const gl = this.gl;
    const n = LUMA_SAMPLE_GRID;
    const stepX = Math.max(1, Math.floor(this.canvas.width / n));
    const stepY = Math.max(1, Math.floor(this.canvas.height / n));

    const rowBytes = this.canvas.width * 4;
    if (this.sampleRow.length < rowBytes) this.sampleRow = new Uint8Array(rowBytes);

    let total = 0;
    let count = 0;
    // One readPixels per sampled row: n calls instead of n^2, while still
    // spanning the whole frame.
    for (let row = 0; row < n; row++) {
      const y = Math.min(this.canvas.height - 1, row * stepY);
      gl.readPixels(0, y, this.canvas.width, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.sampleRow);
      for (let i = 0; i < n; i++) {
        const p = Math.min(this.canvas.width - 1, i * stepX) * 4;
        total +=
          0.2126 * this.sampleRow[p] +
          0.7152 * this.sampleRow[p + 1] +
          0.0722 * this.sampleRow[p + 2];
        count++;
      }
    }
    return count === 0 ? 0 : total / count / 255;
  }
}
