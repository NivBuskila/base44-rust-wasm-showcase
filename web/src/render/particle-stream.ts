/**
 * The WebGL2 particle pool's one vertex buffer, split out of `renderer.ts`.
 *
 * Mirrors what `gpu/particle-sim.ts` is on the WebGPU side — minus the
 * simulation, which stays in the engine on this path: this only owns the VAO,
 * the streamed buffer and the orphan/refill rule that keeps a 220k-particle
 * upload from blocking on the GPU still reading last frame's data.
 *
 * It dies and is rebuilt with the GL context, like every other piece inside
 * `Resources`, and it leaves its own VAO bound — the caller rebinds `quadVao`
 * before the next fullscreen strip, exactly as the overlay requires.
 */

import { MAX_PARTICLES, PARTICLE_STRIDE } from "../constants";
import { RendererError } from "./gl";

/**
 * Particle buffer sizes are rounded up to this many particles.
 *
 * The buffer is re-specified every frame to orphan it, and a size that changed
 * every frame would make the driver allocate a differently sized block each
 * time. Bucketing keeps the allocation stable across frames while still
 * tracking a pool the user shrank from 220k to 2k.
 */
export const PARTICLE_BUCKET = 32768;

export class ParticleStream {
  private readonly buffer: WebGLBuffer;
  private readonly vao: WebGLVertexArrayObject;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const buffer = gl.createBuffer();
    const vao = gl.createVertexArray();
    if (!buffer || !vao)
      throw new RendererError("could not allocate the particle buffer");
    this.buffer = buffer;
    this.vao = vao;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      PARTICLE_BUCKET * PARTICLE_STRIDE * 4,
      gl.DYNAMIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, PARTICLE_STRIDE * 4, 0);
    gl.bindVertexArray(null);
  }

  /**
   * How many particles this frame may draw: the engine's count, clamped to what
   * the interleaved buffer actually holds and to the pool ceiling.
   */
  static countFor(particles: Float32Array, requestedCount: number): number {
    const available = Math.floor(particles.length / PARTICLE_STRIDE);
    const requested = Number.isFinite(requestedCount)
      ? Math.floor(requestedCount)
      : 0;
    return Math.min(Math.max(0, requested), available, MAX_PARTICLES);
  }

  /** Binds the VAO and streams `count` particles into the orphaned buffer. */
  upload(particles: Float32Array, count: number): void {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    // Orphan, then refill. Re-specifying the whole buffer tells the driver the
    // old contents are dead, so this upload never blocks on the GPU still
    // reading last frame's data — at 220k particles that stall is a dropped
    // frame every frame.
    const bucket = Math.min(
      MAX_PARTICLES,
      Math.ceil(count / PARTICLE_BUCKET) * PARTICLE_BUCKET,
    );
    gl.bufferData(
      gl.ARRAY_BUFFER,
      bucket * PARTICLE_STRIDE * 4,
      gl.DYNAMIC_DRAW,
    );
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, particles, 0, count * PARTICLE_STRIDE);
  }

  /** One additive `POINTS` draw into whichever target is bound. */
  draw(count: number): void {
    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);
  }

  dispose(): void {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteVertexArray(this.vao);
  }
}
