/**
 * The hand-skeleton overlay pass, mirroring `gpu/overlay.ts`: it owns the
 * overlay VAO, its dynamic vertex buffer and the mesh scratch, and nothing else
 * in the frame touches them.
 *
 * It is additive so the bloom picks the skeleton up, and it leaves its own VAO
 * bound — the caller re-binds `quadVao` before the fullscreen passes.
 *
 * Lives and dies with the GL context, inside `Resources`.
 */

import { RendererError } from "./gl";
import type { Program, RenderTarget } from "./gl";
import { OVERLAY_CAPACITY, OVERLAY_STRIDE, buildHandMesh } from "./handmesh";

/** Joint point size in scene pixels, before dpr and the scene scale. */
const JOINT_SIZE = 5;

export class HandOverlay {
  private readonly buffer: WebGLBuffer;
  private readonly vao: WebGLVertexArrayObject;
  private readonly scratch = new Float32Array(
    OVERLAY_CAPACITY * OVERLAY_STRIDE,
  );

  constructor(private readonly gl: WebGL2RenderingContext) {
    const buffer = gl.createBuffer();
    const vao = gl.createVertexArray();
    if (!buffer || !vao)
      throw new RendererError("could not allocate the overlay vertex buffer");
    this.buffer = buffer;
    this.vao = vao;

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.scratch.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, OVERLAY_STRIDE * 4, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, OVERLAY_STRIDE * 4, 8);
    gl.bindVertexArray(null);
  }

  /**
   * Draws the bones and joints into `target`. `strength` is the style's overlay
   * weight and `pointScale` folds dpr and the scene scale into the joint size,
   * so the dots keep their on-screen size.
   */
  draw(
    program: Program,
    target: RenderTarget,
    hands: Float32Array,
    strength: number,
    pointScale: number,
  ): void {
    const mesh = buildHandMesh(hands, this.scratch);
    const vertices = mesh.lineVertices + mesh.pointVertices;
    if (vertices === 0) return;

    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.scratch.byteLength, gl.DYNAMIC_DRAW);
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      this.scratch,
      0,
      vertices * OVERLAY_STRIDE,
    );

    program.use();
    program.f1("u_alpha", strength);
    program.f3("u_tint", 0.55, 0.88, 1.0);

    target.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (mesh.lineVertices > 0) {
      program.f1("u_round", 0);
      program.f1("u_size", 1);
      gl.drawArrays(gl.LINES, 0, mesh.lineVertices);
    }
    if (mesh.pointVertices > 0) {
      program.f1("u_round", 1);
      program.f1("u_size", JOINT_SIZE * pointScale);
      gl.drawArrays(gl.POINTS, mesh.lineVertices, mesh.pointVertices);
    }
    gl.disable(gl.BLEND);
  }

  dispose(): void {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteVertexArray(this.vao);
  }
}
