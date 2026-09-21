/**
 * The hand-skeleton overlay pass, split out of `renderer.ts`: it owns the only
 * vertex buffer in the WebGPU chain plus the two uniform buffers the bones and
 * the joints need, and nothing else in the frame touches them.
 *
 * Two uniform buffers, not one, and two render passes for one overlay: a
 * `queue.writeBuffer` lands before the encoder is submitted, so writing the
 * joint values into the buffer the line draw was recorded against would change
 * that draw too.
 */

import {
  OVERLAY_CAPACITY,
  OVERLAY_STRIDE,
  buildHandMesh,
} from "../render/handmesh";
import { OVERLAY_UNIFORM_FLOATS } from "./shaders/overlay";
import type { Target } from "./target";

/** Joint quad radius in scene pixels, before dpr and the scene scale. */
const JOINT_RADIUS = 5;

export class HandOverlay {
  private readonly lineUniform: GPUBuffer;
  private readonly pointUniform: GPUBuffer;
  private readonly vertices: GPUBuffer;
  private readonly lineBind: GPUBindGroup;
  private readonly pointBind: GPUBindGroup;
  private readonly scratch = new Float32Array(8);
  private readonly verts = new Float32Array(OVERLAY_CAPACITY * OVERLAY_STRIDE);

  constructor(
    private readonly device: GPUDevice,
    private readonly linePipe: GPURenderPipeline,
    private readonly pointPipe: GPURenderPipeline,
  ) {
    const uniform = (label: string): GPUBuffer =>
      device.createBuffer({
        label,
        size: OVERLAY_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.lineUniform = uniform("overlay-lines-uniform");
    this.pointUniform = uniform("overlay-points-uniform");
    this.vertices = device.createBuffer({
      label: "overlay",
      size: this.verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const bind = (pipe: GPURenderPipeline, buffer: GPUBuffer): GPUBindGroup =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
      });
    this.lineBind = bind(linePipe, this.lineUniform);
    this.pointBind = bind(pointPipe, this.pointUniform);
  }

  /**
   * Records the bones and joints over `target`. `strength` is the style's
   * overlay weight and `pointScale` folds dpr and the scene scale into the
   * joint radius, so the dots keep their on-screen size.
   */
  record(
    encoder: GPUCommandEncoder,
    target: Target,
    hands: Float32Array,
    strength: number,
    pointScale: number,
  ): void {
    const mesh = buildHandMesh(hands, this.verts);
    const total = mesh.lineVertices + mesh.pointVertices;
    if (total === 0) return;
    const queue = this.device.queue;
    queue.writeBuffer(this.vertices, 0, this.verts, 0, total * OVERLAY_STRIDE);

    const u = this.scratch;
    u[0] = strength;
    u[2] = target.width;
    u[3] = target.height;
    u[4] = 0.55;
    u[5] = 0.88;
    u[6] = 1.0;

    const begin = (label: string): GPURenderPassEncoder =>
      encoder.beginRenderPass({
        label,
        colorAttachments: [
          { view: target.view!, loadOp: "load", storeOp: "store" },
        ],
      });

    if (mesh.lineVertices > 0) {
      u[1] = 1;
      u[7] = 0;
      queue.writeBuffer(this.lineUniform, 0, u);
      const p = begin("overlay");
      p.setPipeline(this.linePipe);
      p.setBindGroup(0, this.lineBind);
      p.setVertexBuffer(0, this.vertices);
      p.draw(mesh.lineVertices);
      p.end();
    }
    if (mesh.pointVertices > 0) {
      u[1] = JOINT_RADIUS * pointScale;
      u[7] = 1;
      queue.writeBuffer(this.pointUniform, 0, u);
      const q = begin("overlay-joints");
      q.setPipeline(this.pointPipe);
      q.setBindGroup(0, this.pointBind);
      q.setVertexBuffer(
        0,
        this.vertices,
        mesh.lineVertices * OVERLAY_STRIDE * 4,
        mesh.pointVertices * OVERLAY_STRIDE * 4,
      );
      q.draw(6, mesh.pointVertices);
      q.end();
    }
  }

  dispose(): void {
    this.lineUniform.destroy();
    this.pointUniform.destroy();
    this.vertices.destroy();
  }
}
