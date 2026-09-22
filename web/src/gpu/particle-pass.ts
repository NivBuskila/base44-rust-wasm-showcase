/**
 * The WebGPU instanced particle draw, split out of `renderer.ts`.
 *
 * `ParticlePass` owns the draw uniform buffer and the bind group over the live
 * pool. The pool belongs to `ParticleSim` and is reallocated when it grows, so
 * the version it reports — never a size comparison — is what invalidates the
 * bind group here, the same rule the camera texture follows in `scene-pass.ts`.
 */

import { particleGain, particleSize } from "../render/look";
import type { ModeStyle } from "../render/styles";
import type { GpuSimFrame } from "../types";
import type { ParticleSim } from "./particle-sim";
import type { GpuPipelines } from "./pipelines";
import { DRAW_UNIFORM_FLOATS } from "./shaders/particles";
import type { Target } from "./target";

export class ParticlePass {
  private readonly uniform: GPUBuffer;
  private readonly scratch = new Float32Array(DRAW_UNIFORM_FLOATS);
  private bind: GPUBindGroup | null = null;
  /** Pool generation the bind group was built against. */
  private boundPoolVersion = -1;

  constructor(
    private readonly device: GPUDevice,
    private readonly pipes: GpuPipelines,
    private readonly sim: ParticleSim,
  ) {
    this.uniform = device.createBuffer({
      label: "draw-uniform",
      size: DRAW_UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** Points the bind group at the live pool; false while there is none. */
  ensureBind(): boolean {
    const pool = this.sim.pool;
    if (!pool) return false;
    if (this.boundPoolVersion !== this.sim.poolVersion) {
      this.boundPoolVersion = this.sim.poolVersion;
      this.bind = this.device.createBindGroup({
        layout: this.pipes.particleDraw.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: pool } },
        ],
      });
    }
    return this.bind !== null;
  }

  /** One instanced quad per live particle, additive into `scene`. */
  record(
    encoder: GPUCommandEncoder,
    scene: Target,
    count: number,
    style: ModeStyle,
    intensity: number,
    sim: GpuSimFrame,
    pixelScale: number,
  ): void {
    const d = this.scratch;
    d[0] = particleSize(pixelScale, count);
    d[1] = particleGain(style, count);
    d[2] = intensity;
    d[3] = 0;
    d[4] = scene.width;
    d[5] = scene.height;
    d[6] = Math.max(1, sim.gridW - 1);
    d[7] = Math.max(1, sim.gridH - 1);
    this.device.queue.writeBuffer(this.uniform, 0, d);

    const p = encoder.beginRenderPass({
      label: "particles",
      colorAttachments: [
        { view: scene.view!, loadOp: "load", storeOp: "store" },
      ],
    });
    p.setPipeline(this.pipes.particleDraw);
    p.setBindGroup(0, this.bind!);
    p.draw(6, count);
    p.end();
  }
}
