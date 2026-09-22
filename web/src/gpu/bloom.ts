/**
 * The bloom chain: a 13-tap downsample ladder and a 9-tap tent fold-back,
 * matching the WebGL2 path pass for pass.
 *
 * It owns its own targets, uniforms and bind groups so the renderer only has
 * to tell it the scene size (`resize`), hand it the scene view when a resize
 * invalidated the bind groups (`buildBinds`) and record the passes
 * (`record`). Level 0 always exists — the composite samples it — even when the
 * canvas has not been laid out yet.
 */

import type { ModeStyle } from "../render/styles";
import type { FullscreenPass } from "./target";
import { Target } from "./target";
import { BLOOM_UNIFORM_FLOATS } from "./shaders/bloom";

const BLOOM_LEVELS = 5;
const BLOOM_TENT = 1.1;

export class BloomChain {
  private readonly levels: Target[];
  private readonly downUniforms: GPUBuffer[];
  private readonly upUniforms: GPUBuffer[];
  private downBinds: GPUBindGroup[] = [];
  private upBinds: GPUBindGroup[] = [];
  private count = 1;
  private readonly scratch = new Float32Array(4);

  constructor(
    private readonly device: GPUDevice,
    private readonly downPipe: GPURenderPipeline,
    private readonly upPipe: GPURenderPipeline,
    private readonly sampler: GPUSampler,
    format: GPUTextureFormat,
  ) {
    this.levels = Array.from(
      { length: BLOOM_LEVELS },
      (_, i) => new Target(device, format, `bloom${i}`),
    );
    const uniform = (label: string): GPUBuffer =>
      device.createBuffer({
        label,
        size: BLOOM_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    this.downUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) =>
      uniform(`down${i}`),
    );
    this.upUniforms = Array.from({ length: BLOOM_LEVELS }, (_, i) =>
      uniform(`up${i}`),
    );
  }

  /** The level the composite samples. */
  get output(): GPUTextureView {
    return this.levels[0].view!;
  }

  /** Halves down from the scene size; true when a target was reallocated. */
  resize(sceneW: number, sceneH: number): boolean {
    let dirty = false;
    let lw = sceneW;
    let lh = sceneH;
    let levels = 0;
    while (levels < BLOOM_LEVELS) {
      const nw = Math.max(1, lw >> 1);
      const nh = Math.max(1, lh >> 1);
      if (levels > 0 && (nw < 8 || nh < 8)) break;
      dirty = this.levels[levels].resize(nw, nh) || dirty;
      lw = nw;
      lh = nh;
      levels++;
    }
    this.count = Math.max(1, levels);
    return dirty;
  }

  /** Rebuilds the per-level bind groups; `scene` is the ladder's level -1. */
  buildBinds(scene: Target): void {
    const d = this.device;
    const sampled = (
      pipe: GPURenderPipeline,
      view: GPUTextureView,
      uniformBuffer: GPUBuffer,
    ): GPUBindGroup =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: view },
          { binding: 2, resource: this.sampler },
        ],
      });

    this.downBinds = [];
    this.upBinds = [];
    for (let i = 0; i < this.count; i++) {
      const src = i === 0 ? scene : this.levels[i - 1];
      this.downBinds.push(sampled(this.downPipe, src.view!, this.downUniforms[i]));
      this.upBinds.push(
        sampled(this.upPipe, this.levels[i].view!, this.upUniforms[i]),
      );
    }
  }

  record(
    encoder: GPUCommandEncoder,
    pass: FullscreenPass,
    style: ModeStyle,
    scene: Target,
  ): void {
    const knee = Math.max(0.05, style.threshold * 0.7);
    const b = this.scratch;
    for (let i = 0; i < this.count; i++) {
      const src = i === 0 ? scene : this.levels[i - 1];
      b[0] = 1 / src.width;
      b[1] = 1 / src.height;
      b[2] = i === 0 ? style.threshold : 0;
      b[3] = knee;
      this.device.queue.writeBuffer(this.downUniforms[i], 0, b);
      pass(encoder, this.levels[i].view!, this.downPipe, this.downBinds[i]);
    }
    for (let i = this.count - 1; i > 0; i--) {
      const src = this.levels[i];
      b[0] = BLOOM_TENT / src.width;
      b[1] = BLOOM_TENT / src.height;
      b[2] = 1;
      b[3] = 0;
      this.device.queue.writeBuffer(this.upUniforms[i], 0, b);
      pass(encoder, this.levels[i - 1].view!, this.upPipe, this.upBinds[i], "load");
    }
  }

  dispose(): void {
    for (const level of this.levels) level.dispose();
  }
}
