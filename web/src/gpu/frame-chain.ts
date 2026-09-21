/**
 * The WebGPU post chain's resources, split out of `renderer.ts` by lifetime
 * like the rest of `gpu/`.
 *
 * `FrameChain` owns the four sized things the frame draws through — the HDR
 * scene target, the bloom ladder, the 8-bit frame texture and the luminance
 * probe — plus the two bind groups that reference their views and therefore die
 * with every resize. The renderer keeps recording the passes; this only answers
 * "which view, which bind group, at what size", so the resize/rebuild rule
 * lives in one place instead of being spread across `resize` and
 * `buildTargetBinds`.
 *
 * The composite uniform buffer outlives a resize, so it is the renderer's and
 * is only referenced here when the composite bind group is rebuilt.
 */

import type { ModeStyle } from "../render/styles";
import { BloomChain } from "./bloom";
import type { GpuPipelines } from "./pipelines";
import { HDR_FORMAT } from "./pipelines";
import { LuminanceProbe } from "./probe";
import type { SceneSources } from "./sources";
import type { FullscreenPass } from "./target";
import { Target } from "./target";

export class FrameChain {
  /** HDR target every light-emitting draw lands in. */
  readonly scene: Target;
  private readonly bloom: BloomChain;
  private readonly frameTarget: Target;
  private readonly probe: LuminanceProbe;

  private compositeBind: GPUBindGroup | null = null;
  private blitFrameBind: GPUBindGroup | null = null;

  constructor(
    private readonly device: GPUDevice,
    private readonly pipes: GpuPipelines,
    private readonly sources: SceneSources,
    private readonly compositeUniform: GPUBuffer,
  ) {
    this.scene = new Target(device, HDR_FORMAT, "scene");
    this.frameTarget = new Target(device, "rgba8unorm", "frame");
    this.bloom = new BloomChain(
      device,
      pipes.down,
      pipes.up,
      sources.clampSampler,
      HDR_FORMAT,
    );
    this.probe = new LuminanceProbe(device, pipes.probe, sources.clampSampler);
  }

  /**
   * Matches every target to this frame's sizes: the canvas-sized frame texture,
   * the scene and bloom ladder at the quality-scaled size. Rebuilds the bind
   * groups over them only when a texture was actually reallocated.
   */
  resize(
    canvasW: number,
    canvasH: number,
    sceneW: number,
    sceneH: number,
  ): void {
    let dirty = this.scene.resize(sceneW, sceneH);
    dirty = this.bloom.resize(sceneW, sceneH) || dirty;
    dirty = this.frameTarget.resize(canvasW, canvasH) || dirty;
    dirty = this.probe.resize() || dirty;
    if (dirty) this.buildBinds();
  }

  private buildBinds(): void {
    const d = this.device;
    this.bloom.buildBinds(this.scene);
    this.compositeBind = d.createBindGroup({
      layout: this.pipes.composite.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.compositeUniform } },
        { binding: 1, resource: this.scene.view! },
        { binding: 2, resource: this.bloom.output },
        { binding: 3, resource: this.sources.noiseTex.createView() },
        { binding: 4, resource: this.sources.clampSampler },
      ],
    });
    this.blitFrameBind = d.createBindGroup({
      layout: this.pipes.blit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.frameTarget.view! },
        { binding: 1, resource: this.sources.clampSampler },
      ],
    });
    this.probe.buildBind(this.frameTarget.view!);
  }

  /** Bloom: downsample chain then tent fold-back, back into `scene`. */
  recordBloom(
    encoder: GPUCommandEncoder,
    pass: FullscreenPass,
    style: ModeStyle,
  ): void {
    this.bloom.record(encoder, pass, style, this.scene);
  }

  /** Grades the scene into the 8-bit frame texture. */
  recordComposite(encoder: GPUCommandEncoder, pass: FullscreenPass): void {
    pass(
      encoder,
      this.frameTarget.view!,
      this.pipes.composite,
      this.compositeBind!,
    );
  }

  /** Frame texture to the canvas, plus the 16x16 luminance probe. */
  recordBlit(
    encoder: GPUCommandEncoder,
    pass: FullscreenPass,
    canvasView: GPUTextureView,
  ): void {
    pass(encoder, canvasView, this.pipes.blit, this.blitFrameBind!);
    this.probe.record(encoder, pass);
  }

  /** Starts the probe's read. Must run after the submit, never before it. */
  pollProbe(): void {
    this.probe.poll();
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  get luminance(): number {
    return this.probe.luminance;
  }

  dispose(): void {
    this.scene.dispose();
    this.bloom.dispose();
    this.frameTarget.dispose();
    this.probe.dispose();
  }
}
