/**
 * A colour target plus its size, resized in place.
 *
 * Shared by the renderer and the bloom chain: every intermediate in the
 * WebGPU post chain is one of these, and nothing on the steady-state path
 * reallocates — `resize` only rebuilds when the size actually changed, and
 * returns whether it did so callers know when their bind groups died.
 */
export class Target {
  texture: GPUTexture | null = null;
  view: GPUTextureView | null = null;
  width = 0;
  height = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly format: GPUTextureFormat,
    private readonly label: string,
    private readonly extraUsage: number = 0,
  ) {}

  resize(w: number, h: number): boolean {
    if (this.width === w && this.height === h && this.texture) return false;
    this.texture?.destroy();
    this.texture = this.device.createTexture({
      label: this.label,
      size: { width: w, height: h },
      format: this.format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        this.extraUsage,
    });
    this.view = this.texture.createView();
    this.width = w;
    this.height = h;
    return true;
  }

  dispose(): void {
    this.texture?.destroy();
    this.texture = null;
    this.view = null;
    this.width = 0;
    this.height = 0;
  }
}

/** Records one fullscreen-strip render pass over `view`. */
export type FullscreenPass = (
  encoder: GPUCommandEncoder,
  view: GPUTextureView,
  pipe: GPURenderPipeline,
  bind: GPUBindGroup,
  load?: GPULoadOp,
) => void;
