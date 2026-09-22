/**
 * The input textures of the WebGPU frame and their uploads: the dye grid, the
 * debug grid, the static noise tile, the camera texture and the three samplers
 * every pass shares.
 *
 * Split out of `renderer.ts` by lifetime, like the rest of `gpu/`: these
 * outlive a resize and die only with the renderer — except the camera texture,
 * which is reallocated when the stream's resolution changes. That bumps
 * `version`, and a change in `version` is the *only* thing that may invalidate
 * a bind group over these views, exactly as `ParticleSim.poolVersion` is for
 * the pool.
 */

import { FLUID_H, FLUID_W } from "../constants";
import { NOISE_SIZE, buildNoiseTile } from "../render/noise";

export class SceneSources {
  readonly dyeTex: GPUTexture;
  readonly debugTex: GPUTexture;
  readonly noiseTex: GPUTexture;
  videoTex: GPUTexture;
  videoTexW = 0;
  videoTexH = 0;
  /** Bumped whenever the camera texture is reallocated. */
  version = 0;

  readonly clampSampler: GPUSampler;
  readonly repeatSampler: GPUSampler;
  readonly nearestSampler: GPUSampler;

  private videoTime = -1;

  constructor(private readonly device: GPUDevice) {
    const d = device;
    this.clampSampler = d.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.repeatSampler = d.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
    this.nearestSampler = d.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });

    const grid = (label: string): GPUTexture =>
      d.createTexture({
        label,
        size: { width: FLUID_W, height: FLUID_H },
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
    this.dyeTex = grid("dye");
    this.debugTex = grid("debug");
    this.noiseTex = d.createTexture({
      label: "noise",
      size: { width: NOISE_SIZE, height: NOISE_SIZE },
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    d.queue.writeTexture(
      { texture: this.noiseTex },
      buildNoiseTile() as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: NOISE_SIZE * 4 },
      { width: NOISE_SIZE, height: NOISE_SIZE },
    );
    this.videoTex = this.createVideoTexture(2, 2);
  }

  /** Forgets the last uploaded camera frame, so the next one is copied again. */
  resetVideo(): void {
    this.videoTime = -1;
  }

  /** True when the camera texture holds a usable frame. */
  uploadVideo(v: HTMLVideoElement | null): boolean {
    if (!v || v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0)
      return false;
    if (this.videoTexW !== v.videoWidth || this.videoTexH !== v.videoHeight) {
      this.videoTex.destroy();
      this.videoTex = this.createVideoTexture(v.videoWidth, v.videoHeight);
      this.videoTime = -1;
      this.version++;
    }
    if (v.currentTime !== this.videoTime) {
      try {
        this.device.queue.copyExternalImageToTexture(
          { source: v },
          { texture: this.videoTex },
          { width: v.videoWidth, height: v.videoHeight },
        );
      } catch {
        // Not decodable yet, or a stream the copy refuses; keep the last frame.
        return this.videoTime >= 0;
      }
      this.videoTime = v.currentTime;
    }
    return true;
  }

  /** Uploads one `FLUID_W * FLUID_H` RGBA8 grid; false when it is short. */
  uploadGrid(tex: GPUTexture, data: Uint8Array): boolean {
    const bytes = FLUID_W * FLUID_H * 4;
    if (data.length < bytes) return false;
    this.device.queue.writeTexture(
      { texture: tex },
      // A view over WASM memory, which may be a SharedArrayBuffer.
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: FLUID_W * 4, rowsPerImage: FLUID_H },
      { width: FLUID_W, height: FLUID_H },
    );
    return true;
  }

  dispose(): void {
    for (const tex of [
      this.dyeTex,
      this.debugTex,
      this.noiseTex,
      this.videoTex,
    ])
      tex.destroy();
  }

  private createVideoTexture(w: number, h: number): GPUTexture {
    this.videoTexW = w;
    this.videoTexH = h;
    return this.device.createTexture({
      label: "video",
      size: { width: w, height: h },
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
}
