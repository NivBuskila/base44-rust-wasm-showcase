/**
 * The input textures of the WebGL2 frame and their uploads: the dye grid, the
 * debug grid, the static noise tile and the camera texture.
 *
 * The mirror of `gpu/sources.ts`. These die with the GL context, so the whole
 * group is created inside `Resources` and released with it — including the
 * camera's allocation state, which is why `videoTexW`/`videoTexH` live here and
 * not in the renderer.
 */

import { FLUID_H, FLUID_W } from "../constants";
import { createTexture } from "./gl";
import { NOISE_SIZE, buildNoiseTile } from "./noise";

export class SceneSources {
  readonly dyeTex: WebGLTexture;
  readonly debugTex: WebGLTexture;
  readonly videoTex: WebGLTexture;
  readonly noiseTex: WebGLTexture;

  /** Camera texture allocation state; `0` means "not allocated yet". */
  videoTexW = 0;
  videoTexH = 0;
  /** Last uploaded video timestamp, so a 30 Hz feed is not re-uploaded at 60. */
  private videoTime = -1;

  constructor(private readonly gl: WebGL2RenderingContext) {
    const rgba8 = {
      internal: gl.RGBA8,
      format: gl.RGBA,
      type: gl.UNSIGNED_BYTE,
    };
    this.dyeTex = createTexture(gl, FLUID_W, FLUID_H, rgba8, gl.LINEAR);
    // NEAREST for the debug view on purpose: it is meant to show the raw
    // 256x144 cells, and interpolating them would hide exactly what it is for.
    this.debugTex = createTexture(gl, FLUID_W, FLUID_H, rgba8, gl.NEAREST);
    this.videoTex = createTexture(gl, 2, 2, rgba8, gl.LINEAR);
    this.noiseTex = createTexture(
      gl,
      NOISE_SIZE,
      NOISE_SIZE,
      rgba8,
      gl.LINEAR,
      gl.REPEAT,
    );
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      NOISE_SIZE,
      NOISE_SIZE,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      buildNoiseTile(),
    );
  }

  /** Forgets the last uploaded camera frame, so the next one is uploaded again. */
  resetVideo(): void {
    this.videoTime = -1;
  }

  /** Camera aspect ratio, or 0 before the first allocation. */
  get videoAspect(): number {
    return this.videoTexW / Math.max(1, this.videoTexH);
  }

  /**
   * True when the video texture holds a usable frame.
   *
   * `readyState < 2` is the case that matters: a camera element exists from the
   * moment `getUserMedia` resolves but has no decoded frame for a while after,
   * and uploading one of those is either a no-op or a throw depending on the
   * browser.
   */
  uploadVideo(v: HTMLVideoElement | null): boolean {
    if (!v || v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0)
      return false;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);

    if (this.videoTexW !== v.videoWidth || this.videoTexH !== v.videoHeight) {
      // Only reallocation path for the video texture: the camera's resolution
      // is fixed for the life of a stream, so this runs once.
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA8,
        v.videoWidth,
        v.videoHeight,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
      this.videoTexW = v.videoWidth;
      this.videoTexH = v.videoHeight;
      this.videoTime = -1;
    }

    if (v.currentTime !== this.videoTime) {
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
      try {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, v);
      } catch {
        // A frame that is not decodable yet, or a cross-origin stream. Keep
        // whatever was uploaded last rather than dropping the camera layer.
        return this.videoTime >= 0;
      }
      this.videoTime = v.currentTime;
    }
    return true;
  }

  /**
   * Uploads one `FLUID_W * FLUID_H` RGBA8 grid. A short buffer means the view
   * over WASM memory was detached by a reallocation this frame — skip the
   * upload and keep the previous contents rather than throwing out of the
   * render loop.
   */
  uploadGrid(tex: WebGLTexture, data: Uint8Array): boolean {
    if (data.length < FLUID_W * FLUID_H * 4) return false;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      FLUID_W,
      FLUID_H,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      data,
    );
    return true;
  }

  dispose(): void {
    for (const tex of [
      this.dyeTex,
      this.debugTex,
      this.videoTex,
      this.noiseTex,
    ]) {
      this.gl.deleteTexture(tex);
    }
  }
}
