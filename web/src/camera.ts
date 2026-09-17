/**
 * Camera capture and luma extraction.
 *
 * The engine's optical flow wants a small grayscale plane, not a video frame.
 * Doing that downscale on the GPU via `drawImage` into a tiny 2D canvas and
 * reading back `FLOW_W * FLOW_H` pixels costs far less than reading back the
 * full frame and averaging in JS — the readback is the expensive part, so the
 * trick is to make the thing being read back small.
 */

import { FLOW_H, FLOW_W } from './constants';

export interface CameraOptions {
  width: number;
  height: number;
  facingMode: 'user' | 'environment';
}

const DEFAULTS: CameraOptions = { width: 1280, height: 720, facingMode: 'user' };

export class CameraError extends Error {
  constructor(
    message: string,
    /** True when the user or the environment denied camera access, as opposed
     * to there being no camera at all — the HUD wording differs. */
    readonly denied: boolean,
  ) {
    super(message);
    this.name = 'CameraError';
  }
}

export class Camera {
  readonly video: HTMLVideoElement;
  private stream: MediaStream | null = null;

  /** Small offscreen canvas used purely as a downscaler. */
  private readonly scratch: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly luma: Uint8Array;

  constructor() {
    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.autoplay = true;

    this.scratch = document.createElement('canvas');
    this.scratch.width = FLOW_W;
    this.scratch.height = FLOW_H;
    const ctx = this.scratch.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D canvas context unavailable; cannot extract luma.');
    this.ctx = ctx;
    this.luma = new Uint8Array(FLOW_W * FLOW_H);
  }

  async start(options: Partial<CameraOptions> = {}): Promise<void> {
    const opts = { ...DEFAULTS, ...options };
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new CameraError('This browser exposes no camera API.', false);
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: opts.width },
          height: { ideal: opts.height },
          facingMode: opts.facingMode,
        },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : '';
      const denied = name === 'NotAllowedError' || name === 'SecurityError';
      throw new CameraError(
        denied
          ? 'Camera permission was denied.'
          : `No usable camera found (${name || String(err)}).`,
        denied,
      );
    }

    this.video.srcObject = this.stream;
    await this.video.play();
    await this.waitForFrame();
  }

  /**
   * Resolves once the video actually has pixels. `play()` resolving is not
   * enough — the first frames can still be 0x0, and drawing one of those
   * throws in some browsers.
   */
  private waitForFrame(): Promise<void> {
    if (this.hasFrame) return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => {
        if (this.hasFrame) resolve();
        else requestAnimationFrame(check);
      };
      check();
    });
  }

  get hasFrame(): boolean {
    return this.video.readyState >= 2 && this.video.videoWidth > 0 && this.video.videoHeight > 0;
  }

  get width(): number {
    return this.video.videoWidth;
  }

  get height(): number {
    return this.video.videoHeight;
  }

  /**
   * Downscales the current frame and returns its luma plane.
   *
   * `mirror` flips horizontally so the plane matches the mirrored preview the
   * user sees — without it, moving your hand right would push the fluid left.
   * Returns null when no frame is available yet.
   */
  readLuma(mirror = true): Uint8Array | null {
    if (!this.hasFrame) return null;

    this.ctx.save();
    if (mirror) {
      this.ctx.translate(FLOW_W, 0);
      this.ctx.scale(-1, 1);
    }
    this.ctx.drawImage(this.video, 0, 0, FLOW_W, FLOW_H);
    this.ctx.restore();

    const { data } = this.ctx.getImageData(0, 0, FLOW_W, FLOW_H);
    // Rec. 709 luma in fixed point: the shifts keep this in the integer unit
    // and it runs on every camera frame.
    for (let i = 0, p = 0; i < this.luma.length; i++, p += 4) {
      this.luma[i] = (data[p] * 54 + data[p + 1] * 183 + data[p + 2] * 19) >> 8;
    }
    return this.luma;
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}
