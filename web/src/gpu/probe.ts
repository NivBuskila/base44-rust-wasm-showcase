/**
 * The luminance probe: a 16x16 reduction of the finished frame, read back
 * asynchronously so `sampleLuminance` costs no queue stall. Sample at 10 Hz:
 * this diagnostic does not need a GPU pass and readback every render frame.
 *
 * The copy is skipped while a read is in flight — a submission touching a
 * mapped buffer is dropped whole by the driver — which is `Readback`'s rule,
 * asked here through `idle`.
 */

import { Readback } from "./readback";
import type { FullscreenPass } from "./target";
import { Target } from "./target";

/** Side of the luminance probe. */
const PROBE = 16;
/** `copyTextureToBuffer` wants rows aligned to 256 bytes. */
const PROBE_ROW_BYTES = 256;
/** Diagnostics only: one read per six 60 Hz frames is enough. */
const PROBE_EVERY_FRAMES = 6;

/** Mean Rec.709 luminance of the probe readback, `[0, 1]`, rows 256-aligned. */
export function meanLuminance(px: Uint8Array): number {
  let total = 0;
  for (let row = 0; row < PROBE; row++) {
    const base = row * PROBE_ROW_BYTES;
    for (let i = 0; i < PROBE; i++) {
      const p = base + i * 4;
      total += 0.2126 * px[p] + 0.7152 * px[p + 1] + 0.0722 * px[p + 2];
    }
  }
  return total / (PROBE * PROBE) / 255;
}

export class LuminanceProbe {
  /** `rgba8unorm`, not the canvas format, which is often BGRA. */
  private readonly target: Target;
  private readonly read: Readback;
  private bind: GPUBindGroup | null = null;
  private value = 0;
  private framesUntilRead = 0;
  private copied = false;

  constructor(
    private readonly device: GPUDevice,
    /** A blit pipeline targeting `rgba8unorm`. */
    private readonly pipe: GPURenderPipeline,
    private readonly sampler: GPUSampler,
  ) {
    this.target = new Target(
      device,
      "rgba8unorm",
      "probe",
      GPUTextureUsage.COPY_SRC,
    );
    this.read = new Readback(device, "probe-read", PROBE_ROW_BYTES * PROBE);
  }

  /** True when the probe target was reallocated. */
  resize(): boolean {
    return this.target.resize(PROBE, PROBE);
  }

  /** Points the probe blit at the finished frame texture. */
  buildBind(frameView: GPUTextureView): void {
    this.bind = this.device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: frameView },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  /** Reduces the frame periodically, unless a read is still pending. */
  record(encoder: GPUCommandEncoder, pass: FullscreenPass): void {
    if (this.framesUntilRead > 0) {
      this.framesUntilRead--;
      return;
    }
    if (!this.read.idle || !this.bind) return;
    pass(encoder, this.target.view!, this.pipe, this.bind);
    encoder.copyTextureToBuffer(
      { texture: this.target.texture! },
      {
        buffer: this.read.buffer,
        bytesPerRow: PROBE_ROW_BYTES,
        rowsPerImage: PROBE,
      },
      { width: PROBE, height: PROBE },
    );
    this.framesUntilRead = PROBE_EVERY_FRAMES - 1;
    this.copied = true;
  }

  /** Map only after an actual copy; never remap the previous frame's buffer. */
  poll(): void {
    if (!this.copied) return;
    this.copied = false;
    this.read.poll((bytes) => {
      this.value = meanLuminance(new Uint8Array(bytes));
    });
  }

  /** Mean luminance of the last completed frame, `[0, 1]`. */
  get luminance(): number {
    return this.value;
  }

  dispose(): void {
    this.target.dispose();
    this.read.dispose();
  }
}
