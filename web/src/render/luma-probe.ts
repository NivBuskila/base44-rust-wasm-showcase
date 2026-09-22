/**
 * Luminance probe for the WebGL2 path: the synchronous counterpart of
 * `gpu/probe.ts`.
 *
 * Used by the headless tests to tell "something is being drawn" from "the
 * render path is dead". It reads an NxN grid of samples spread *across* the
 * frame rather than a contiguous block, so a bright region anywhere registers —
 * a single pixel is not enough, because the fluid is sparse and the centre
 * pixel is legitimately black much of the time. One `readPixels` per sampled
 * row: `n` calls instead of `n^2`.
 */

const LUMA_SAMPLE_GRID = 16;

export class LumaProbe {
  /** One-row scratch, grown to the canvas width. */
  private row = new Uint8Array(4);

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly canvas: HTMLCanvasElement,
  ) {}

  /** Mean Rec.709 luminance of the default framebuffer, `[0, 1]`. */
  sample(): number {
    const gl = this.gl;
    const { width, height } = this.canvas;
    const n = LUMA_SAMPLE_GRID;
    const stepX = Math.max(1, Math.floor(width / n));
    const stepY = Math.max(1, Math.floor(height / n));

    const rowBytes = width * 4;
    if (this.row.length < rowBytes) this.row = new Uint8Array(rowBytes);

    // The composite pass leaves the default framebuffer bound, but a caller
    // could read between passes, and readPixels would then sample an
    // intermediate target at the wrong size.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let total = 0;
    let count = 0;
    for (let row = 0; row < n; row++) {
      const y = Math.min(height - 1, row * stepY);
      gl.readPixels(0, y, width, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.row);
      for (let i = 0; i < n; i++) {
        const p = Math.min(width - 1, i * stepX) * 4;
        total += 0.2126 * this.row[p] + 0.7152 * this.row[p + 1] + 0.0722 * this.row[p + 2];
        count++;
      }
    }
    return count === 0 ? 0 : total / count / 255;
  }
}
