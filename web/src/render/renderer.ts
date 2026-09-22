/**
 * WebGL2 compositor.
 *
 * One frame, in order:
 *
 * 1. **scene** — treated camera behind a cubic-resampled, domain-warped dye
 *    field, written as linear radiance into an HDR target.
 * 2. **particles** — one `gl.POINTS` draw from a single streamed buffer,
 *    additive into the same target.
 * 3. **overlay** — the hand skeleton, additive, so the bloom picks it up.
 * 4. **bloom** — a 13-tap downsample chain and a 9-tap tent fold-back, five
 *    small draws, driven by `frame.intensity`.
 * 5. **composite** — chromatic aberration, ACES tone map, split tone, vignette,
 *    sRGB transfer and a triangular dither, straight to the default
 *    framebuffer.
 *
 * Invariants the rest of the app depends on:
 * - Nothing is allocated inside `render` on the steady-state path. Textures and
 *   framebuffers are (re)allocated only from `resize`, from the constructor, or
 *   when the camera's resolution changes.
 * - The default framebuffer is bound and fully written when `render` returns,
 *   so `sampleLuminance` and Playwright screenshots see a finished frame.
 * - Every intermediate target survives on either a float or an 8-bit format;
 *   the shaders are compiled against whichever one the context actually
 *   supports.
 */

import type { RenderBackend, RenderFrame, SceneRenderer } from "../types";
import { STYLES } from "./styles";
import { drawCompositePass, drawParticlePass } from "./composite-pass";
import { RendererError } from "./gl";
import { LumaProbe } from "./luma-probe";
import type { Resources } from "./resources";
import { createResources, releaseResources } from "./resources";
import { drawDebugPass, drawScenePass } from "./scene-pass";
import {
  MIN_SCENE_SCALE,
  SCENE_PIXEL_BUDGET,
  bufferSize,
  deviceScale,
  scaledSize,
  sceneScale,
  viewportScale,
} from "./sizing";

export { RendererError } from "./gl";

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

export class Renderer implements SceneRenderer {
  readonly backend: RenderBackend = "webgl2";

  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  /** Forces the 8-bit intermediate path, to exercise the fallback on demand. */
  private readonly forceSdr: boolean;

  private res: Resources | null = null;
  private video: HTMLVideoElement | null = null;
  private lostContext = false;

  private dpr = 1;
  private frameIndex = 0;
  /** Scene/bloom resolution as a fraction of the canvas; 1 unless capped. */
  private sceneScale = 1;
  /** Particle size relative to the viewport width; see `resize`. */
  private viewScale = 1;
  /** Scene pixel ceiling, tightened when the context is a CPU rasteriser. */
  private pixelBudget = SCENE_PIXEL_BUDGET;
  /** `?rscale=` override, which wins over the budget. */
  private readonly pinnedScale: number | null;
  /** Adaptive multiplier on the budgeted scale; 1 until the governor lowers it. */
  private qualityScale = 1;

  private readonly luma: LumaProbe;

  private readonly onResize = (): void => this.resize();
  private readonly onLost = (e: Event): void => {
    // Without preventDefault the browser never fires `webglcontextrestored`,
    // and a lost context becomes permanent.
    e.preventDefault();
    this.lostContext = true;
    console.warn("[aether] WebGL context lost; waiting for restore");
  };
  private readonly onRestored = (): void => {
    // Every GL object died with the context, so the wrappers holding them are
    // stale: rebuild rather than reuse, and re-probe the formats because a
    // restored context can be a different (often software) implementation.
    this.release();
    try {
      this.res = createResources(this.gl, this.forceSdr);
      this.pixelBudget = this.res.pixelBudget;
      this.lostContext = false;
      console.info("[aether] WebGL context restored");
    } catch (err) {
      console.error(
        "[aether] could not rebuild the renderer after context restore",
        err,
      );
    }
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      // Needed so `sampleLuminance` and Playwright screenshots can read the
      // framebuffer after the frame is drawn.
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    if (!gl) throw new RendererError("WebGL2 is unavailable in this browser.");
    this.gl = gl;
    // Two diagnostic overrides, both off in normal use: `?sdr8` forces the
    // 8-bit intermediate path and `?rscale=` pins the internal resolution, so
    // the fallbacks can be looked at without the hardware that triggers them.
    const query =
      typeof location !== "undefined"
        ? new URLSearchParams(location.search)
        : null;
    this.forceSdr = query?.has("sdr8") ?? false;
    const pinned = Number(query?.get("rscale") ?? NaN);
    this.pinnedScale =
      Number.isFinite(pinned) && pinned > 0 ? clamp(pinned, 0.25, 1) : null;

    canvas.addEventListener("webglcontextlost", this.onLost);
    canvas.addEventListener("webglcontextrestored", this.onRestored);
    window.addEventListener("resize", this.onResize);

    this.luma = new LumaProbe(gl, canvas);
    this.res = createResources(gl, this.forceSdr);
    this.pixelBudget = this.res.pixelBudget;
    this.resize();
  }

  // ------------------------------------------------------------- resources

  private release(): void {
    const res = this.res;
    if (!res) return;
    this.res = null;
    releaseResources(this.gl, res);
  }

  /** Releases every GL object this renderer owns and detaches its listeners. */
  dispose(): void {
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    window.removeEventListener("resize", this.onResize);
    this.release();
    this.video = null;
    this.lostContext = true;
  }

  // ------------------------------------------------------------------ size

  /**
   * Sets the adaptive resolution multiplier and re-sizes the scene targets.
   *
   * Separate from `pixelBudget`, which is a static property of the context:
   * this one tracks how the frame rate is actually doing.
   */
  setQualityScale(scale: number): void {
    const next = clamp(scale, MIN_SCENE_SCALE, 1);
    if (next === this.qualityScale) return;
    this.qualityScale = next;
    this.resize();
  }

  /** Matches the drawing buffer and every target to the CSS size and DPR. */
  resize(): void {
    this.dpr = deviceScale(window.devicePixelRatio);
    const [w, h] = bufferSize(
      this.canvas.clientWidth,
      this.canvas.clientHeight,
      this.dpr,
    );
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    this.viewScale = viewportScale(this.canvas.clientWidth);

    const res = this.res;
    if (!res) return;

    this.sceneScale = sceneScale(w, h, {
      budget: this.pixelBudget,
      quality: this.qualityScale,
      pinned: this.pinnedScale,
    });
    const [sw, sh] = scaledSize(w, h, this.sceneScale);
    res.scene.resize(sw, sh);
    res.bloom.resize(sw, sh);
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    this.res?.sources.resetVideo();
  }

  get hasVideo(): boolean {
    return this.video !== null;
  }

  // ----------------------------------------------------------------- frame

  render(frame: RenderFrame): void {
    const gl = this.gl;
    if (this.lostContext || gl.isContextLost()) return;
    const res = this.res;
    if (!res) return;

    this.resize();
    this.frameIndex = (this.frameIndex + 1) % 1024;
    const intensity = Number.isFinite(frame.intensity)
      ? clamp(frame.intensity, 0, 1)
      : 0;
    const time = Number.isFinite(frame.time) ? frame.time : 0;

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(res.quadVao);

    if (frame.mode === "debug") {
      drawDebugPass(
        gl,
        res.programs,
        res.sources,
        frame,
        this.canvas.width,
        this.canvas.height,
      );
      return;
    }

    const style = STYLES[frame.mode];
    drawScenePass(
      gl,
      res.programs,
      res.sources,
      res.scene,
      frame,
      style,
      intensity,
      time,
      this.canvas.width,
      this.canvas.height,
      this.video,
    );
    drawParticlePass(
      res,
      frame,
      style,
      intensity,
      this.dpr * this.sceneScale * this.viewScale,
    );
    if (style.overlay > 0 && frame.hands)
      res.overlay.draw(
        res.programs.overlay,
        res.scene,
        frame.hands,
        style.overlay,
        this.dpr * this.sceneScale,
      );
    // The overlay left its own VAO bound; the ladder draws fullscreen strips.
    gl.bindVertexArray(res.quadVao);
    res.bloom.record(res.scene, style);
    drawCompositePass(
      gl,
      res,
      frame,
      style,
      intensity,
      this.frameIndex,
      this.canvas.width,
      this.canvas.height,
    );
  }

  // ------------------------------------------------------------ inspection

  /** Mean luminance over a grid of samples across the rendered frame, `[0, 1]`. */
  sampleLuminance(): number {
    const gl = this.gl;
    if (this.lostContext || gl.isContextLost()) return 0;
    return this.luma.sample();
  }
}
