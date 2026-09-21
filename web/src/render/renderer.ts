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

import { FLUID_H, FLUID_W, MAX_PARTICLES, PARTICLE_STRIDE } from "../constants";
import type { RenderBackend, RenderFrame, SceneRenderer } from "../types";
import type { ModeStyle } from "./styles";
import { STYLES } from "./styles";
import { BloomChain } from "./bloom";
import {
  RenderTarget,
  RendererError,
  isSoftwareRasteriser,
  probeHdr,
} from "./gl";
import { LumaProbe } from "./luma-probe";
import { HandOverlay } from "./overlay";
import type { ScenePrograms } from "./programs";
import { createPrograms, disposePrograms } from "./programs";
import type { ShaderEnv } from "./shaders/common";
import { SceneSources } from "./sources";

export { RendererError } from "./gl";

/**
 * Device pixel ratio ceiling. Beyond 2 the fill cost buys nothing visible for a
 * field this soft, and it halves the frame rate on phones.
 */
const MAX_DPR = 2;

/**
 * Ceiling on the scene and bloom pixel count, per frame.
 *
 * The chain is fill-bound: roughly twenty texture fetches per output pixel.
 * That is nothing at 1080p on a GPU and ruinous at 5K on a retina panel with
 * the DPR cap doubling it again, so the scene and the bloom are drawn at
 * whatever fraction keeps them under this budget and the composite scales them
 * back up. The composite itself always runs at full canvas resolution, so the
 * grade, the vignette and the dither stay per-pixel and the result does not
 * read as an upscale.
 *
 * 5.2 Mpx leaves 1440p at DPR 2 untouched and pulls 5K retina back to ~0.6.
 * A CPU rasteriser gets a much tighter budget — but one still above 1280x720,
 * because that is the headless test resolution and the screenshots taken there
 * are a deliverable, not just a smoke check.
 */
const SCENE_PIXEL_BUDGET = 5_200_000;
const SOFTWARE_PIXEL_BUDGET = 1_600_000;

/** Floor on the internal scale; below this the softening is obvious. */
const MIN_SCENE_SCALE = 0.4;

/** Viewport width, in CSS pixels, that particle size is calibrated against. */
const PARTICLE_SIZE_REF_WIDTH = 1200;

/**
 * Particle buffer sizes are rounded up to this many particles.
 *
 * The buffer is re-specified every frame to orphan it, and a size that changed
 * every frame would make the driver allocate a differently sized block each
 * time. Bucketing keeps the allocation stable across frames while still
 * tracking a pool the user shrank from 220k to 2k.
 */
const PARTICLE_BUCKET = 32768;

/** Everything that dies with the GL context and is rebuilt on restore. */
interface Resources {
  scene: RenderTarget;
  bloom: BloomChain;
  /** The input textures and their uploads; see `sources.ts`. */
  sources: SceneSources;
  /** Every linked program; see `programs.ts`. */
  programs: ScenePrograms;
  particleBuffer: WebGLBuffer;
  particleVao: WebGLVertexArrayObject;
  /** The hand skeleton and its buffers; see `overlay.ts`. */
  overlay: HandOverlay;
  /** Empty VAO for the fullscreen passes, which fetch no attributes. */
  quadVao: WebGLVertexArrayObject;
}

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
    this.releaseResources();
    try {
      this.res = this.createResources();
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
    this.res = this.createResources();
    this.resize();
  }

  // ------------------------------------------------------------- resources

  private createResources(): Resources {
    const gl = this.gl;
    const caps = probeHdr(gl, this.forceSdr);
    const env: ShaderEnv = { float: caps.float, range: caps.range };
    if (!caps.float) {
      console.info(
        "[aether] float render targets unavailable; HDR chain on 8-bit targets",
      );
    }
    // Probed here rather than in the constructor because a restored context is
    // often a software one: a GPU reset commonly falls back to SwiftShader.
    this.pixelBudget = isSoftwareRasteriser(gl)
      ? SOFTWARE_PIXEL_BUDGET
      : SCENE_PIXEL_BUDGET;

    const scene = new RenderTarget(gl, caps.format);
    const bloom = new BloomChain(gl, env, caps.format);

    const particleBuffer = gl.createBuffer();
    const particleVao = gl.createVertexArray();
    const quadVao = gl.createVertexArray();
    if (!particleBuffer || !particleVao || !quadVao) {
      throw new RendererError("could not allocate the vertex buffers");
    }

    gl.bindVertexArray(particleVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      PARTICLE_BUCKET * PARTICLE_STRIDE * 4,
      gl.DYNAMIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, PARTICLE_STRIDE * 4, 0);

    gl.bindVertexArray(null);

    return {
      scene,
      bloom,
      sources: new SceneSources(gl),
      programs: createPrograms(gl, env),
      particleBuffer,
      particleVao,
      overlay: new HandOverlay(gl),
      quadVao,
    };
  }

  private releaseResources(): void {
    const res = this.res;
    if (!res) return;
    const gl = this.gl;
    this.res = null;
    res.scene.dispose();
    res.bloom.dispose();
    res.sources.dispose();
    gl.deleteBuffer(res.particleBuffer);
    gl.deleteVertexArray(res.particleVao);
    res.overlay.dispose();
    gl.deleteVertexArray(res.quadVao);
    disposePrograms(res.programs);
  }

  /** Releases every GL object this renderer owns and detaches its listeners. */
  dispose(): void {
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    window.removeEventListener("resize", this.onResize);
    this.releaseResources();
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
    // Ceiling only, no floor. `devicePixelRatio` drops below 1 whenever the
    // page is zoomed out (0.5 at 50% zoom), and flooring it at 1 there
    // allocates four times the pixels the browser asked for — the opposite of
    // what the user requested — and breaks the drawing-buffer size the app's
    // own DPR test asserts. `|| 1` still covers a 0 or undefined ratio.
    this.dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * this.dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    // A dot sized in CSS pixels is a dot three times larger *relative to the
    // frame* on a 390-wide phone than on a desktop window, which is why the
    // dust reads as coarse confetti there. Size follows the viewport instead,
    // normalised to a typical desktop width, so the grain looks the same
    // fraction of the picture on every screen.
    this.viewScale = clamp(
      this.canvas.clientWidth / PARTICLE_SIZE_REF_WIDTH,
      0.45,
      1,
    );

    const res = this.res;
    if (!res) return;

    const pixels = w * h;
    const budgeted =
      pixels > this.pixelBudget ? Math.sqrt(this.pixelBudget / pixels) : 1;
    // `?rscale=` is a diagnostic pin and outranks both the budget and the
    // governor, so someone comparing resolutions gets the one they asked for.
    this.sceneScale = clamp(
      this.pinnedScale ?? budgeted * this.qualityScale,
      MIN_SCENE_SCALE,
      1,
    );
    const sw = Math.max(1, Math.round(w * this.sceneScale));
    const sh = Math.max(1, Math.round(h * this.sceneScale));
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
      this.drawDebug(res, frame);
      return;
    }

    const style = STYLES[frame.mode];
    this.drawScene(res, frame, style, intensity, time);
    this.drawParticles(res, frame, style, intensity);
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
    this.drawComposite(res, frame, style, intensity);
  }

  /** Raw obstacle/flow texture, straight to the screen with no grading. */
  private drawDebug(res: Resources, frame: RenderFrame): void {
    const gl = this.gl;
    const source =
      frame.debug && res.sources.uploadGrid(res.sources.debugTex, frame.debug)
        ? res.sources.debugTex
        : null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    if (!source) {
      // `frame.debug` is null whenever the engine has nothing to show. Say so
      // with a flat field rather than leaving the last frame on screen.
      gl.clearColor(0.03, 0.035, 0.05, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    res.programs.blit.use();
    res.programs.blit.tex("u_src", 0, source);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private drawScene(
    res: Resources,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
    time: number,
  ): void {
    const gl = this.gl;
    const p = res.programs.scene;

    res.sources.uploadGrid(res.sources.dyeTex, frame.dye);
    // The camera mode is a stronger statement than the camera toggle: the user
    // asked to look at the feed, so `showCamera` only gates the aether view.
    //
    // `camUsed` is not a style choice, it is the identity: `particles` tints
    // both the body and the rim to black, so running the layer there costs a
    // full-frame `texSubImage2D` plus five fetches per pixel to add exactly
    // zero. On a software rasteriser that was a quarter of the frame.
    const camUsed =
      style.camTint[0] + style.camTint[1] + style.camTint[2] > 0 ||
      style.camEdge[0] + style.camEdge[1] + style.camEdge[2] > 0;
    const wantCamera =
      camUsed &&
      (frame.mode === "camera" || frame.mode === "blend" || frame.showCamera);
    const hasVideo =
      wantCamera && res.sources.uploadVideo(frame.video ?? this.video);

    res.scene.bind();
    p.use();
    p.tex("u_dye", 0, res.sources.dyeTex);
    p.tex("u_video", 1, res.sources.videoTex);
    p.tex("u_noise", 2, res.sources.noiseTex);
    p.f2("u_dyeSize", FLUID_W, FLUID_H);
    p.f2("u_dyeTexel", 1 / FLUID_W, 1 / FLUID_H);
    p.f1("u_dyeAmount", style.dye);
    p.f1("u_bgAmount", style.bg);
    p.f1("u_intensity", intensity);
    p.f1("u_time", time);
    p.f1("u_hasVideo", hasVideo ? 1 : 0);
    p.f3("u_camTint", style.camTint[0], style.camTint[1], style.camTint[2]);
    p.f3("u_camEdge", style.camEdge[0], style.camEdge[1], style.camEdge[2]);
    p.f1("u_camToe", style.camToe);
    p.f1("u_camRaw", style.camRaw);
    p.f2(
      "u_videoTexel",
      1 / Math.max(1, res.sources.videoTexW),
      1 / Math.max(1, res.sources.videoTexH),
    );

    // Cover fit: crop the long axis so the feed fills the canvas at its own
    // aspect ratio. Letterboxing would put black bars inside the simulation,
    // and stretching would make gestures land off their landmarks.
    const canvasAspect = this.canvas.width / Math.max(1, this.canvas.height);
    const videoAspect = res.sources.videoAspect;
    if (videoAspect > canvasAspect) {
      p.f2("u_camScale", canvasAspect / videoAspect, 1);
    } else {
      p.f2("u_camScale", 1, videoAspect / canvasAspect);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private drawParticles(
    res: Resources,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
  ): void {
    if (style.particles <= 0) return;
    const gl = this.gl;
    const available = Math.floor(frame.particles.length / PARTICLE_STRIDE);
    const requested = Number.isFinite(frame.particleCount)
      ? Math.floor(frame.particleCount)
      : 0;
    const count = Math.min(Math.max(0, requested), available, MAX_PARTICLES);
    if (count === 0) return;

    gl.bindVertexArray(res.particleVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, res.particleBuffer);
    // Orphan, then refill. Re-specifying the whole buffer tells the driver the
    // old contents are dead, so this upload never blocks on the GPU still
    // reading last frame's data — at 220k particles that stall is a dropped
    // frame every frame.
    const bucket = Math.min(
      MAX_PARTICLES,
      Math.ceil(count / PARTICLE_BUCKET) * PARTICLE_BUCKET,
    );
    gl.bufferData(
      gl.ARRAY_BUFFER,
      bucket * PARTICLE_STRIDE * 4,
      gl.DYNAMIC_DRAW,
    );
    gl.bufferSubData(
      gl.ARRAY_BUFFER,
      0,
      frame.particles,
      0,
      count * PARTICLE_STRIDE,
    );

    const p = res.programs.particles;
    p.use();
    // Energy normalisation: the pool is user-adjustable from 2k to 220k, and
    // without this the 220k setting is a white screen and the 2k setting is
    // nearly invisible. Total emitted light stays roughly constant instead.
    // The floor sits low enough that the 1M overdrive pool reads as dense
    // dust rather than a saturated sheet.
    p.f1("u_gain", style.particles * clamp(90_000 / count, 0.1, 2.0));
    // Point size follows the target, not the canvas: the scene can be drawn
    // below canvas resolution, and a size in canvas pixels would then make the
    // dust swell into blobs when the composite scales it back up.
    const px = this.dpr * this.sceneScale * this.viewScale;
    p.f1("u_size", px * (3.1 + (1.35 - 3.1) * clamp(count / 200_000, 0, 1)));
    p.f1("u_intensity", intensity);

    res.scene.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);
  }

  private drawComposite(
    res: Resources,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    const p = res.programs.composite;
    p.use();
    p.tex("u_scene", 0, res.scene.texture);
    p.tex("u_bloom", 1, res.bloom.output);
    p.tex("u_noise", 2, res.sources.noiseTex);
    // Motion drives the glow, not the exposure: pushing exposure with movement
    // makes the whole frame pump, while pushing bloom makes the bright parts
    // bloom harder, which is what "a burst of movement blazes" should feel like.
    p.f1("u_bloomAmount", style.bloom * (0.72 + 0.75 * intensity));
    // The camera view is ungraded and must not breathe with motion.
    p.f1(
      "u_exposure",
      style.exposure * (style.grade > 0 ? 0.96 + 0.22 * intensity : 1),
    );
    p.f1("u_aberration", style.aberration);
    p.f1("u_vignette", style.vignette);
    p.f1("u_grade", style.grade);
    p.f1("u_frame", this.frameIndex);
    // The engine's origin is in the dye grid's convention (y down); this pass
    // samples the scene target, which holds that image flipped, so the y has to
    // be flipped with it or the rush would dolly in on the mirror of the palm.
    const rush = frame.rush;
    const power =
      rush && rush.length >= 4 && Number.isFinite(rush[3])
        ? Math.max(0, rush[3])
        : 0;
    p.f2(
      "u_rushAt",
      power > 0 ? rush![0] : 0.5,
      power > 0 ? 1 - rush![1] : 0.5,
    );
    p.f1("u_rushProgress", power > 0 ? rush![2] : 0);
    p.f1("u_rushPower", power);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ------------------------------------------------------------ inspection

  /** Mean luminance over a grid of samples across the rendered frame, `[0, 1]`. */
  sampleLuminance(): number {
    const gl = this.gl;
    if (this.lostContext || gl.isContextLost()) return 0;
    return this.luma.sample();
  }
}
