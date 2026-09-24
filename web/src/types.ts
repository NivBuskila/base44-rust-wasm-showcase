/**
 * The interfaces the browser modules implement against. Everything crossing a
 * module boundary is declared here so `perception.ts`, `render/` and `main.ts`
 * can be written independently.
 */

/** A segmentation mask at the model's own resolution. */
export interface MaskFrame {
  /** Confidence per pixel, `[0, 1]`, row-major, `width * height` long. */
  data: Float32Array;
  width: number;
  height: number;
}

/** One perception result, packed in the layouts `constants.ts` documents. */
export interface PerceptionFrame {
  /** `HAND_BUFFER` floats. Slot 0 and 1 are independent hands. */
  hands: Float32Array;
  /** `POSE_STRIDE` floats. */
  pose: Float32Array;
  /** Body silhouette, or null when the model produced none this frame. */
  mask: MaskFrame | null;
  /** Wall-clock milliseconds the inference took. */
  latencyMs: number;
  /** True when at least one hand was detected. */
  anyHand: boolean;
  /** Worker path only: asynchronous bitmap decode time (not render-loop work). */
  decodeMs?: number;
  /** Worker path only: time from posting the bitmap to receiving the result. */
  workerRoundTripMs?: number;
}

/** Why perception is not running, for the HUD to explain to the user. */
export type PerceptionStatus =
  | { kind: 'loading' }
  | { kind: 'ready'; delegate: 'GPU' | 'CPU' }
  | { kind: 'unavailable'; reason: string };

/**
 * Landmark + segmentation source. The real implementation wraps MediaPipe;
 * the headless test swaps in a scripted one, which is why this is an interface
 * rather than a concrete class.
 */
export interface PerceptionSource {
  init(): Promise<void>;
  /**
   * Runs inference on the current video frame. Returns null when the source is
   * not ready or the frame timestamp has not advanced.
   *
   * Must be synchronous and non-throwing: it is called from the render loop,
   * and a rejected promise there would stall the whole animation.
   */
  process(video: HTMLVideoElement, timestampMs: number): PerceptionFrame | null;
  readonly status: PerceptionStatus;
  close(): void;
}

/** What the user is looking at. */
export type ViewMode = 'aether' | 'camera' | 'blend' | 'debug' | 'particles';

/** Which graphics API is drawing the frame. */
export type RenderBackend = 'webgl2' | 'webgpu';

/**
 * The renderer contract `main.ts` drives. Two implementations: the WebGL2
 * compositor in `render/`, which draws the particle buffer the Rust engine
 * streams out, and the WebGPU one in `gpu/`, which also *simulates* the
 * particles in a compute pass and draws them from its own storage buffer.
 */
export interface SceneRenderer {
  readonly backend: RenderBackend;
  render(frame: RenderFrame): void;
  setVideo(video: HTMLVideoElement | null): void;
  /** Adaptive internal resolution multiplier, `[MIN_SCENE_SCALE, 1]`. */
  setQualityScale(scale: number): void;
  /** Mean luminance of the last frame in `[0, 1]`; may lag a frame on async backends. */
  sampleLuminance(): number;
  /**
   * Particles the backend measured alive last frame. Only the WebGPU backend
   * owns the pool, so only it answers; the count lags a frame because the
   * readback is asynchronous.
   */
  particlesAlive?(): number;
  dispose(): void;
}

/**
 * What the WebGPU backend needs to step the pool itself: the fluid the engine
 * just solved, the obstacle the particles collide with, and the op log of
 * everything the spells did this frame. All are zero-copy views over WASM
 * memory, so they are read during `render` and never retained.
 */
export interface GpuSimFrame {
  /** Fluid velocity planes, each `gridW * gridH` floats, grid cells per second. */
  velU: Float32Array;
  velV: Float32Array;
  gridW: number;
  gridH: number;
  /** Obstacle field and its `[w, h, maxAbs]` description. */
  obstacle: Float32Array;
  obstacleInfo: Float32Array;
  /** `PARTICLE_OP_STRIDE` floats per record, `opCount` records. */
  ops: Float32Array;
  opCount: number;
  /** `[simDt, life, drag, spawnRate, damping, curlInfluence, gravity]`. */
  params: Float32Array;
  /** Particles the engine wants simulated. */
  active: number;
}

/** Everything the renderer needs for one frame. */
export interface RenderFrame {
  /** RGBA8 dye texture, `FLUID_W * FLUID_H * 4` bytes. */
  dye: Uint8Array;
  /**
   * Interleaved `x, y, heat, life`, positions normalised to `[0, 1]`. Read by
   * the WebGL2 backend only; the WebGPU backend owns its particles.
   */
  particles: Float32Array;
  particleCount: number;
  /** RGBA8 obstacle/flow texture; only read in `debug` view. */
  debug: Uint8Array | null;
  /** Camera element to composite behind the simulation, if available. */
  video: HTMLVideoElement | null;
  /** Packed hand buffer for the landmark overlay, or null to hide it. */
  hands: Float32Array | null;
  /** Seconds since boot, for shader-side animation. */
  time: number;
  /**
   * Global visual intensity in `[0, 1]`, driven by measured motion energy.
   * The renderer uses it to scale bloom and exposure so a still frame is calm
   * and a burst of movement blazes.
   */
  intensity: number;
  mode: ViewMode;
  /** Draw the camera feed behind the fluid. */
  showCamera: boolean;
  /**
   * Lens-rush staging from the engine: `[x, y, progress, power]`, the origin in
   * the engine's normalised view space. `power === 0` means nothing is in
   * flight, and the renderer then skips the whole effect.
   */
  rush: Float32Array | null;
  /**
   * Fluid, obstacle and op log for a backend that simulates the particles
   * itself. Null on the WebGL2 path, which reads `particles` instead.
   */
  sim?: GpuSimFrame | null;
}

/** Live numbers for the HUD. */
export interface HudStats {
  fps: number;
  stepMs: number;
  renderMs: number;
  inferenceMs: number;
  stats: Float32Array;
  spells: [string, string];
  /** Gesture-sequence spellbook from the engine, `name:step,step;...`. */
  comboBook?: string;
  /** Packed sequence progress: combo, steps matched, charge, fired. */
  comboProgress?: Float32Array;
  /** Two-hand duet spellbook from the engine, `name:step,step;...`. */
  duetBook?: string;
  /** Packed duet progress: active duet, charge, fired. */
  duetProgress?: Float32Array;
  /** Packed hand buffer, for the tutorial's live pose mirror; null when no hand. */
  hands?: Float32Array | null;
  perception: PerceptionStatus;
}

/** Controls the HUD exposes back to the app. */
export interface HudCallbacks {
  onParam(key: string, value: number): void;
  onParticleCount(n: number): void;
  onOverdrive(on: boolean): void;
  onViewMode(mode: ViewMode): void;
  onToggleCamera(on: boolean): void;
  /**
   * Tutorial mode: single gestures still cast, but sequences and duets are held
   * back so learning one pose after another cannot chain into a combo.
   */
  onPractice(on: boolean): void;
  onReset(): void;
}
