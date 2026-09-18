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
export type ViewMode = 'aether' | 'camera' | 'debug' | 'particles';

/** Everything the renderer needs for one frame. */
export interface RenderFrame {
  /** RGBA8 dye texture, `FLUID_W * FLUID_H * 4` bytes. */
  dye: Uint8Array;
  /** Interleaved `x, y, heat, life`, positions normalised to `[0, 1]`. */
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
}

/** Live numbers for the HUD. */
export interface HudStats {
  fps: number;
  stepMs: number;
  renderMs: number;
  inferenceMs: number;
  stats: Float32Array;
  spells: [string, string];
  perception: PerceptionStatus;
}

/** Controls the HUD exposes back to the app. */
export interface HudCallbacks {
  onParam(key: string, value: number): void;
  onParticleCount(n: number): void;
  onOverdrive(on: boolean): void;
  onViewMode(mode: ViewMode): void;
  onToggleCamera(on: boolean): void;
  onReset(): void;
}
