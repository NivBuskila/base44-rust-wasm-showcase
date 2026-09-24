/**
 * Typed-array views over WASM linear memory.
 *
 * Views detach whenever the memory grows, and the pointers themselves move when
 * the engine reallocates, so every consumer goes through this object and calls
 * `rebuild` after any reallocating engine call. `refresh` is the per-frame check
 * for a buffer swapped out from under us.
 */

import type { AetherEngine } from './wasm/aether';
import { FLUID_H, FLUID_W, PARTICLE_OP_STRIDE, PARTICLE_STRIDE } from './constants';
import type { GpuSimFrame } from './types';

export class EngineViews {
  buffer!: ArrayBufferLike;
  dye!: Uint8Array;
  luma!: Uint8Array;
  mask!: Float32Array;

  constructor(
    private readonly engine: AetherEngine,
    private readonly memory: WebAssembly.Memory,
  ) {
    this.rebuild();
  }

  rebuild(): void {
    const buffer = this.memory.buffer;
    this.buffer = buffer;
    this.dye = new Uint8Array(buffer, this.engine.dye_ptr(), this.engine.dye_len());
    this.luma = new Uint8Array(buffer, this.engine.luma_ptr(), this.engine.luma_len());
    this.mask = new Float32Array(buffer, this.engine.mask_ptr(), this.engine.mask_capacity());
  }

  /** Rebuilds only when the memory grew or a view detached. */
  refresh(): void {
    if (this.buffer !== this.memory.buffer || this.dye.length === 0) this.rebuild();
  }

  /** A fresh particle view; its length tracks the live particle count. */
  particles(): Float32Array {
    // `step` can grow memory after the frame's first `refresh`.
    this.refresh();
    return new Float32Array(
      this.buffer,
      this.engine.particle_ptr(),
      this.engine.particle_count() * PARTICLE_STRIDE,
    );
  }

  debug(): Uint8Array {
    // debug_ptr() re-encodes the texture, so the view is built after the call.
    const ptr = this.engine.debug_ptr();
    return new Uint8Array(this.memory.buffer, ptr, this.engine.dye_len());
  }

  /**
   * Fluid, obstacle and op log for the GPU pool, as views over WASM memory.
   *
   * Rebuilt every frame on purpose: the pointers move whenever the engine
   * reallocates, and a retained view would silently read freed memory.
   */
  gpuSim(): GpuSimFrame {
    this.refresh();
    const buffer = this.buffer;
    const cells = FLUID_W * FLUID_H;
    const info = this.engine.obstacle_info();
    const obstacleCells = Math.max(1, Math.floor(info[0])) * Math.max(1, Math.floor(info[1]));
    return {
      velU: new Float32Array(buffer, this.engine.velocity_u_ptr(), cells),
      velV: new Float32Array(buffer, this.engine.velocity_v_ptr(), cells),
      gridW: FLUID_W,
      gridH: FLUID_H,
      obstacle: new Float32Array(buffer, this.engine.obstacle_ptr(), obstacleCells),
      obstacleInfo: info,
      ops: new Float32Array(
        buffer,
        this.engine.particle_ops_ptr(),
        this.engine.particle_ops_len() * PARTICLE_OP_STRIDE,
      ),
      opCount: this.engine.particle_ops_len(),
      params: this.engine.particle_step_params(),
      active: this.engine.particle_count(),
    };
  }
}
