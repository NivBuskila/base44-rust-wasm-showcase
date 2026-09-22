/**
 * The particle pool as it lives on the GPU: its storage, its compute passes and
 * the count that comes back.
 *
 * On this backend the pool is *not* the renderer's business. The engine hands
 * ownership over (`AetherEngine.set_gpu_particles`), the Rust pool stops
 * advancing, and everything a spell did arrives as an op log that the compute
 * shader replays per particle before integrating — so this object holds the pool
 * buffer, the fluid and obstacle fields it reads, the op log, and the two entry
 * points (`seed_main` for a fresh or grown pool, `step_main` every frame).
 *
 * It was split out of `renderer.ts`, which was carrying three unrelated jobs in
 * one 1,300-line class. Keeping it separate matters beyond tidiness: the sim owns
 * resources with lifetimes the render targets do not share — the pool is
 * reallocated when the user drags the count slider, the obstacle buffer when the
 * camera resolution changes — and those reallocations must invalidate exactly the
 * bind groups that name them. That is what `poolVersion` is for: the renderer
 * still draws straight from `pool`, so it watches the version and rebuilds its
 * own draw bind group instead of guessing.
 *
 * The one hard sequencing rule is in `readback.ts`: the counter copy is encoded
 * only while its staging buffer is idle, and `pollAlive` runs after the submit.
 */

import {
  FLUID_H,
  FLUID_W,
  MAX_PARTICLE_OPS,
  MAX_PARTICLES,
  PARTICLE_OP_STRIDE,
} from "../constants";
import type { GpuSimFrame } from "../types";
import { Readback } from "./readback";
import {
  PARTICLE_BYTES,
  PARTICLE_COMPUTE_WGSL,
  SIM_UNIFORM_FLOATS,
  STEP_WORKGROUP,
} from "./shaders/particles";
import { packSimUniform } from "./sim-uniform";

/** Pool allocations are rounded up to this, so a slider drag is not a realloc. */
const PARTICLE_BUCKET = 32768;

/** Bytes of the counter block: `[spawned, alive]`. */
const COUNTER_BYTES = 8;

export class ParticleSim {
  private readonly device: GPUDevice;

  private readonly layout: GPUBindGroupLayout;
  private readonly stepPipe: GPUComputePipeline;
  private readonly seedPipe: GPUComputePipeline;

  private readonly simUniform: GPUBuffer;
  private readonly opsBuffer: GPUBuffer;
  private readonly velUBuffer: GPUBuffer;
  private readonly velVBuffer: GPUBuffer;
  private obstacleBuffer: GPUBuffer;
  private readonly counters: GPUBuffer;
  private readonly counterRead: Readback;

  private particleBuffer: GPUBuffer | null = null;
  private capacity = 0;
  private seededCount = 0;
  private version = 0;
  private simBind: GPUBindGroup | null = null;

  private readonly scratch = new ArrayBuffer(SIM_UNIFORM_FLOATS * 4);
  private readonly scratchF32 = new Float32Array(this.scratch);
  private readonly scratchU32 = new Uint32Array(this.scratch);
  private readonly zeroCounters = new Uint32Array(2);

  private respawnCredit = 0;
  private aliveCount = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    const d = device;

    const computeModule = d.createShaderModule({
      code: PARTICLE_COMPUTE_WGSL,
      label: "particles-step",
    });
    // One explicit layout for both entry points: `'auto'` would only include
    // the bindings each entry point touches, and the seed pass reads two of
    // seven, so the shared bind group would not fit it.
    const simBuffer = (
      binding: number,
      type: GPUBufferBindingType,
    ): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.layout = d.createBindGroupLayout({
      label: "particles-sim",
      entries: [
        simBuffer(0, "uniform"),
        simBuffer(1, "storage"),
        simBuffer(2, "read-only-storage"),
        simBuffer(3, "read-only-storage"),
        simBuffer(4, "read-only-storage"),
        simBuffer(5, "read-only-storage"),
        simBuffer(6, "storage"),
      ],
    });
    const pipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.layout],
    });
    this.stepPipe = d.createComputePipeline({
      label: "particles-step",
      layout: pipelineLayout,
      compute: { module: computeModule, entryPoint: "step_main" },
    });
    this.seedPipe = d.createComputePipeline({
      label: "particles-seed",
      layout: pipelineLayout,
      compute: { module: computeModule, entryPoint: "seed_main" },
    });

    this.simUniform = d.createBuffer({
      label: "sim-uniform",
      size: SIM_UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const storage = (bytes: number, label: string): GPUBuffer =>
      d.createBuffer({
        label,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    this.opsBuffer = storage(MAX_PARTICLE_OPS * PARTICLE_OP_STRIDE * 4, "ops");
    this.velUBuffer = storage(FLUID_W * FLUID_H * 4, "vel-u");
    this.velVBuffer = storage(FLUID_W * FLUID_H * 4, "vel-v");
    this.obstacleBuffer = storage(FLUID_W * FLUID_H * 4, "obstacle");
    this.counters = d.createBuffer({
      label: "counters",
      size: COUNTER_BYTES,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    this.counterRead = new Readback(d, "counters-read", COUNTER_BYTES);
  }

  /** The pool, for the renderer's instanced draw. Null until the first step. */
  get pool(): GPUBuffer | null {
    return this.particleBuffer;
  }

  /** Bumped on every pool reallocation; bind groups naming it must be rebuilt. */
  get poolVersion(): number {
    return this.version;
  }

  /** Live particles as of the last completed readback — a frame behind. */
  get alive(): number {
    return this.aliveCount;
  }

  /**
   * Encodes this frame's work: uploads the fluid, the obstacle field and the op
   * log, then one compute pass. Returns the particles that were stepped, or 0
   * when the frame carries nothing worth stepping.
   */
  step(
    encoder: GPUCommandEncoder,
    sim: GpuSimFrame,
    frameIndex: number,
  ): number {
    const active = Math.min(MAX_PARTICLES, Math.max(0, Math.floor(sim.active)));
    if (active === 0 || sim.gridW < 2 || sim.gridH < 2) return 0;
    if (!this.ensurePool(active)) return 0;

    const cells = sim.gridW * sim.gridH;
    if (sim.velU.length < cells || sim.velV.length < cells) return 0;
    this.uploadFloats(this.velUBuffer, sim.velU);
    this.uploadFloats(this.velVBuffer, sim.velV);

    const obstacleW = Math.max(1, Math.floor(sim.obstacleInfo[0] ?? 0));
    const obstacleH = Math.max(1, Math.floor(sim.obstacleInfo[1] ?? 0));
    const maxAbs = sim.obstacleInfo[2] ?? 0;
    const hasObstacle =
      sim.obstacle.length >= obstacleW * obstacleH && maxAbs >= 0.5;
    if (hasObstacle && !this.ensureObstacle(obstacleW * obstacleH * 4, active))
      return 0;
    if (hasObstacle) this.uploadFloats(this.obstacleBuffer, sim.obstacle);

    const opCount = Math.min(
      MAX_PARTICLE_OPS,
      Math.max(0, Math.floor(sim.opCount)),
    );
    if (opCount > 0) {
      this.device.queue.writeBuffer(
        this.opsBuffer,
        0,
        sim.ops.buffer,
        sim.ops.byteOffset,
        opCount * PARTICLE_OP_STRIDE * 4,
      );
    }

    const packed = packSimUniform(this.scratchF32, this.scratchU32, {
      sim,
      active,
      opCount,
      obstacleW,
      obstacleH,
      hasObstacle,
      frameIndex,
      respawnCredit: this.respawnCredit,
    });
    this.respawnCredit = packed.respawnCredit;
    this.device.queue.writeBuffer(this.simUniform, 0, this.scratch);
    this.device.queue.writeBuffer(this.counters, 0, this.zeroCounters);

    const groups = Math.ceil(active / STEP_WORKGROUP);
    const pass = encoder.beginComputePass({ label: "particles" });
    if (this.seededCount < active) {
      // A fresh or grown pool: scatter it once, exactly as `seed_uniform` does.
      pass.setPipeline(this.seedPipe);
      pass.setBindGroup(0, this.simBind!);
      pass.dispatchWorkgroups(groups);
      this.seededCount = active;
    }
    pass.setPipeline(this.stepPipe);
    pass.setBindGroup(0, this.simBind!);
    pass.dispatchWorkgroups(groups);
    pass.end();

    // Only while the staging buffer is idle — see `readback.ts` for why a copy
    // into a busy one costs the whole frame.
    if (this.counterRead.idle) {
      encoder.copyBufferToBuffer(
        this.counters,
        0,
        this.counterRead.buffer,
        0,
        COUNTER_BYTES,
      );
    }
    return active;
  }

  /** Starts the alive-count read. Call once per frame, after the submit. */
  pollAlive(): void {
    this.counterRead.poll((bytes) => {
      this.aliveCount = new Uint32Array(bytes)[1];
    });
  }

  /** (Re)allocates the pool and its bind group; false if it has no storage. */
  private ensurePool(active: number): boolean {
    const want = Math.min(MAX_PARTICLES, Math.max(1, active));
    if (want > this.capacity) {
      const capacity = Math.min(
        MAX_PARTICLES,
        Math.ceil(want / PARTICLE_BUCKET) * PARTICLE_BUCKET,
      );
      this.particleBuffer?.destroy();
      this.particleBuffer = this.device.createBuffer({
        label: "particles",
        size: capacity * PARTICLE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.capacity = capacity;
      this.seededCount = 0;
      this.version++;
      this.buildBind();
    }
    return this.particleBuffer !== null;
  }

  /**
   * Grows the obstacle buffer when the camera's mask does. The bind group names
   * the old buffer, so it is rebuilt here rather than left dangling.
   */
  private ensureObstacle(bytes: number, active: number): boolean {
    if (this.obstacleBuffer.size >= bytes) return true;
    this.obstacleBuffer.destroy();
    this.obstacleBuffer = this.device.createBuffer({
      label: "obstacle",
      size: Math.ceil(bytes / 256) * 256,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.buildBind();
    return this.particleBuffer !== null && active > 0;
  }

  private buildBind(): void {
    const pool = this.particleBuffer;
    if (!pool) return;
    this.simBind = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.simUniform } },
        { binding: 1, resource: { buffer: pool } },
        { binding: 2, resource: { buffer: this.velUBuffer } },
        { binding: 3, resource: { buffer: this.velVBuffer } },
        { binding: 4, resource: { buffer: this.obstacleBuffer } },
        { binding: 5, resource: { buffer: this.opsBuffer } },
        { binding: 6, resource: { buffer: this.counters } },
      ],
    });
  }

  /** Copies a WASM-memory view into a storage buffer, never past its size. */
  private uploadFloats(buffer: GPUBuffer, data: Float32Array): void {
    const floats = Math.min(data.length, buffer.size / 4);
    if (floats <= 0) return;
    this.device.queue.writeBuffer(
      buffer,
      0,
      data.buffer,
      data.byteOffset,
      floats * 4,
    );
  }

  dispose(): void {
    this.counterRead.dispose();
    this.particleBuffer?.destroy();
    this.particleBuffer = null;
    this.simBind = null;
  }
}
