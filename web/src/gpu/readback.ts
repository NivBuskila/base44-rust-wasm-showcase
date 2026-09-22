/**
 * One asynchronous GPU→CPU read, with the rule that makes it safe.
 *
 * WebGPU has a trap here that is silent by design. A staging buffer being read
 * is *mapped*, and a command buffer that touches a mapped buffer — or one with a
 * map still pending — is not partially executed and does not throw: the driver
 * discards the **entire** submission. Since a frame is submitted as one command
 * buffer, a single ill-timed `copyBufferToBuffer` erases the compute pass and
 * every draw with it, so the tab shows the previous frame again. Intermittently,
 * that is precisely what a user calls flicker, and nothing appears in a `try`
 * block anywhere near it. It cost a debugging round to find through
 * `device.onuncapturederror` (see `error-log.ts`).
 *
 * The rule is therefore: never encode a copy into a buffer that is busy. That
 * used to be an unwritten convention duplicated at each call site, and one of
 * the two copies simply forgot it. Here it is one object instead: ask `idle`
 * before encoding the copy, call `poll` after the submit, and the flag that
 * links them is not something a future call site can fail to declare.
 *
 * Reads are naturally a frame or more behind; every consumer of this (the alive
 * count, the luminance probe) is documented as lagging, because stalling the
 * queue to make them current would cost far more than they are worth.
 */

export class Readback {
  /** The staging buffer: the destination of the copy, `MAP_READ` on the CPU. */
  readonly buffer: GPUBuffer;

  private pending = false;
  private destroyed = false;

  constructor(device: GPUDevice, label: string, size: number) {
    this.buffer = device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * Whether a copy into this buffer may be encoded now. False while a read is
   * in flight — skip the copy for this frame rather than submitting with it.
   */
  get idle(): boolean {
    return !this.pending && !this.destroyed;
  }

  /**
   * Starts a read of whatever was last copied in, calling `consume` with the
   * mapped bytes once they land. A no-op while a read is already in flight, so
   * it is safe to call unconditionally once per frame.
   *
   * `consume` gets the live mapped range and must not retain it; copy out what
   * it needs. Failures are swallowed: a lost device or a cancelled map is not
   * worth breaking a frame over, and the previous value simply stands.
   */
  poll(consume: (bytes: ArrayBuffer) => void): void {
    if (!this.idle) return;
    this.pending = true;
    void this.buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        consume(this.buffer.getMappedRange());
        this.buffer.unmap();
      })
      .catch(() => undefined)
      .finally(() => {
        this.pending = false;
      });
  }

  dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    // Destroying a buffer rejects any pending map, which `poll` already
    // swallows, so the in-flight read needs no separate wind-down.
    this.buffer.destroy();
  }
}
