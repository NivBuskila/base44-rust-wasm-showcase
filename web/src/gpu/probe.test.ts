import { afterEach, expect, it, vi } from 'vitest';
import { LuminanceProbe } from './probe';

afterEach(() => vi.unstubAllGlobals());

it('reads the luminance probe once per six frames, mapping only copied frames', async () => {
  vi.stubGlobal('GPUTextureUsage', { COPY_SRC: 1, RENDER_ATTACHMENT: 2, TEXTURE_BINDING: 4 });
  vi.stubGlobal('GPUBufferUsage', { COPY_DST: 1, MAP_READ: 2 });
  vi.stubGlobal('GPUMapMode', { READ: 1 });

  const mapAsync = vi.fn(() => Promise.resolve());
  const buffer = {
    mapAsync,
    getMappedRange: () => new ArrayBuffer(16 * 256),
    unmap: vi.fn(),
    destroy: vi.fn(),
  };
  const device = {
    createBuffer: () => buffer,
    createTexture: () => ({ createView: () => ({}), destroy: vi.fn() }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const pipe = { getBindGroupLayout: () => ({}) } as unknown as GPURenderPipeline;
  const probe = new LuminanceProbe(device, pipe, {} as GPUSampler);
  probe.resize();
  probe.buildBind({} as GPUTextureView);

  const copyTextureToBuffer = vi.fn();
  const encoder = { copyTextureToBuffer } as unknown as GPUCommandEncoder;
  const pass = vi.fn();
  probe.record(encoder, pass);
  probe.poll();
  await vi.waitFor(() => expect(buffer.unmap).toHaveBeenCalledTimes(1));

  for (let frame = 0; frame < 5; frame++) {
    probe.record(encoder, pass);
    probe.poll();
  }
  expect(copyTextureToBuffer).toHaveBeenCalledTimes(1);
  expect(mapAsync).toHaveBeenCalledTimes(1);

  probe.record(encoder, pass);
  probe.poll();
  await vi.waitFor(() => expect(buffer.unmap).toHaveBeenCalledTimes(2));
  expect(copyTextureToBuffer).toHaveBeenCalledTimes(2);
  expect(pass).toHaveBeenCalledTimes(2);
  probe.dispose();
});
