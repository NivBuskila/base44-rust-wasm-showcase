import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Camera, CameraError } from './camera';
import { FLOW_H, FLOW_W } from './constants';

/**
 * jsdom has no 2D canvas. The camera only needs `drawImage` + `getImageData`
 * plus the mirror transform, so a recording stub is enough — and lets the
 * tests assert the transform without a pixel in sight.
 */
function stubCanvas(pixels: () => Uint8ClampedArray) {
  const ctx = {
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: pixels() })),
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
    () => ctx as unknown as CanvasRenderingContext2D,
  );
  return ctx;
}

/** Makes `video` look like it has decoded a frame. */
function giveFrame(video: HTMLVideoElement, w = 1280, h = 720) {
  Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
  Object.defineProperty(video, 'videoWidth', { configurable: true, value: w });
  Object.defineProperty(video, 'videoHeight', { configurable: true, value: h });
}

function solidFrame(r: number, g: number, b: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(FLOW_W * FLOW_H * 4);
  for (let p = 0; p < data.length; p += 4) {
    data[p] = r;
    data[p + 1] = g;
    data[p + 2] = b;
    data[p + 3] = 255;
  }
  return data;
}

function mediaDevices(getUserMedia: (c: MediaStreamConstraints) => Promise<MediaStream>) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  });
}

describe('Camera', () => {
  beforeEach(() => {
    stubCanvas(() => solidFrame(0, 0, 0));
  });
  afterEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: undefined });
  });

  it('refuses to construct without a 2D context, loudly', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    expect(() => new Camera()).toThrow(/2D canvas/);
  });

  it('reports a missing camera API as unavailable, not denied', async () => {
    const cam = new Camera();
    const err = await cam.start().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(CameraError);
    expect((err as CameraError).denied).toBe(false);
  });

  it.each([
    ['NotAllowedError', true],
    ['SecurityError', true],
    ['NotFoundError', false],
    ['NotReadableError', false],
  ])('maps a %s from getUserMedia to denied=%s', async (name, denied) => {
    mediaDevices(() => Promise.reject(new DOMException('nope', name)));
    const cam = new Camera();
    const err = (await cam.start().catch((e: unknown) => e)) as CameraError;
    expect(err).toBeInstanceOf(CameraError);
    expect(err.denied).toBe(denied);
    if (!denied) expect(err.message).toContain(name);
  });

  it('asks for the defaults as ideals, so a weaker webcam is still accepted', async () => {
    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] } as unknown as MediaStream;
    const getUserMedia = vi.fn(() => Promise.resolve(stream));
    mediaDevices(getUserMedia);

    const cam = new Camera();
    vi.spyOn(cam.video, 'play').mockResolvedValue();
    giveFrame(cam.video);
    await cam.start({ frameRate: 30 });

    expect(getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        facingMode: 'user',
      },
    });
    expect(cam.video.srcObject).toBe(stream);
    expect(cam.width).toBe(1280);
    expect(cam.height).toBe(720);

    cam.stop();
    expect(track.stop).toHaveBeenCalledOnce();
    expect(cam.video.srcObject).toBeNull();
  });

  it('returns no luma before the first frame has pixels', () => {
    const cam = new Camera();
    expect(cam.hasFrame).toBe(false);
    expect(cam.readLuma()).toBeNull();
  });

  it.each([
    ['red', [255, 0, 0], 53],
    ['green', [0, 255, 0], 182],
    ['blue', [0, 0, 255], 18],
    ['white', [255, 255, 255], 255],
    ['black', [0, 0, 0], 0],
  ])('converts a %s frame with Rec. 709 weights', (_name, [r, g, b], luma) => {
    stubCanvas(() => solidFrame(r!, g!, b!));
    const cam = new Camera();
    giveFrame(cam.video);
    const plane = cam.readLuma();
    expect(plane).not.toBeNull();
    expect(plane).toHaveLength(FLOW_W * FLOW_H);
    expect(plane![0]).toBe(luma);
    expect(plane![plane!.length - 1]).toBe(luma);
  });

  it('mirrors by default and draws straight when asked not to', () => {
    const ctx = stubCanvas(() => solidFrame(0, 0, 0));
    const cam = new Camera();
    giveFrame(cam.video);

    cam.readLuma();
    expect(ctx.translate).toHaveBeenCalledWith(FLOW_W, 0);
    expect(ctx.scale).toHaveBeenCalledWith(-1, 1);
    expect(ctx.drawImage).toHaveBeenCalledWith(cam.video, 0, 0, FLOW_W, FLOW_H);
    // The flip is scoped: whatever happens next starts from a clean transform.
    expect(ctx.save).toHaveBeenCalledOnce();
    expect(ctx.restore).toHaveBeenCalledOnce();

    ctx.translate.mockClear();
    ctx.scale.mockClear();
    cam.readLuma(false);
    expect(ctx.translate).not.toHaveBeenCalled();
    expect(ctx.scale).not.toHaveBeenCalled();
  });

  it('reuses one luma buffer across frames', () => {
    const cam = new Camera();
    giveFrame(cam.video);
    const a = cam.readLuma();
    const b = cam.readLuma();
    expect(a).toBe(b);
  });
});
