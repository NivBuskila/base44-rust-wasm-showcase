/// <reference lib="webworker" />

/**
 * Perception worker: MediaPipe inference, off the main thread.
 *
 * `tasks-vision` is synchronous, so running it where the render loop lives
 * means the fluid and the particles stop dead for the length of a pass — about
 * 26 ms on a laptop with both models and the segmentation readback, which is
 * most of a 60 Hz frame and the reason the app was stuck near 17 fps. Nothing
 * about the inference needs the main thread: it takes pixels and returns
 * numbers.
 *
 * So `MediaPipePerception` is hosted here unchanged and driven by
 * `ImageBitmap`s the main thread decodes (cheap) and transfers (zero-copy). The
 * main thread's share of perception drops to that decode plus a buffer swap.
 *
 * The trade is one frame of latency: a result describes the camera frame that
 * was submitted, not the one being rendered. At 30 Hz inference that is ~33 ms,
 * well inside the gesture smoothing the engine already applies, and the frames
 * gained are worth more than the staleness saved.
 */

// Side-effect import, and it must come first: MediaPipe's runtime cannot load
// in a module worker without it. See the file for why.
import './worker-import-scripts';

import { MediaPipePerception } from './perception';
import type { FromWorker, ToWorker } from './perception-protocol';

/**
 * The DOM and WebWorker libs both define `self`, and the DOM's `Window` wins in
 * a project that includes both — which loses the `postMessage` overload taking a
 * transfer list. Narrowing it once here keeps every call site honest.
 */
const ctx = self as unknown as DedicatedWorkerGlobalScope;

// No rationing here: the limiter protects a render loop this worker does not
// have, and its idle gaps are pure gesture latency. See the field's docs.
const perception = new MediaPipePerception({ rationInference: false });

/** Buffer the mask is copied into, recycled between frames by the main thread. */
let maskBuf: ArrayBuffer | null = null;

function post(message: FromWorker, transfer: Transferable[] = []): void {
  ctx.postMessage(message, transfer);
}

ctx.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;

  if (message.type === 'init') {
    try {
      await perception.init();
    } catch {
      // `status` already carries the reason; the throw is for callers that
      // await init directly, which the main thread deliberately does not.
    }
    post({ type: 'status', status: perception.status });
    return;
  }

  if (message.type === 'close') {
    perception.close();
    ctx.close();
    return;
  }

  const { bitmap, timestampMs } = message;
  if (message.maskBuf) maskBuf = message.maskBuf;

  let frame: ReturnType<MediaPipePerception['processSource']> = null;
  try {
    frame = perception.processSource(bitmap, timestampMs);
  } finally {
    // The bitmap holds decoded pixels — GPU memory on most browsers — and the
    // main thread transferred ownership, so this side is the only one that can
    // free it. Skipping it on the limiter path leaks a frame every time.
    bitmap.close();
  }

  if (!frame) {
    const spare = maskBuf;
    maskBuf = null;
    post({ type: 'skip', maskBuf: spare }, spare ? [spare] : []);
    return;
  }

  // Copies rather than transfers: these are the class's own persistent buffers,
  // and transferring them would detach the arrays the next inference writes to.
  const hands = frame.hands.slice();
  const pose = frame.pose.slice();

  let mask: Float32Array | null = null;
  let maskWidth = 0;
  let maskHeight = 0;
  if (frame.mask) {
    const { data, width, height } = frame.mask;
    const needed = width * height;
    if (!maskBuf || maskBuf.byteLength < needed * 4) maskBuf = new ArrayBuffer(needed * 4);
    mask = new Float32Array(maskBuf, 0, needed);
    mask.set(data.subarray(0, needed));
    maskWidth = width;
    maskHeight = height;
  }

  const buffers = mask ? maskBuf : null;
  maskBuf = null;

  post(
    {
      type: 'result',
      hands: hands.buffer,
      pose: pose.buffer,
      mask: buffers,
      maskWidth,
      maskHeight,
      latencyMs: frame.latencyMs,
      anyHand: frame.anyHand,
    },
    buffers ? [hands.buffer, pose.buffer, buffers] : [hands.buffer, pose.buffer],
  );

  // Perception's own status can change mid-session (it gives up after repeated
  // delegate failures), and the main thread has no other way to learn that.
  post({ type: 'status', status: perception.status });
};
