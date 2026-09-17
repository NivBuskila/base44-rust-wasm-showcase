/**
 * Generates a synthetic Y4M clip for Chromium's fake camera device, so the
 * whole app can be exercised headless with no webcam and no ffmpeg.
 *
 * The content is chosen to make optical flow *verifiable* rather than pretty:
 *
 * - A static textured background. Flow must report ~zero motion there, so any
 *   drift shows up as a test failure rather than as plausible noise.
 * - One bright disc moving on a known path. Its velocity is the ground truth
 *   the flow test asserts against.
 *
 * Usage: node scripts/make-fixture-video.mjs [out.y4m] [seconds]
 */
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const WIDTH = 640;
const HEIGHT = 480;
const FPS = 30;

/** Disc travels left-to-right across the middle at this speed. */
const DISC_SPEED_PX_PER_SEC = 220;
const DISC_RADIUS = 46;

const here = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(
  process.argv[2] ?? join(here, '..', 'web', 'tests', 'fixtures', 'motion.y4m'),
);
const seconds = Number(process.argv[3] ?? 4);
const frameCount = Math.max(1, Math.round(seconds * FPS));

/** Deterministic PRNG so the fixture is byte-identical on every machine. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Static background: a soft diagonal gradient plus fixed speckle. The speckle
// is what gives Horn-Schunck gradients to lock onto; a flat background has no
// measurable flow at all and would make the test vacuous.
const background = new Uint8Array(WIDTH * HEIGHT);
{
  const rand = mulberry32(0x5eed);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const gradient = 40 + 50 * ((x / WIDTH) * 0.6 + (y / HEIGHT) * 0.4);
      const speckle = (rand() - 0.5) * 36;
      background[y * WIDTH + x] = Math.max(0, Math.min(255, gradient + speckle)) | 0;
    }
  }
}

const ySize = WIDTH * HEIGHT;
const cSize = (WIDTH / 2) * (HEIGHT / 2);
const header = Buffer.from(`YUV4MPEG2 W${WIDTH} H${HEIGHT} F${FPS}:1 Ip A1:1 C420jpeg\n`, 'ascii');
const frameMarker = Buffer.from('FRAME\n', 'ascii');

const chunks = [header];
const yPlane = Buffer.allocUnsafe(ySize);
// Neutral chroma: the clip is greyscale apart from the disc, and the engine
// only ever consumes luma.
const uPlane = Buffer.alloc(cSize, 128);
const vPlane = Buffer.alloc(cSize, 128);

for (let f = 0; f < frameCount; f++) {
  const t = f / FPS;
  // Wrap the disc's path so the clip loops seamlessly when Chromium repeats it.
  const span = WIDTH + DISC_RADIUS * 4;
  const cx = ((DISC_SPEED_PX_PER_SEC * t) % span) - DISC_RADIUS * 2;
  const cy = HEIGHT * 0.5 + Math.sin(t * 1.7) * HEIGHT * 0.18;

  yPlane.set(background);

  const x0 = Math.max(0, Math.floor(cx - DISC_RADIUS));
  const x1 = Math.min(WIDTH - 1, Math.ceil(cx + DISC_RADIUS));
  const y0 = Math.max(0, Math.floor(cy - DISC_RADIUS));
  const y1 = Math.min(HEIGHT - 1, Math.ceil(cy + DISC_RADIUS));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (d > DISC_RADIUS) continue;
      // Soft edge: a hard step aliases badly at this resolution and produces
      // flow artefacts that have nothing to do with the real motion.
      const edge = Math.min(1, (DISC_RADIUS - d) / 6);
      const base = yPlane[y * WIDTH + x];
      yPlane[y * WIDTH + x] = Math.min(255, base + 150 * edge) | 0;
    }
  }

  chunks.push(frameMarker, Buffer.from(yPlane), uPlane, vPlane);
}

await mkdir(dirname(outPath), { recursive: true });
const out = Buffer.concat(chunks);
await writeFile(outPath, out);

console.log(
  `[make-fixture-video] ${outPath}\n` +
    `  ${WIDTH}x${HEIGHT} @ ${FPS}fps, ${frameCount} frames, ${(out.length / 1e6).toFixed(1)} MB\n` +
    `  disc: ${DISC_SPEED_PX_PER_SEC} px/s horizontal, radius ${DISC_RADIUS}`,
);
