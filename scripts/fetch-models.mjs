// Downloads the two MediaPipe task bundles Aether needs into
// web/public/models. Gitignored (14 MB); the app falls back to Google's CDN
// when they are absent, so this is an optional "make it work offline" step.
import { mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE = 'https://storage.googleapis.com/mediapipe-models';
const MODELS = [
  ['gesture_recognizer.task', `${BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`],
  ['pose_landmarker_lite.task', `${BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`],
];

const dest = join(dirname(fileURLToPath(import.meta.url)), '..', 'web', 'public', 'models');
await mkdir(dest, { recursive: true });

for (const [name, url] of MODELS) {
  const out = join(dest, name);
  try {
    const { size } = await stat(out);
    if (size > 1e6) {
      console.log(`[fetch-models] ${name} already present (${(size / 1e6).toFixed(1)} MB)`);
      continue;
    }
  } catch {
    /* not downloaded yet */
  }
  process.stdout.write(`[fetch-models] ${name} ... `);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(out, buf);
  console.log(`${(buf.length / 1e6).toFixed(1)} MB`);
}
