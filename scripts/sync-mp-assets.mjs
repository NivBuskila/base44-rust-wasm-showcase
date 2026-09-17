// Copies the MediaPipe Tasks-Vision WASM runtime out of node_modules into
// web/public/mp-wasm so the app serves it from its own origin. These files are
// ~12 MB each, so they are gitignored and re-synced on every `npm install`.
import { cp, mkdir, readdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'web', 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const dest = join(here, '..', 'web', 'public', 'mp-wasm');

// The nosimd build is a 10 MB fallback we never select (FilesetResolver is
// always given useModule=false + SIMD, which every target browser supports).
const SKIP = /nosimd/;

try {
  await stat(src);
} catch {
  console.warn('[sync-mp-assets] @mediapipe/tasks-vision not installed yet; skipping.');
  process.exit(0);
}

await mkdir(dest, { recursive: true });
let copied = 0;
let bytes = 0;
for (const name of await readdir(src)) {
  if (SKIP.test(name)) continue;
  const from = join(src, name);
  bytes += (await stat(from)).size;
  await cp(from, join(dest, name));
  copied++;
}
console.log(`[sync-mp-assets] copied ${copied} files (${(bytes / 1e6).toFixed(1)} MB) -> web/public/mp-wasm`);
