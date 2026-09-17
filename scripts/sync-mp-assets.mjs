// Copies the MediaPipe Tasks-Vision WASM runtime out of node_modules into
// web/public/mp-wasm so the app serves it from its own origin. These files are
// ~12 MB each, so they are gitignored and re-synced on every `npm install`.
//
// This runs as an npm `postinstall` hook, which constrains how it may fail. A
// postinstall that exits non-zero fails the whole install, so a missing or
// half-written node_modules — normal during a partial or interrupted install —
// must be reported and skipped, never thrown. It also never calls
// `process.exit`: that tears the process down with writes still queued, and
// inside a lifecycle script it robs npm of a clean child exit.
import { cp, mkdir, readdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', 'web', 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const dest = join(here, '..', 'web', 'public', 'mp-wasm');

// The nosimd build is a 10 MB fallback we never select (FilesetResolver is
// always given useModule=false + SIMD, which every target browser supports).
const SKIP = /nosimd/;

async function sync() {
  try {
    await stat(src);
  } catch {
    console.warn(
      '[sync-mp-assets] @mediapipe/tasks-vision is not installed yet; skipping.\n' +
        '[sync-mp-assets] Run `npm run sync:mp` once the install finishes.',
    );
    return;
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
  console.log(
    `[sync-mp-assets] copied ${copied} files (${(bytes / 1e6).toFixed(1)} MB) -> web/public/mp-wasm`,
  );
}

try {
  await sync();
} catch (err) {
  // Deliberately not fatal. The app falls back to Google's CDN for the WASM
  // runtime, so a failed copy costs offline support, not the install.
  console.warn(`[sync-mp-assets] skipped: ${err?.message ?? err}`);
  console.warn('[sync-mp-assets] Run `npm run sync:mp` to retry.');
}
