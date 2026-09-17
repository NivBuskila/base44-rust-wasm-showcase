import { expect, test } from '@playwright/test';

import { resolveModel, resolveWasmPath } from '../src/perception';

/**
 * Unit coverage for the two asset resolvers, run in Node with an injected
 * `fetch` — no browser, no network, milliseconds.
 *
 * Both are exported with a `probe` parameter specifically so this is possible,
 * and neither had a test. They are worth covering because they are the code
 * that decides whether perception works at all on a machine where the local
 * copy is missing, and because their tricky case is counter-intuitive: a 200 is
 * not proof the file is there. A dev server with an SPA fallback answers a
 * missing path with `index.html` and status 200, and taking that as success
 * hands MediaPipe an HTML page where it expects a model.
 */

type Probe = typeof fetch;

/** A `fetch` that returns one canned response, whatever the URL. */
function respond(init: { ok?: boolean; status?: number; headers?: Record<string, string> }): Probe {
  return (async () =>
    new Response(null, {
      status: init.status ?? (init.ok === false ? 404 : 200),
      headers: init.headers ?? {},
    })) as Probe;
}

/** A `fetch` that rejects, as a blocked or offline probe does. */
const rejects: Probe = (async () => {
  throw new TypeError('Failed to fetch');
}) as Probe;

const MODEL = { local: '/models/thing.task', remote: 'https://cdn.example/thing.task' };

test.describe('resolveModel', () => {
  test('takes the local copy when the probe looks like a real model', async () => {
    const probe = respond({
      headers: { 'content-length': '8000000', 'content-type': 'application/octet-stream' },
    });
    expect(await resolveModel(MODEL, probe)).toBe(MODEL.local);
  });

  test('falls back when the local copy is absent', async () => {
    expect(await resolveModel(MODEL, respond({ status: 404 }))).toBe(MODEL.remote);
  });

  test('falls back when the probe is blocked or offline', async () => {
    expect(await resolveModel(MODEL, rejects)).toBe(MODEL.remote);
  });

  test('is not fooled by an SPA fallback answering 200 with HTML', async () => {
    const probe = respond({
      headers: { 'content-length': '4000000', 'content-type': 'text/html; charset=utf-8' },
    });
    expect(await resolveModel(MODEL, probe)).toBe(MODEL.remote);
  });

  test('rejects a body too small to be a model', async () => {
    const probe = respond({
      headers: { 'content-length': '512', 'content-type': 'application/octet-stream' },
    });
    expect(await resolveModel(MODEL, probe)).toBe(MODEL.remote);
  });

  test('accepts a server that declares no length', async () => {
    // Refusing these would break the local path on any server that streams the
    // body without a `content-length`, which is the more common failure.
    const probe = respond({ headers: { 'content-type': 'application/octet-stream' } });
    expect(await resolveModel(MODEL, probe)).toBe(MODEL.local);
  });
});

test.describe('resolveWasmPath', () => {
  test('takes the vendored runtime when it is served', async () => {
    const probe = respond({ headers: { 'content-type': 'text/javascript' } });
    expect(await resolveWasmPath(probe)).toBe('/mp-wasm');
  });

  test('falls back to the CDN when the runtime was never synced', async () => {
    // The real case this exists for: `sync-mp-assets.mjs` never ran, because
    // the install was partial or the npm registry was unreachable.
    const path = await resolveWasmPath(respond({ status: 404 }));
    expect(path).toContain('cdn.jsdelivr.net');
    expect(path).toContain('@mediapipe/tasks-vision@');
  });

  test('falls back when the probe is blocked', async () => {
    expect(await resolveWasmPath(rejects)).toContain('cdn.jsdelivr.net');
  });

  test('is not fooled by an SPA fallback', async () => {
    const probe = respond({ headers: { 'content-type': 'text/html' } });
    expect(await resolveWasmPath(probe)).toContain('cdn.jsdelivr.net');
  });

  test('pins the CDN fallback to the installed package version', async () => {
    // A floating version would let the fallback be a different build than the
    // loader expects, which fails as an opaque WASM link error.
    const path = await resolveWasmPath(respond({ status: 404 }));
    const pinned = /@mediapipe\/tasks-vision@(\d+\.\d+\.\d+)\//.exec(path);
    expect(pinned, `no pinned version in ${path}`).not.toBeNull();

    const pkg = await import('../package.json', { with: { type: 'json' } });
    const declared = pkg.default.dependencies['@mediapipe/tasks-vision'];
    expect(
      pinned![1],
      `fallback pins ${pinned![1]} but package.json declares ${declared}`,
    ).toBe(declared.replace(/^[^\d]*/, ''));
  });
});
