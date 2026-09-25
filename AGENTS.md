# Development notes

Non-obvious rules only: what the README and the manifests do not say. Each bullet stands alone.

## Commits and pull requests

- Never put a link to an AI chat or session (such as `claude.ai/code/session_…`, or a `Claude-Session:` trailer) in a commit message, a PR title or description, a code comment or any file. The repository is public.

## Local environment (Docker)

- Browser-only app: no API, database or credentials. Without camera permission it runs an ambient simulation by design.
- `docker-compose.base44.yml` runs two services from the mounted source. `wasm` builds both engines with `scripts/base44-wasm.sh` and polls Rust sources for rebuilds. `web` runs Vite on port 3000 and has no Rust toolchain, so it relies on the artifacts already being built. Rebuild through the `wasm` service, and restart it after editing `base44-wasm.sh`.
- There are two WASM builds: `web/src/wasm` (baseline) and `web/src/wasm-mt` (rayon + Web Workers, nightly `-Zbuild-std`, own target dir `target/wasm-mt`). The threaded link must export `__wasm_init_tls`/`__tls_*` explicitly (see `scripts/build-wasm.sh`). The `wasm` healthcheck passes only after both builds are attempted, so Vite never starts without `web/src/wasm-mt/`. If that directory is missing, the threaded build failed: check the `wasm` log.
- Cross-origin isolation: some proxies drop `Cross-Origin-Embedder-Policy`. `web/public/coi-serviceworker.js` + `web/src/cross-origin-isolation.ts` re-attach it from a Service Worker and reload once. This runs only at top level, so an embedded iframe is always single-threaded. Check the engine with `window.__aether.diagnostics().engine` in a standalone tab, and judge frame rate only there.

## Verifying

- Commands: `docker compose -f docker-compose.base44.yml exec -T wasm cargo test --workspace`, plus `... exec -T web npm run typecheck` and `... exec -T web npm run test:unit`. Test counts: Rust 237, vitest 206, Playwright 35. Playwright is not run in CI or in the container. Update the README's Testing section when these counts change.
- CI pins toolchain `1.90.0`, the same as `rust-toolchain.toml`; otherwise rustfmt/clippy go missing. CI also runs the threaded build, because the typecheck follows `engine-loader.ts`'s dynamic import into `web/src/wasm-mt`. Run fmt/clippy before pushing. The two `needless_range_loop` allows in `fluid/` are deliberate.
- Browser readiness: `#boot.ready` = engine running and perception warmed (capped at 30 s). `.done` is invisible, so wait for attachment rather than visibility. `window.__aether.diagnostics()` reports frames, camera/perception status, `renderBackend` and `qualityTier`. Only a canvas screenshot proves rendering.
- Welcome flow state lives in `localStorage` (`aether.landing.seen`, `aether.tutorial.done`). Clear it to replay the welcome, or pass `?welcome`.
- `web/src/qa-recorder.ts` records a bounded QA summary: fps percentiles, stage times, tier/perception transitions, `engineReason`, `crossOriginIsolated`. `outsideMs` is time the frame callback did not spend, such as GPU completion or compositing. `perceptionBudgetHz` is a target, not throughput. In dev the summary is also sent to `/__qa/session` (`web/qa-session-endpoint.ts`), so read it with `curl localhost:3000/__qa/session`. `localStorage` is partitioned per top-level site and cannot bridge an iframe and a tab. Only top-level documents upload.

## Deploying (Cloudflare Pages)

- `.github/workflows/deploy.yml` builds both WASM variants in Actions, because Cloudflare's image can't run the nightly build. It self-hosts the MediaPipe runtime and models, then deploys `web/dist`. The README lists the settings it needs.
- Deploy runs on `workflow_run` of the workflow named `CI`, only after it succeeds on a push to `main`. Renaming `ci.yml`'s `name:` silently stops deploys, so rename both.
- The `og:*`/`twitter:*` tags in `web/index.html` use `%VITE_SITE_URL%`, because crawlers need absolute URLs. `web/.env` holds the dev default, and the deploy workflow overrides it.
- Production-only pitfall: the minifier turns an indirect `eval` into a direct one, which breaks the perception worker ("ModuleFactory not set"). `worker-import-scripts.ts` therefore calls `globalThis.eval`. After a build, check with `grep responseText web/dist/assets/perception.worker-*.js`.
- `web/public/_headers` sends COOP/COEP. Pages caps files at 25 MB, and the largest dist file is about 12 MB.

## Web: boot and frame loop

- `main.ts` holds boot and the `window.__aether` test hooks. `app.ts` holds the `App` and the frame loop. `frame-clock.ts` owns the deltas and `frame-timings.ts` the per-stage timings. `engine-views.ts` owns every typed-array view over WASM memory: call `rebuild` after any reallocating engine call and `refresh` once per frame. `engine-pool.ts` is the only writer of pool size. `perception-pump.ts` owns the perception source's lifetime and cadence.
- `EngineViews.stats()` returns a live view into WASM memory. The next call overwrites it and memory growth detaches it, so copy any values you keep across frames. Readers outside the loop use `lastStats()`.
- The compact loader is prerendered in `web/index.html`, with `styles.css` linked there. Keep that markup in sync with `build` in `landing.ts`. Animate the landing exit with opacity/transform only: full-screen filters and masks stutter while the engine runs.
- `performance-governor.ts` steps down internal resolution, then `pressure_iters`, then pool size. Its windows are wall-clock ms fed from `realDt`, and under 30 fps it drops two rungs at once. It holds its tier while `PerceptionPump.warming`, because MediaPipe's cold start would otherwise drive it to the bottom rung.
- Two gesture-latency floors: the worker constructs the limiter with `rationInference: false` (rationing only matters for the inline fallback), and the camera requests `frameRate: { ideal: 60 }`, since frames are submitted only when `video.currentTime` advances.

## Web: perception

- Inference runs in `perception.worker.ts` on `ImageBitmap`s at `INFERENCE_WIDTH`. `perception-worker-client.ts` keeps one inference in flight, and `process` returns the previous result. Inline `MediaPipePerception` is the fallback and stays for the session once taken. `FRAME_TIMEOUT_MS` is generous (60 s), because MediaPipe compiles GPU shaders lazily per subgraph. `perception-protocol.ts` is the shared message contract.
- A module worker's inherited `importScripts` throws, so `worker-import-scripts.ts` overrides it. Vite's `?import` query is refused for `public/` files, so the `aether:mp-wasm-as-asset` plugin in `vite.config.ts` strips it. Rayon threads need `worker.format: 'es'`.
- `perception/packing/` is DOM- and MediaPipe-free. `detections.ts` is the only place landmarks are mirrored; the mask is mirrored by `push_mask`. Import the directory index, never a `packing.ts` file.

## Web: rendering

- Two backends sit behind `SceneRenderer` (`types.ts`): WebGL2 (`render/`) and WebGPU (`gpu/`, which also simulates the particles). `gpu/index.ts` selects WebGPU only on a hardware adapter. The look is shared in `render/styles.ts`, `render/look.ts` and `render/sizing.ts`. Make look changes there, and keep only the packing per backend. `render-backends.test.ts` pins the surface on both prototypes.
- Both backends are split the same way (renderer, bloom chain, overlay, programs/pipelines, sources, scene/particle passes), so mirrored changes land in mirrored files. On WebGL2, everything that dies with the context lives in `render/resources.ts` and is rebuilt as one unit.
- WebGPU op log: the particle pool moves to the GPU and every spell mutation is appended to a per-frame op log. `OP_KINDS` in `particles/offload.rs` is the single source; `assertOpLayout` checks it at boot and `op-protocol.test.ts` pins it. Add a kind in Rust and TS together. Randomness differs from the CPU path, so compare the backends statistically.
- A WebGPU submit that touches a mapped (or map-pending) buffer is silently dropped, which shows up as flicker. All readbacks go through `gpu/readback.ts`: check `idle` before copying and call `poll` after submit. Keep `onuncapturederror` → `gpu/error-log.ts` → QA session intact.
- Bind groups are rebuilt only from `SceneSources.version` and `sim.poolVersion`. `FrameChain.resize` reallocates targets and their bind groups together.
- WGSL is tested without a GPU by `shaders.test.ts` (via `wgsl_reflect`), covering uniform sizes, bindings, workgroup size and struct layout. `vitest.config.ts` aliases `wgsl_reflect` to its ESM build. `sim-uniform.test.ts` pins the `Sim` slot order, so change the shader struct and that test together. The compute and draw halves import `particles-layout.ts`, never the `particles.ts` index, because that creates a cycle.
- In the threaded build, WASM views can be backed by `SharedArrayBuffer`, so the two `writeTexture` casts are deliberate.

## Web: HUD, styles, tutorial

- `hud-spec.ts` is the HUD's static vocabulary. `hud-markup.ts` builds the DOM once, and all later writes change only text, `data-*` attributes or CSS variables. The DOM-free logic (`hud-keys`, `hud-status`, `hud-readouts`, `hud-frame-timer`, `hud-cells`) is unit-tested.
- `styles.css` is only `@import`s, and their order is the cascade order: `responsive`, `overdrive` and `spellbook` override the files before them on purpose.
- The tutorial is a mode. It calls `set_practice(true)`, which keeps single spells but suppresses combos and duets. A step passes after the spell has been latched for `HOLD_MS`, except `release`, which passes on sight.
- `hand-skeleton.ts` is the only hand drawing. `goal-pose.ts` generates poses from finger declarations, except the fist and thumbs-up, which are traced landmark tables. `hand-mirror.ts` reads the first occupied slot, because slots are sticky by handedness.
- `Release` is reported by `spells.rs` only and is never latched. Duet distances are in hand widths. Throw was removed: `rush_state()` always returns `IDLE`, and the type is kept for WASM compatibility.

## Rust layout

- Every subsystem is a directory. `mod.rs` holds the contract (constants, struct, `new`, accessors), with one child file per stage or ingest. Children reach private fields through Rust's descendant privacy, so `pub(super)`/`pub(in …)` marks a cross-module call, not a public API.
- The stage order in `engine/frame.rs` and `fluid/step.rs` is load-bearing and commented inline. `set_obstacle` is the only writer of `obstacle`, and any new writer must call `refresh_solid`.
- `fluid/kernels/scalar.rs` is compiled everywhere so it can be cross-checked against `simd.rs`, the only `cfg(wasm32 + simd)` module.
- Test folders follow a fixed layout: `mod.rs` holds only the `mod` list, `support.rs` holds the fixtures (all `pub(super)`) and re-exports, and each theme file starts with `use super::support::*;`.
- `visual.rs` scores the dye field for tuning. `contrast` is brightness-normalised, and `divergence` is absolute, so compare it with `max_speed`.
- Threading: `Engine::step` enters the rayon pool once through `par::install`, the Jacobi sweeps run serially (`sweep_rows`), and particle lanes are several per thread (`par::balanced_len`). All three were measured faster in the browser, with a 4 ms gap between steps. Compare a threading change against `?engine=single` before keeping it, because on a 256×144 grid a fork/join can cost more than it saves.
- CI also runs `cargo test -p aether-core --features parallel`. `fluid/tests/threads.rs` requires the solver to be bit-identical on 1, 2, 3, 5 and 7 threads. Index a band's rows by the requested chunk length, never by the chunk's own `len()`, because the last band is shorter.
- `aether-wasm` spreads `#[wasm_bindgen] impl` blocks across `lib.rs`, `buffers.rs`, `gpu.rs` and `reports.rs`, and holds no logic.
