# Aether

A gesture-driven fluid reality engine. Stand in front of a webcam and move smoke
with your hands.

An incompressible Navier–Stokes solver and a 120,000-particle system, written in
Rust and compiled to WebAssembly, driven in real time by hand tracking, body
pose and a body segmentation mask. Your silhouette becomes a solid obstacle the
fluid flows around. Your gestures are spells: pinch to open a vortex, open your
palm to push, close your fist to pull everything in, point to paint fire, rotate
both hands to warp time.

Everything runs locally in the browser. No server, no upload, no video leaving
the machine.

## Why it is built this way

The interesting constraint is that this has a **33 ms budget end to end** —
capture, inference, simulation, render — and three of those four stages are
fighting for the same frame.

**The physics is in Rust, the perception is in the browser.** Neural inference
is already a solved, GPU-accelerated problem in the browser via MediaPipe; the
fluid solver is not, and it is the part that benefits from a real language with
real control over memory layout. So `aether-core` is a dependency-free Rust
crate holding the entire simulation, and the browser layer does capture,
inference and rendering.

**Two models, not five.** `GestureRecognizer` returns hand landmarks *and* a
canned gesture label from one inference. `PoseLandmarker` with
`outputSegmentationMasks` returns body joints *and* the silhouette from one
inference. Naively you would reach for four separate tasks — hands, gestures,
pose, segmentation — and spend the whole frame budget on inference.

**Nothing is serialised per frame.** Camera luma goes *into* a Rust-owned buffer
and the dye texture and particle buffer come *out* of Rust-owned buffers, all as
typed-array views over WASM linear memory. The only per-frame copies are the two
the hardware forces: the camera readback and the texture upload.

**The engine does not depend on the models.** A dense multi-scale Horn–Schunck
optical flow solver runs over the camera's luma plane in Rust, so the fluid
reacts to whatever physically moves in front of the lens. If the models fail to
load, are blocked, or are simply too slow on the device, Aether still works — it
loses gestures, not responsiveness. With no camera at all it drives itself.

**The simulation is testable on the host.** `aether-core` compiles for x86 and
wasm32 from the same source with no `cfg` divergence in the algorithms, so the
fluid solver's incompressibility, the optical flow's sign and scale, and the
gesture state machine's hysteresis are all asserted by `cargo test` on the
machine that built it — not eyeballed through a canvas.

## Running it

Requires Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, and Node 20+.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

cd web
npm install              # also vendors the MediaPipe WASM runtime
npm run fetch:models     # optional: 14 MB of models for offline use
npm run dev              # builds the wasm, then serves on :5173
```

Open the page and allow camera access. Show your hands.

### The multithreaded engine

`npm run dev` builds the baseline engine: single-threaded, SIMD128. A second
build, `scripts/build-wasm.sh --threads`, compiles the same crate with the
`parallel` feature: the fluid kernels and the particle system run on a
[rayon](https://github.com/rayon-rs/rayon) pool of Web Workers over shared
WASM linear memory. The engine picks it at boot whenever the page is
cross-origin isolated (the dev server sends COOP/COEP for this) and falls back
to the baseline otherwise, so nothing breaks in an embedded frame or an old
browser. `?engine=single` forces the baseline for side-by-side comparison; the
HUD shows the tier and thread count either way.

The threaded build needs the nightly toolchain, because `std` itself has to be
recompiled with atomics:

```bash
rustup toolchain install nightly -c rust-src -t wasm32-unknown-unknown
scripts/build-wasm.sh release --both     # baseline into web/src/wasm, threads into web/src/wasm-mt
```

The script installs nightly on demand and treats a failed threaded build as a
warning, so a machine without it still gets a working app.

Isolation is the part that usually bites. `SharedArrayBuffer` requires both
`Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy`,
and hosts or preview proxies routinely forward the first while dropping the
second — isolation then fails silently and the HUD stays on `1 thr`. Since
headers cannot be set from the page, `public/coi-serviceworker.js` re-attaches
both from a Service Worker and the app reloads once to pick them up. Inside an
iframe this is skipped: isolation is a property of the whole frame tree, so an
embedded page follows its host and runs single-threaded.

Without `fetch:models` the app loads the models from Google's CDN on first run,
so it works out of the box; fetching them locally just makes it work offline and
start faster.

If `npm install` fails with npm's own `Exit handler never called!`, that is a
known npm bug rather than anything in this project. `npm cache clean --force`
then retry; `npm install --ignore-scripts` followed by `npm run sync:mp` gets
you moving in the meantime.

### Gestures

| Gesture | Effect |
| --- | --- |
| Open palm | Pushes fluid and particles away |
| Closed fist | Pulls everything into a gravity well |
| Pinch | Opens a vortex that winds up the longer you hold it |
| Point | Paints a hot dye trail from your fingertip |
| Victory | Freezes the fluid locally, cold palette |
| Thumb up | One-shot particle burst |
| Both hands, rotating | Warps simulation time |

Moving your hand pushes fluid regardless of gesture, and your whole body is a
collision boundary, so waving an arm displaces smoke.

## Architecture

```
web/                          TypeScript, no framework
├── camera.ts                 getUserMedia + Rec. 709 luma downscale
├── perception.ts             MediaPipe: GestureRecognizer + PoseLandmarker
├── render/                   WebGL2 compositor
├── hud.ts                    telemetry, controls, gesture legend
├── constants.ts              mirror of the Rust layout, asserted at boot
└── main.ts                   the frame loop

crates/aether-core/           zero dependencies, native + wasm32
├── field.rs                  Grid / VecField, bilinear sampling
├── fluid.rs                  Stable Fluids: advect, diffuse, vorticity, project
├── flow.rs                   multi-scale Horn–Schunck optical flow
├── particles.rs              SoA particle system, RK2 advection
├── mask.rs                   segmentation mask -> obstacle + boundary velocity
├── gesture.rs                landmark filtering, gesture state machine
├── spells.rs                 gestures -> forces, dye, particle bursts
└── engine.rs                 orchestration + the zero-copy buffer protocol

crates/aether-wasm/           wasm-bindgen glue, no logic
```

### The frame loop

Three clocks, deliberately decoupled — conflating them is what makes this class
of app stutter:

- **Render, ~60 Hz.** `engine.step(dt)` then draw. Never waits on anything.
- **Camera, ~30 Hz.** Luma readback for optical flow, and only when the video
  element has actually advanced.
- **Inference, adaptive.** `PerceptionSource.process` is synchronous —
  MediaPipe's video API has no async form — so its cost lands in the frame that
  calls it. The cadence is therefore derived from measured latency, spending a
  bounded share of wall time and leaving the rest to the simulation. If
  inference cannot fit that share even at the maximum interval, it is too
  expensive for the device: perception stands down with a reason the user can
  act on, and the optical-flow path carries the app at full responsiveness.
  Between inferences the last known landmarks keep driving the simulation.

### Coordinate conventions

One convention, stated once, because mixing these up is the source of most bugs
in a pipeline like this:

- Inside `aether-core`, everything is **grid space**: a value lives at integer
  lattice point `(i, j)` and a `w × h` grid spans `[0, w-1] × [0, h-1]`.
  Sampling is bilinear and clamps at the border.
- Velocities are **grid cells per second**, never per frame. Every decay goes
  through `math::decay(rate, dt)`, so a 30 fps and a 144 fps session look
  identical.
- Normalised `[0, 1]` coordinates appear **only** at the engine boundary, where
  landmarks come in.
- The user sees a **mirrored** camera view, so landmarks and the luma plane are
  mirrored to match. Otherwise moving your hand right would push the fluid left.

The fluid grid is 192 × 108 for a 16:9 frame, so cells are square in screen
space and no anisotropic correction is needed anywhere.

## Measured

Native release, on a 4-core container, from
`cargo test --release -p aether-core --test perf -- --nocapture`:

| stage | cost |
| --- | --- |
| fluid step (advect, diffuse, vorticity, project) | 5.1 ms |
| + 120k particles | 13.6 ms |
| optical flow, per camera frame | 1.5 ms |
| one 60 fps frame | 16.7 ms |

In-browser engine step: **17.8 ms** with 120k particles, 9.7 ms with none.

Two numbers that are *not* the engine, recorded so they are not misread:

- **Headless frame rate is fill-rate bound on SwiftShader.** fps tracks pixel
  count almost exactly (922k px → 3.7 fps, 518k → 5.5, 230k → 8.8, 58k → 12.1)
  while turning off all 120k particles moves it only 3.7 → 4.8. That is a
  software rasteriser shading ~20 texture fetches per pixel across the bloom
  chain, which a GPU does in single-digit milliseconds. The suite therefore
  asserts on `stepMs` and on the loop not stalling, never on fps.
- **Headless inference is ~1200 ms per call.** MediaPipe's GPU delegate on
  SwiftShader is software. The app measures this and stands down rather than
  stalling — see the frame loop section.

## Testing

```bash
cargo test --workspace     # 196 tests: the simulation, on the host
cd web && npm run test:e2e # 24 tests: the browser, headless, no webcam
```

The end-to-end suite runs the real app in headless Chromium with **no camera and
no GPU**:

- Chromium's fake capture device is fed a synthetic Y4M clip generated by
  `scripts/make-fixture-video.mjs` — a static textured background with one disc
  moving at a known velocity, so the optical flow path has ground truth to be
  asserted against rather than eyeballed.
- The gesture suite injects a scripted `PerceptionSource` through a test hook and
  drives the exact packed buffer layout the real perception module fills. That
  exercises the whole chain from landmarks to latched spells to pixels without
  needing a real hand in front of a real camera.
- WebGL2 runs on SwiftShader.

On a machine where Playwright's own browser download is unavailable, point it at
an existing Chromium with `AETHER_CHROMIUM=/path/to/chromium`.

## License

MIT
