# Base44 development notes

- This is browser-only: no API, database, or external credentials. Camera access needs browser permission; without it the app intentionally runs an animated ambient simulation.
- Base44 Compose builds the Rust/WASM from mounted source before starting Vite on host port 3000. The Node container deliberately has no Rust toolchain: its `npm run dev` skips the already-built artifact. Do not remove that WASM output while the web service is running; rebuild through the `wasm` service instead.
- `scripts/base44-wasm.sh` installs a cached wasm-pack, builds on each service start, and polls Rust sources/manifests for rebuilds. Build output, npm dependencies, vendored MediaPipe runtime, and downloaded models are generated/ignored, not committed.
- Vite uses polling and Compose passes the platform's `__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS` for rotating preview hosts. The CLI binds 0.0.0.0 without changing the normal localhost default.
- Verify with `docker compose -f docker-compose.base44.yml ps`, `curl -f http://localhost:3000/`, `docker compose -f docker-compose.base44.yml exec -T wasm cargo test --workspace`, and `docker compose -f docker-compose.base44.yml exec -T web npm run typecheck`.
- Browser readiness: `#boot.done` is intentionally invisible, so wait for attachment, not visibility. Read `window.__aether.diagnostics()` for advancing frames and camera/perception status. A canvas screenshot confirms actual rendering; a populated DOM alone does not.
- Initial setup verified native workspace tests, TypeScript checking, live source serving, and the preview rendering in ambient mode. Real webcam gesture recognition requires camera permission and was not verified during setup.
