#!/usr/bin/env bash
# Build from the mounted source, then rebuild Rust changes for Vite's reload loop.
set -euo pipefail
cd /app
rm -f /tmp/aether-wasm-ready
rustup target add wasm32-unknown-unknown
if ! command -v wasm-pack >/dev/null 2>&1; then
  cargo install wasm-pack --version 0.13.1 --locked
fi

touch /tmp/aether-wasm-built
# Baseline first so the healthcheck passes as soon as the app can run at all;
# the threaded engine (nightly, build-std) follows and is optional.
bash scripts/build-wasm.sh release
touch /tmp/aether-wasm-ready
bash scripts/build-wasm.sh release --threads || echo 'threaded engine build failed; running single-threaded' >&2

while sleep 2; do
  if find crates Cargo.toml Cargo.lock scripts/build-wasm.sh -type f \
    -newer /tmp/aether-wasm-built -print -quit | grep -q .; then
    touch /tmp/aether-wasm-built
    if ! bash scripts/build-wasm.sh release --both; then
      echo 'WASM rebuild failed; fix the source to retry.' >&2
    fi
  fi
done
