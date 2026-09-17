#!/usr/bin/env bash
# Builds aether-core + aether-wasm into web/src/wasm/ as an ES module.
#
# simd128 is on unconditionally: every browser that supports the MediaPipe
# Tasks runtime also supports WASM SIMD, so there is no fallback to maintain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/web/src/wasm"
PROFILE="${1:-release}"

# Target-scoped rather than plain RUSTFLAGS: the latter also reaches host
# build scripts and proc macros, where simd128 is not a valid feature and
# every crate warns about it.
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="${CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS:-} -C target-feature=+simd128"

echo "==> building aether-wasm ($PROFILE, simd128)"
cd "$ROOT"

# TypeScript declarations are on by default; --no-pack skips the npm
# package.json wasm-pack would otherwise drop into the output directory.
ARGS=(--target web --out-dir "$OUT" --out-name aether --no-pack)
if [ "$PROFILE" = "dev" ]; then
  ARGS+=(--dev)
else
  ARGS+=(--release)
fi

wasm-pack build crates/aether-wasm "${ARGS[@]}" -- --features aether-core/simd

# wasm-pack emits a package.json that confuses Vite's dependency scanner,
# and a .gitignore that would hide the whole output directory.
rm -f "$OUT/package.json" "$OUT/.gitignore" "$OUT/README.md"

echo "==> $(du -h "$OUT/aether_bg.wasm" | cut -f1) -> $OUT/aether_bg.wasm"
