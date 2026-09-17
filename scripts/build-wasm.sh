#!/usr/bin/env bash
# Builds aether-core + aether-wasm into web/src/wasm/ as an ES module.
#
# Usage:
#   scripts/build-wasm.sh [release|dev] [--if-missing]
#
# --if-missing skips the build when the artefact is already there, so
# `npm run dev` does not pay 70 seconds of LTO on every start.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/web/src/wasm"
PROFILE="release"
IF_MISSING=0

for arg in "$@"; do
  case "$arg" in
    release | dev) PROFILE="$arg" ;;
    --if-missing) IF_MISSING=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: $0 [release|dev] [--if-missing]" >&2
      exit 2
      ;;
  esac
done

if [ "$IF_MISSING" = "1" ] && [ -f "$OUT/aether_bg.wasm" ] && [ -f "$OUT/aether.js" ]; then
  echo "==> web/src/wasm is already built; skipping (drop --if-missing to force)"
  exit 0
fi

if ! command -v wasm-pack > /dev/null 2>&1; then
  cat >&2 << 'MSG'
error: wasm-pack is not on PATH.

Install it with one of:
  cargo install wasm-pack
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

You also need the wasm target:
  rustup target add wasm32-unknown-unknown
MSG
  exit 1
fi

if ! rustup target list --installed 2> /dev/null | grep -q wasm32-unknown-unknown; then
  echo "==> adding the wasm32-unknown-unknown target"
  rustup target add wasm32-unknown-unknown
fi

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

# wasm-pack leaves behind a .gitignore that would hide the whole output
# directory from tooling, and a README that is not ours.
rm -f "$OUT/package.json" "$OUT/.gitignore" "$OUT/README.md"

echo "==> $(du -h "$OUT/aether_bg.wasm" | cut -f1) -> web/src/wasm/aether_bg.wasm"
