#!/usr/bin/env bash
# Builds aether-core + aether-wasm into web/src/wasm/ as an ES module.
#
# Usage:
#   scripts/build-wasm.sh [release|dev] [--if-missing] [--threads|--both]
#
# --if-missing skips the build when the artefact is already there, so
# `npm run dev` does not pay 70 seconds of LTO on every start.
#
# --threads builds the multithreaded engine instead, into web/src/wasm-mt/:
# rayon over Web Workers sharing WASM linear memory. That needs the atomics
# target features and a std compiled with them, which only nightly's
# `-Zbuild-std` can do, so it uses the nightly toolchain (installed on demand
# with rust-src) and its own target dir so the two builds never thrash each
# other's fingerprints. --both builds the single-threaded module and then the
# threaded one; a missing nightly is a warning, not a failure, so the app
# always has at least its baseline engine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"
IF_MISSING=0
THREADS=0
BOTH=0

for arg in "$@"; do
  case "$arg" in
    release | dev) PROFILE="$arg" ;;
    --if-missing) IF_MISSING=1 ;;
    --threads) THREADS=1 ;;
    --both) BOTH=1 ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: $0 [release|dev] [--if-missing] [--threads|--both]" >&2
      exit 2
      ;;
  esac
done

if [ "$BOTH" = "1" ]; then
  ARGS=("$PROFILE")
  [ "$IF_MISSING" = "1" ] && ARGS+=(--if-missing)
  bash "$0" "${ARGS[@]}"
  if bash "$0" "${ARGS[@]}" --threads; then
    exit 0
  fi
  echo "==> WARNING: the threaded engine did not build; the app will run single-threaded" >&2
  exit 0
fi

if [ "$THREADS" = "1" ]; then
  OUT="$ROOT/web/src/wasm-mt"
  LABEL="threads + simd128"
else
  OUT="$ROOT/web/src/wasm"
  LABEL="simd128"
fi

if [ "$IF_MISSING" = "1" ] && [ -f "$OUT/aether_bg.wasm" ] && [ -f "$OUT/aether.js" ]; then
  echo "==> ${OUT#"$ROOT/"} is already built; skipping (drop --if-missing to force)"
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

# Target-scoped rather than plain RUSTFLAGS: the latter also reaches host
# build scripts and proc macros, where simd128 is not a valid feature and
# every crate warns about it.
FLAGS="${CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS:-} -C target-feature=+simd128"
FEATURES="aether-core/simd"
EXTRA=()

if [ "$THREADS" = "1" ]; then
  if ! rustup toolchain list 2> /dev/null | grep -q '^nightly'; then
    echo "==> installing the nightly toolchain for the threaded build"
    rustup toolchain install nightly --profile minimal -c rust-src -t wasm32-unknown-unknown
  fi
  # rust-src must be present for -Zbuild-std; a nightly installed without it
  # is silently useless here, so make sure.
  rustup component add rust-src --toolchain nightly > /dev/null 2>&1 || true
  rustup target add wasm32-unknown-unknown --toolchain nightly > /dev/null 2>&1 || true
  export RUSTUP_TOOLCHAIN=nightly
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}/wasm-mt"
  # The target features make std thread-safe; the link args make the memory
  # itself shared and imported, so every worker instantiates the module over
  # the one buffer the main thread created. 1 GiB is the ceiling, not a
  # reservation — the engine needs ~64 MB at the 1M-particle pool.
  FLAGS="$FLAGS -C target-feature=+atomics,+bulk-memory,+mutable-globals"
  FLAGS="$FLAGS -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=1073741824"
  # wasm-bindgen's threading pass needs the TLS bootstrap symbols exported.
  # Recent nightlies leave them hidden unless asked, and it then fails with
  # "failed to find `__wasm_init_tls`".
  for sym in __wasm_init_tls __tls_size __tls_align __tls_base; do
    FLAGS="$FLAGS -C link-arg=--export=$sym"
  done
  FEATURES="$FEATURES,parallel"
  EXTRA+=(-Z build-std=std,panic_abort)
elif ! rustup target list --installed 2> /dev/null | grep -q wasm32-unknown-unknown; then
  echo "==> adding the wasm32-unknown-unknown target"
  rustup target add wasm32-unknown-unknown
fi
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="$FLAGS"

echo "==> building aether-wasm ($PROFILE, $LABEL)"
cd "$ROOT"

# TypeScript declarations are on by default; --no-pack skips the npm
# package.json wasm-pack would otherwise drop into the output directory.
ARGS=(--target web --out-dir "$OUT" --out-name aether --no-pack)
if [ "$PROFILE" = "dev" ]; then
  ARGS+=(--dev)
else
  ARGS+=(--release)
fi

wasm-pack build crates/aether-wasm "${ARGS[@]}" -- --features "$FEATURES" "${EXTRA[@]}"

# wasm-pack leaves behind a .gitignore that would hide the whole output
# directory from tooling, and a README that is not ours.
rm -f "$OUT/package.json" "$OUT/.gitignore" "$OUT/README.md"

if [ "$THREADS" = "1" ]; then
  # wasm-bindgen-rayon's worker helper reaches the main module with
  # `import('../../..')` — a directory import that expects a package entry
  # point. --no-pack removed package.json, so give the directory an index.
  cat > "$OUT/index.js" << 'JS'
export * from './aether.js';
export { default } from './aether.js';
JS
fi

echo "==> $(du -h "$OUT/aether_bg.wasm" | cut -f1) -> ${OUT#"$ROOT/"}/aether_bg.wasm"
