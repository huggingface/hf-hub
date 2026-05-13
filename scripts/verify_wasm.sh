#!/usr/bin/env bash
# Verify that `hf-hub` compiles for `wasm32-unknown-unknown`.
#
# Filesystem-heavy modules (cache, blocking, buckets, snapshot download, etc.)
# are gated off on wasm. What this script verifies:
#   1. The whole crate builds against the wasm32 target with no default features.
#   2. The wasm-only `wasm_streaming` module typechecks (`xet_stream_file`
#      uses `hf-xet`'s xet streaming download path, the same one validated in
#      the `xet-core` repo under `wasm/hf_xet_wasm_download/`).
#
# Run locally with `./scripts/verify_wasm.sh`. CI runs this as the `wasm` job
# in `.github/workflows/rust.yml`.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  echo "Installing wasm32-unknown-unknown target..."
  rustup target add wasm32-unknown-unknown
fi

echo "==> cargo check -p hf-hub --target wasm32-unknown-unknown --no-default-features"
cargo check -p hf-hub --target wasm32-unknown-unknown --no-default-features

echo
echo "==> cargo check (wasm/smoke) — exercises hf-hub::wasm_streaming through wasm-bindgen"
(cd wasm/smoke && cargo check --target wasm32-unknown-unknown)

echo
echo "wasm32-unknown-unknown build is green."
