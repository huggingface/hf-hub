#!/usr/bin/env bash
# Verify that `hf-hub` compiles for `wasm32-unknown-unknown`.
#
# Filesystem-heavy modules (cache, blocking, buckets, snapshot download, etc.)
# are gated off on wasm. What this script verifies:
#   1. The whole crate builds against the wasm32 target with no default features.
#   2. The `wasm/smoke` crate typechecks against the wasm32 target — it
#      exercises `HFRepository::download_file_stream` through wasm-bindgen, which
#      is the same call shape used on native and is backed by `hf-xet`'s xet
#      streaming download (also validated in `xet-core` under
#      `wasm/hf_xet_wasm_download/`).
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
echo "==> cargo check (wasm/smoke) — exercises HFRepository::download_file_stream through wasm-bindgen"
(cd wasm/smoke && cargo check --target wasm32-unknown-unknown)

echo
echo "==> cargo check --tests (wasm/tests) — wasm-bindgen-test integration tests"
(cd wasm/tests && cargo check --target wasm32-unknown-unknown --tests)

echo
echo "wasm32-unknown-unknown build is green."
