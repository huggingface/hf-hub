#!/usr/bin/env bash
# Verify that `hf-hub` compiles for `wasm32-unknown-unknown`.
#
# Checks that hf-hub, the `wasm/smoke` crate, and the `wasm/tests` crate all
# typecheck against the wasm32 target. CI runs this as the `wasm` job in
# `.github/workflows/rust.yml`.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  echo "Installing wasm32-unknown-unknown target..."
  rustup target add wasm32-unknown-unknown
fi

echo "==> cargo check -p hf-hub --target wasm32-unknown-unknown --no-default-features"
cargo check -p hf-hub --target wasm32-unknown-unknown --no-default-features

# docs.rs builds the wasm32 target with the same features as the linux target
# (see [package.metadata.docs.rs]), so `blocking` must stay wasm-compatible.
echo
echo "==> cargo check -p hf-hub --target wasm32-unknown-unknown --features blocking"
cargo check -p hf-hub --target wasm32-unknown-unknown --features blocking

echo
echo "==> cargo check (wasm/smoke) — exercises HFRepository::download_file_stream through wasm-bindgen"
(cd wasm/smoke && cargo check --target wasm32-unknown-unknown)

echo
echo "==> cargo check --tests (wasm/tests) — wasm-bindgen-test integration tests"
(cd wasm/tests && cargo check --target wasm32-unknown-unknown --tests)

echo
echo "wasm32-unknown-unknown build is green."
