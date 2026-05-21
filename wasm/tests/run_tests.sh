#!/bin/sh
#
# Run hf-hub's wasm32-unknown-unknown integration tests. Mirrors
# `wasm/smoke/build_wasm.sh`'s compile flags — atomics, bulk-memory,
# mutable-globals, shared memory, and a rebuilt std (via `-Z build-std`) —
# because the xet download test exercises `hf-xet`'s threaded wasm at
# runtime. The rest of the tests (plain HTTP) would build without these
# flags, but a single build matrix keeps the crate simple.
#
# Two run modes:
#
#   * Browser (default): drives a headless browser through
#     `wasm-bindgen-test-runner`. v0.2.121 serves
#     `Cross-Origin-Opener-Policy: same-origin` and
#     `Cross-Origin-Embedder-Policy: require-corp` by default, so
#     `SharedArrayBuffer` is available in the browser test page without
#     extra configuration. (Disable via
#     `WASM_BINDGEN_TEST_NO_ORIGIN_ISOLATION` if you ever need to compare
#     behaviour.)
#   * Node (`RUN_IN_NODE=1`): builds with `--features node-tests`, which
#     drops the `wasm_bindgen_test_configure!(run_in_browser)` directive
#     and lets `wasm-bindgen-test-runner` fall back to its default Node.js
#     runner. No webdriver / headless browser required. Validated against
#     Node 20+; older versions may not support the shared-memory + atomics
#     combination `hf-xet` needs.
#
# Requirements:
#   - Nightly Rust toolchain with `rust-src` (`rustup component add
#     rust-src --toolchain nightly`)
#   - `wasm-bindgen-cli` 0.2.121 (installed automatically if missing)
#   - Browser mode only: one of Chrome + chromedriver, Firefox +
#     geckodriver, or Safari + safaridriver in PATH (selected via the
#     relevant `*DRIVER` env var — see
#     https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html).
#   - Node mode only: `node` on PATH (Node 20+).

set -ex

cd "$(dirname "$0")"

WASM_BINDGEN_VERSION="0.2.121"

if command -v wasm-bindgen >/dev/null 2>&1; then
    INSTALLED_WASM_BINDGEN_VERSION="$(wasm-bindgen --version | awk '{print $2}')"
else
    INSTALLED_WASM_BINDGEN_VERSION=""
fi

if [ "$INSTALLED_WASM_BINDGEN_VERSION" != "$WASM_BINDGEN_VERSION" ]; then
    cargo install -f wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION"
fi

TARGET_RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--shared-memory \
  -C link-arg=--max-memory=4294967296 \
  -C link-arg=--import-memory \
  -C link-arg=--export=__wasm_init_tls \
  -C link-arg=--export=__tls_size \
  -C link-arg=--export=__tls_align \
  -C link-arg=--export=__tls_base \
  -C link-arg=--export=__heap_base \
  --cfg getrandom_backend=\"wasm_js\""

# `wasm-bindgen-test` defaults to a 20s per-test timeout; live Hub calls
# (especially the xet download, which fans out to the CAS gateway) can
# exceed that.
: "${WASM_BINDGEN_TEST_TIMEOUT:=180}"
export WASM_BINDGEN_TEST_TIMEOUT

FEATURE_FLAGS=""
if [ "${RUN_IN_NODE:-0}" = "1" ]; then
    FEATURE_FLAGS="--features node-tests"
fi

CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="$TARGET_RUSTFLAGS" \
cargo +nightly test \
    --target wasm32-unknown-unknown \
    --release \
    -Z build-std=std,panic_abort \
    $FEATURE_FLAGS \
    "$@"
