#!/bin/sh
#
# Run hf-hub's wasm32-unknown-unknown integration tests in a headless browser.
#
# The tests drive `hf-xet`'s threaded wasm at runtime, so the build needs
# atomics, bulk-memory, mutable-globals, shared memory, and a rebuilt std (via
# `-Z build-std`) — hence the nightly toolchain and the rustflags below.
#
# Browser only: the test crate compiles in
# `wasm_bindgen_test_configure!(run_in_browser)` unconditionally, so
# `wasm-bindgen-test-runner` always drives a headless browser. Node is not
# supported — `hf-xet`'s upload path spawns Web Workers that only function in a
# `crossOriginIsolated` context, which Node's wasm-bindgen-test runner can't
# provide (Node has `SharedArrayBuffer` but no Web `Worker`). `wasm-bindgen-
# test-runner` (>= 0.2.121) serves the required
# `Cross-Origin-Opener-Policy: same-origin` and
# `Cross-Origin-Embedder-Policy: require-corp` headers automatically, so
# `SharedArrayBuffer` and workers are available on the test page. Node
# consumers should use a native addon instead of the wasm build.
#
# Requirements:
#   - Nightly Rust toolchain with `rust-src` (`rustup component add
#     rust-src --toolchain nightly`)
#   - `wasm-bindgen-cli` 0.2.121 (installed automatically if missing)
#   - One of Chrome + chromedriver, Firefox + geckodriver, or Safari +
#     safaridriver in PATH (selected via the relevant `*DRIVER` env var — see
#     https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html).
#
# Extra arguments are forwarded to `cargo test` (e.g. a test-name filter).

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
# (especially the xet transfers, which fan out to the CAS gateway) can
# exceed that.
: "${WASM_BINDGEN_TEST_TIMEOUT:=180}"
export WASM_BINDGEN_TEST_TIMEOUT

CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="$TARGET_RUSTFLAGS" \
cargo +nightly test \
    --target wasm32-unknown-unknown \
    --release \
    -Z build-std=std,panic_abort \
    "$@"
