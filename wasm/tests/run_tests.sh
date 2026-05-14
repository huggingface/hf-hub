#!/bin/sh
#
# Run hf-hub's wasm32-unknown-unknown integration tests in a headless
# browser.
#
# The tests target the wasm-safe, pure-HTTP subset of `hf-hub` (file
# download via plain HTTP, model info, raw diff, list_spaces) — none of
# them exercise hf-xet at runtime, so the threaded-wasm setup from
# `wasm/smoke/build_wasm.sh` (atomics, shared memory, build-std) is
# intentionally NOT mirrored here. Stable Rust + a plain
# `cargo test --target wasm32-unknown-unknown` is enough.
#
# Requirements:
#   - `wasm32-unknown-unknown` target installed (`rustup target add ...`)
#   - `wasm-bindgen-cli` 0.2.121 (installed automatically if missing)
#   - One of: Chrome + chromedriver, Firefox + geckodriver, or Safari +
#     safaridriver in PATH. Selection via the `--firefox` / `--chrome`
#     / `--safari` flag passed after `--`. Defaults to `--chrome` here.
#     See https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/browsers.html

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

# `wasm-bindgen-test` defaults to a 20s per-test timeout; live Hub calls
# (especially first-fetch cold paths) routinely take longer.
: "${WASM_BINDGEN_TEST_TIMEOUT:=120}"
export WASM_BINDGEN_TEST_TIMEOUT

cargo test \
    --target wasm32-unknown-unknown \
    --release \
    "$@"
