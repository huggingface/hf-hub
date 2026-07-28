# hf-hub-wasm-smoke: streaming downloads from the browser

`cdylib + rlib` crate that wraps `hf-hub` with `#[wasm_bindgen]` and exposes
`HFRepository::download_file_stream` to JavaScript. The same call shape
(`client.model(owner, name).download_file_stream()…send().await`) used on
native is driven through xet-backed and plain LFS files from the browser.

This crate doubles as:

- a runnable browser example (`examples/download.html`), and
- a wasm32 compile smoke check (CI runs `cargo check` against this crate via
  `scripts/verify_wasm.sh`).

## JS API

```typescript
// Streams the file and returns the total number of bytes seen.
// Useful for verifying the pipeline without buffering the file in memory.
export function smoke_stream_total_bytes(
  endpoint: string,        // e.g. "https://huggingface.co"
  token: string | undefined,
  repoTypePlural: string,  // "models" | "datasets" | "spaces" | "kernels"
  owner: string,
  name: string,
  revision: string,        // branch, tag, or commit SHA
  filename: string,        // path within the repo
): Promise<number>;

// Streams the file and returns the bytes as a Uint8Array. Buffers the
// whole file in memory — use with caution for large files.
export function download_file_bytes(
  endpoint: string,
  token: string | undefined,
  repoTypePlural: string,
  owner: string,
  name: string,
  revision: string,
  filename: string,
): Promise<Uint8Array>;

// Streams the file and invokes `onProgress` with a plain object for every
// ProgressEvent, e.g. `{ kind: "download.start", total_files, total_bytes }`,
// `{ kind: "download.progress", files: [...] }`,
// `{ kind: "download.aggregate_progress", bytes_completed, total_bytes, bytes_per_sec }`,
// `{ kind: "download.complete" }`. Resolves with the total number of bytes
// streamed. This entry point only drives downloads, so only `download.*`
// variants fire here (`upload.*` kinds exist in the JS shape for parity —
// see `download_with_progress` in `src/lib.rs`).
export function download_with_progress(
  endpoint: string,
  token: string | undefined,
  repoTypePlural: string,
  owner: string,
  name: string,
  revision: string,
  filename: string,
  onProgress: (event: object) => void,
): Promise<number>;
```

## Build

```bash
./build_wasm.sh
```

Outputs `pkg/{hf_hub_wasm_smoke.js, hf_hub_wasm_smoke.d.ts,
hf_hub_wasm_smoke_bg.wasm}`.

Requires a nightly Rust toolchain plus `wasm-bindgen-cli` 0.2.121 — the
script installs the pinned `wasm-bindgen-cli` version if it's missing. The
same threaded-wasm flags (`+atomics,+bulk-memory,+mutable-globals`,
shared memory, `getrandom_backend=wasm_js`) used by
[xet-core's `wasm/hf_xet_wasm_download/`](https://github.com/huggingface/xet-core/tree/main/wasm/hf_xet_wasm_download)
are required because the `hf-xet` dependency uses threaded wasm internally.

## Manual browser test

```bash
./build_wasm.sh
# Serve with COOP/COEP headers — SharedArrayBuffer is required by the
# threaded wasm in hf-xet. `sfz --coi` works:
#   sfz --coi -p 8080
# Or any server that sets:
#   Cross-Origin-Opener-Policy: same-origin
#   Cross-Origin-Embedder-Policy: require-corp
```

Open `examples/download.html`, fill in the inputs (a public xet-backed
file works without a token), and click either:

- **Stream & count bytes** — drives `download_file_stream` and reports the
  total byte count + throughput.
- **Download to memory** — collects the stream into a Blob; **Save file**
  appears once the download completes.

Open `examples/progress.html` to exercise the progress callback path: it
drives `download_with_progress` and renders a live progress bar plus a
per-event-kind counter and event log as the
`hf_hub::progress::ProgressEvent` stream is forwarded from Rust to JS.

Under the hood the wasm code makes the same two Hub calls
(`paths-info` + `xet-read-token`) and then drives
`xet::xet_session::XetDownloadStreamGroup` — exactly what
`HFRepository::download_file_stream` does on native, just with a fresh
`XetSession` per call instead of the cached one.

## Maintainer note

The download path goes through `hf-hub`, `hf-xet` (`xet_pkg`), and the
underlying `xet_client` / `xet_data` / `xet_runtime` crates. Changes to
any of those must keep the wasm build green — see
[`AGENTS.md#WebAssembly compatibility`](../../AGENTS.md) for the patterns
this codebase relies on (`tokio_with_wasm`, conditional `?Send` futures,
filesystem gating).
