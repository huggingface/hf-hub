//! `hf-hub` wasm32 example + smoke verification.
//!
//! Builds for `wasm32-unknown-unknown` and proves that the wasm-safe subset of
//! `hf-hub` (`HFClient`, `HFRepository::download_file_stream`) compiles and is
//! callable in a wasm-bindgen environment.
//!
//! Exports three `#[wasm_bindgen]` functions:
//! - [`smoke_stream_total_bytes`] — streams a file and returns the total bytes seen. Used by `examples/download.html`
//!   for post-hoc reporting and by `scripts/verify_wasm.sh` as a monomorphization smoke check.
//! - [`download_file_bytes`] — streams a file and returns the raw bytes as a `Uint8Array`. Used by
//!   `examples/download.html` for the "Save file" button.
//! - [`download_with_progress`] — streams a file and forwards each [`hf_hub::progress::ProgressEvent`] to a JS callback
//!   as a plain object. Used by `examples/progress.html` to exercise the progress reporting pipeline end-to-end from a
//!   browser.
//!
//! See `README.md` and `examples/download.html` / `examples/progress.html` for the browser harness.

#![cfg(target_family = "wasm")]

use bytes::Bytes;
use futures_util::StreamExt;
use hf_hub::progress::{
    DownloadEvent, FileProgress, FileStatus, Progress, ProgressEvent, ProgressHandler, UploadEvent,
};
use hf_hub::repository::download::HFByteStream;
use hf_hub::{HFClient, HFClientBuilder, HFResult, RepoTypeAny};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

async fn open_stream(
    client: &HFClient,
    repo_type_plural: &str,
    owner: String,
    name: String,
    revision: String,
    filename: String,
    progress: Option<Progress>,
) -> HFResult<(Option<u64>, HFByteStream)> {
    let kind: RepoTypeAny = repo_type_plural.parse()?;
    client
        .repository(kind, owner, name)
        .download_file_stream()
        .filename(filename)
        .revision(revision)
        .maybe_progress(progress)
        .send()
        .await
}

fn build_client(endpoint: String, token: Option<String>) -> Result<HFClient, JsValue> {
    let mut builder = HFClientBuilder::new().endpoint(endpoint);
    if let Some(t) = token {
        builder = builder.token(t);
    }
    builder.build().map_err(|e| JsValue::from_str(&format!("{e}")))
}

/// Stream the bytes of a file from a public Hugging Face Hub repo and return
/// the total number of bytes seen.
///
/// `repo_type_plural` should be `"models"`, `"datasets"`, `"spaces"`, or
/// `"kernels"`. `owner` and `name` make up the `"owner/name"` repo id.
///
/// In a browser the caller is responsible for the COOP/COEP headers needed by
/// the threaded wasm in `hf-xet` (see this crate's `README.md`).
#[wasm_bindgen]
pub async fn smoke_stream_total_bytes(
    endpoint: String,
    token: Option<String>,
    repo_type_plural: String,
    owner: String,
    name: String,
    revision: String,
    filename: String,
) -> Result<f64, JsValue> {
    let client = build_client(endpoint, token)?;
    let (_len, stream) = open_stream(&client, &repo_type_plural, owner, name, revision, filename, None)
        .await
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    futures_util::pin_mut!(stream);

    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| JsValue::from_str(&format!("{e}")))?;
        total += chunk.len() as u64;
    }
    Ok(total as f64)
}

/// Stream the bytes of a file from a public Hugging Face Hub repo and return
/// them as a `Uint8Array`. Use with caution for large files — the full content
/// is buffered in memory before returning to JS.
///
/// Same argument shape as [`smoke_stream_total_bytes`].
#[wasm_bindgen]
pub async fn download_file_bytes(
    endpoint: String,
    token: Option<String>,
    repo_type_plural: String,
    owner: String,
    name: String,
    revision: String,
    filename: String,
) -> Result<js_sys::Uint8Array, JsValue> {
    let client = build_client(endpoint, token)?;
    let (content_length, stream) = open_stream(&client, &repo_type_plural, owner, name, revision, filename, None)
        .await
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    futures_util::pin_mut!(stream);

    let mut chunks: Vec<Bytes> = Vec::new();
    let mut total: usize = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| JsValue::from_str(&format!("{e}")))?;
        total += chunk.len();
        chunks.push(chunk);
    }

    let mut buf = Vec::with_capacity(content_length.map(|n| n as usize).unwrap_or(total));
    for chunk in chunks {
        buf.extend_from_slice(&chunk);
    }
    Ok(js_sys::Uint8Array::from(buf.as_slice()))
}

/// Stream the bytes of a file and forward every [`ProgressEvent`] to a JS
/// callback as a plain object.
///
/// The `on_progress` callback receives one argument per call — a plain object
/// whose `kind` field is one of `"download.start"`, `"download.progress"`,
/// `"download.aggregate_progress"`, `"download.complete"`,
/// `"upload.start"`, `"upload.progress"`, `"upload.committing"`, or
/// `"upload.complete"`. Numeric fields mirror the Rust enum variants;
/// per-file lists (`files`) are arrays of `{ filename, bytes_completed,
/// total_bytes, status }` objects.
///
/// This function only drives `download_file_stream`, so the callback only
/// observes `download.*` variants in this demo. Upload variants are exposed
/// in the JS shape for parity with native — the wasm-compatible upload
/// surface (`create_commit`, `upload_file`, `delete_file`, `delete_folder`
/// with `AddSource::Bytes`) does emit `upload.start` / `upload.committing` /
/// `upload.complete`, but per-byte `upload.progress` events are not wired
/// through the wasm xet upload path today (see `hf-hub/src/xet.rs`'s
/// `#[cfg(target_family = "wasm")] async fn xet_upload_inner`).
///
/// Returns the total number of bytes streamed.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn download_with_progress(
    endpoint: String,
    token: Option<String>,
    repo_type_plural: String,
    owner: String,
    name: String,
    revision: String,
    filename: String,
    on_progress: js_sys::Function,
) -> Result<f64, JsValue> {
    let client = build_client(endpoint, token)?;
    let handler = JsProgressHandler::new(on_progress);
    let progress: Progress = handler.into();
    let (_len, stream) = open_stream(&client, &repo_type_plural, owner, name, revision, filename, Some(progress))
        .await
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    futures_util::pin_mut!(stream);

    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| JsValue::from_str(&format!("{e}")))?;
        total += chunk.len() as u64;
    }
    Ok(total as f64)
}

/// Bridges hf-hub's [`ProgressHandler`] (which requires `Send + Sync`) to a
/// `js_sys::Function`.
///
/// `js_sys::Function` is `!Send + !Sync` because each JS value is bound to its
/// host thread. The `Send + Sync` impls below are sound on
/// `wasm32-unknown-unknown` here for two reasons:
///
/// 1. The hf-xet runtime spawns Web Workers, but each worker has its own isolated JS heap — the callback would not be
///    reachable from there even if we tried to move it.
/// 2. Progress events for `download_file_stream` are emitted on the stream-poll path (`wrap_stream_with_progress`),
///    which runs on the same JS thread that calls `download_with_progress`. The callback is invoked on its original
///    thread.
struct JsProgressHandler {
    callback: js_sys::Function,
}

impl JsProgressHandler {
    fn new(callback: js_sys::Function) -> Self {
        Self { callback }
    }
}

// SAFETY: see the type-level doc comment.
unsafe impl Send for JsProgressHandler {}
// SAFETY: see the type-level doc comment.
unsafe impl Sync for JsProgressHandler {}

impl ProgressHandler for JsProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let payload = event_to_js(event);
        // Ignore errors thrown by the JS callback — they should not abort the download.
        let _ = self.callback.call1(&JsValue::NULL, &payload);
    }
}

fn event_to_js(event: &ProgressEvent) -> JsValue {
    let obj = Object::new();
    match event {
        ProgressEvent::Download(DownloadEvent::Start {
            total_files,
            total_bytes,
        }) => {
            set_kind(&obj, "download.start");
            set_number(&obj, "total_files", *total_files as f64);
            set_number(&obj, "total_bytes", *total_bytes as f64);
        },
        ProgressEvent::Download(DownloadEvent::Progress { files }) => {
            set_kind(&obj, "download.progress");
            set(&obj, "files", &files_to_js(files));
        },
        ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed,
            total_bytes,
            bytes_per_sec,
        }) => {
            set_kind(&obj, "download.aggregate_progress");
            set_number(&obj, "bytes_completed", *bytes_completed as f64);
            set_number(&obj, "total_bytes", *total_bytes as f64);
            set_optional_number(&obj, "bytes_per_sec", *bytes_per_sec);
        },
        ProgressEvent::Download(DownloadEvent::Complete) => {
            set_kind(&obj, "download.complete");
        },
        ProgressEvent::Upload(UploadEvent::Start {
            total_files,
            total_bytes,
        }) => {
            set_kind(&obj, "upload.start");
            set_number(&obj, "total_files", *total_files as f64);
            set_number(&obj, "total_bytes", *total_bytes as f64);
        },
        ProgressEvent::Upload(UploadEvent::Progress {
            bytes_completed,
            total_bytes,
            bytes_per_sec,
            transfer_bytes_completed,
            transfer_bytes,
            transfer_bytes_per_sec,
            files,
        }) => {
            set_kind(&obj, "upload.progress");
            set_number(&obj, "bytes_completed", *bytes_completed as f64);
            set_number(&obj, "total_bytes", *total_bytes as f64);
            set_optional_number(&obj, "bytes_per_sec", *bytes_per_sec);
            set_number(&obj, "transfer_bytes_completed", *transfer_bytes_completed as f64);
            set_number(&obj, "transfer_bytes", *transfer_bytes as f64);
            set_optional_number(&obj, "transfer_bytes_per_sec", *transfer_bytes_per_sec);
            set(&obj, "files", &files_to_js(files));
        },
        ProgressEvent::Upload(UploadEvent::Committing) => {
            set_kind(&obj, "upload.committing");
        },
        ProgressEvent::Upload(UploadEvent::Complete) => {
            set_kind(&obj, "upload.complete");
        },
    }
    obj.into()
}

fn files_to_js(files: &[FileProgress]) -> JsValue {
    let arr = Array::new_with_length(files.len() as u32);
    for (i, file) in files.iter().enumerate() {
        let entry = Object::new();
        set(&entry, "filename", &JsValue::from_str(&file.filename));
        set_number(&entry, "bytes_completed", file.bytes_completed as f64);
        set_number(&entry, "total_bytes", file.total_bytes as f64);
        let status = match file.status {
            FileStatus::Started => "started",
            FileStatus::InProgress => "in_progress",
            FileStatus::Complete => "complete",
        };
        set(&entry, "status", &JsValue::from_str(status));
        arr.set(i as u32, entry.into());
    }
    arr.into()
}

fn set_kind(obj: &Object, kind: &str) {
    set(obj, "kind", &JsValue::from_str(kind));
}

fn set_number(obj: &Object, key: &str, value: f64) {
    set(obj, key, &JsValue::from_f64(value));
}

fn set_optional_number(obj: &Object, key: &str, value: Option<f64>) {
    match value {
        Some(v) => set(obj, key, &JsValue::from_f64(v)),
        None => set(obj, key, &JsValue::NULL),
    }
}

fn set(obj: &Object, key: &str, value: &JsValue) {
    // Reflect::set on a fresh Object only fails if the key is non-coercible to a string,
    // which our static &str keys are not.
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}
