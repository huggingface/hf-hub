//! `hf-hub` wasm32 example + smoke verification.
//!
//! Builds for `wasm32-unknown-unknown` and proves that the wasm-safe subset of
//! `hf-hub` (`HFClient`, `HFRepository::download_file_stream`) compiles and is
//! callable in a wasm-bindgen environment.
//!
//! Exports two `#[wasm_bindgen]` functions:
//! - [`smoke_stream_total_bytes`] — streams a file and returns the total bytes seen. Used by `examples/download.html`
//!   for progress reporting and by `scripts/verify_wasm.sh` as a monomorphization smoke check.
//! - [`download_file_bytes`] — streams a file and returns the raw bytes as a `Uint8Array`. Used by
//!   `examples/download.html` for the "Save file" button.
//!
//! See `README.md` and `examples/download.html` for the browser harness.

#![cfg(target_family = "wasm")]

use bytes::Bytes;
use futures_util::StreamExt;
use hf_hub::repository::download::HFByteStream;
use hf_hub::{HFClient, HFClientBuilder, HFResult, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace};
use wasm_bindgen::prelude::*;

async fn open_stream(
    client: &HFClient,
    repo_type_plural: &str,
    owner: String,
    name: String,
    revision: String,
    filename: String,
) -> HFResult<(Option<u64>, HFByteStream)> {
    match repo_type_plural {
        "models" => {
            client
                .repository::<RepoTypeModel>(owner, name)
                .download_file_stream()
                .filename(filename)
                .revision(revision)
                .send()
                .await
        },
        "datasets" => {
            client
                .repository::<RepoTypeDataset>(owner, name)
                .download_file_stream()
                .filename(filename)
                .revision(revision)
                .send()
                .await
        },
        "spaces" => {
            client
                .repository::<RepoTypeSpace>(owner, name)
                .download_file_stream()
                .filename(filename)
                .revision(revision)
                .send()
                .await
        },
        "kernels" => {
            client
                .repository::<RepoTypeKernel>(owner, name)
                .download_file_stream()
                .filename(filename)
                .revision(revision)
                .send()
                .await
        },
        other => Err(hf_hub::HFError::InvalidParameter(format!("unknown repo_type_plural: {other}"))),
    }
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
    let (_len, stream) = open_stream(&client, &repo_type_plural, owner, name, revision, filename)
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
    let (content_length, stream) = open_stream(&client, &repo_type_plural, owner, name, revision, filename)
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
