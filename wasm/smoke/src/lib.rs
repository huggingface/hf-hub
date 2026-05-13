//! `hf-hub` wasm32 smoke verification.
//!
//! This crate exists solely to prove that the wasm-safe subset of `hf-hub`
//! (HFClient + the `wasm_streaming::xet_stream_file` helper) compiles and is
//! callable in a wasm-bindgen environment.
//!
//! Build with: `wasm-pack build wasm/smoke --target web`
//!
//! It is intentionally a no-op at runtime — the function it exports performs
//! exactly enough work to force the compiler to monomorphize every type
//! through the xet streaming download path that this PR's `hf-xet` git
//! dependency makes wasm-compatible.

#![cfg(target_family = "wasm")]

use futures_util::StreamExt;
use hf_hub::HFClientBuilder;
use hf_hub::wasm_streaming::xet_stream_file;
use wasm_bindgen::prelude::*;

/// Stream the bytes of a xet-backed file from a public Hugging Face Hub repo
/// and return the total number of bytes seen.
///
/// `repo_type_plural` should be `"models"`, `"datasets"`, or `"spaces"`.
/// `repo_id` is `"namespace/name"`.
///
/// In a browser the caller is responsible for the COOP/COEP headers needed by
/// the threaded wasm in `hf-xet` (see the xet-core
/// `wasm/hf_xet_wasm_download/README.md`).
#[wasm_bindgen]
pub async fn smoke_stream_total_bytes(
    endpoint: String,
    token: Option<String>,
    repo_type_plural: String,
    repo_id: String,
    revision: String,
    filename: String,
) -> Result<f64, JsValue> {
    let mut builder = HFClientBuilder::new().endpoint(endpoint);
    if let Some(t) = token {
        builder = builder.token(t);
    }
    let client = builder.build().map_err(|e| JsValue::from_str(&format!("{e}")))?;

    let stream = xet_stream_file(&client, &repo_type_plural, &repo_id, &revision, &filename, None)
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
