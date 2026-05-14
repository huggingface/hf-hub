//! `hf-hub` wasm32 smoke verification.
//!
//! This crate exists solely to prove that the wasm-safe subset of `hf-hub`
//! (HFClient + `HFRepository::download_file_stream`) compiles and is
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
use hf_hub::{HFClientBuilder, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace};
use wasm_bindgen::prelude::*;

/// Stream the bytes of a file from a public Hugging Face Hub repo and return
/// the total number of bytes seen.
///
/// `repo_type_plural` should be `"models"`, `"datasets"`, `"spaces"`, or
/// `"kernels"`. `owner` and `name` make up the `"owner/name"` repo id.
///
/// In a browser the caller is responsible for the COOP/COEP headers needed by
/// the threaded wasm in `hf-xet` (see the xet-core
/// `wasm/hf_xet_wasm_download/README.md`).
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
    let mut builder = HFClientBuilder::new().endpoint(endpoint);
    if let Some(t) = token {
        builder = builder.token(t);
    }
    let client = builder.build().map_err(|e| JsValue::from_str(&format!("{e}")))?;

    let (_len, stream) = match repo_type_plural.as_str() {
        "models" => client
            .repository::<RepoTypeModel>(owner, name)
            .download_file_stream()
            .filename(filename)
            .revision(revision)
            .send()
            .await
            .map_err(|e| JsValue::from_str(&format!("{e}")))?,
        "datasets" => client
            .repository::<RepoTypeDataset>(owner, name)
            .download_file_stream()
            .filename(filename)
            .revision(revision)
            .send()
            .await
            .map_err(|e| JsValue::from_str(&format!("{e}")))?,
        "spaces" => client
            .repository::<RepoTypeSpace>(owner, name)
            .download_file_stream()
            .filename(filename)
            .revision(revision)
            .send()
            .await
            .map_err(|e| JsValue::from_str(&format!("{e}")))?,
        "kernels" => client
            .repository::<RepoTypeKernel>(owner, name)
            .download_file_stream()
            .filename(filename)
            .revision(revision)
            .send()
            .await
            .map_err(|e| JsValue::from_str(&format!("{e}")))?,
        other => return Err(JsValue::from_str(&format!("unknown repo_type_plural: {other}"))),
    };
    futures_util::pin_mut!(stream);

    let mut total: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| JsValue::from_str(&format!("{e}")))?;
        total += chunk.len() as u64;
    }
    Ok(total as f64)
}
