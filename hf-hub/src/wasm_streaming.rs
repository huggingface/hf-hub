//! Minimal wasm-compatible streaming download surface.
//!
//! The full `hf-hub` API is filesystem-heavy (cache, snapshot download,
//! bucket sync, blocking wrappers) and is gated off on `wasm32-unknown-unknown`.
//! This module exposes one function — [`xet_stream_file`] — that takes an
//! `HFClient`, resolves a file's xet hash + read token through the standard
//! Hub endpoints, and yields the file's bytes via the xet streaming download
//! path provided by `hf-xet`.
//!
//! The pattern mirrors `wasm/hf_xet_wasm_download/` in the `xet-core` repo:
//! pure-HTTP metadata calls + xet's `XetDownloadStreamGroup`.
//!
//! No filesystem APIs are touched here, so this compiles on
//! `wasm32-unknown-unknown` (with a browser `reqwest` backend).

use std::ops::Range;

use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use xet::xet_session::{XetFileInfo, XetSessionBuilder};

use crate::client::HFClient;
use crate::error::{HFError, HFResult, NotFoundContext, XetOperation};

#[derive(Debug, Serialize)]
struct PathsInfoRequest<'a> {
    paths: Vec<&'a str>,
    expand: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PathsInfoEntry {
    path: String,
    size: Option<u64>,
    xet_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XetReadTokenResponse {
    access_token: String,
    exp: u64,
    cas_url: String,
}

/// Stream a single file's bytes from the Hub through the xet streaming download
/// path.
///
/// Mirrors how the browser-side `wasm/hf_xet_wasm_download/examples/download.html`
/// works: hit `paths-info` to resolve the `xetHash`/`size`, hit `xet-read-token`
/// to mint a CAS token, then drive `xet::xet_session::XetDownloadStreamGroup`.
///
/// `repo_type_plural` must be `"models"`, `"datasets"`, or `"spaces"` —
/// the plural form used in the Hub API path. `repo_id` is `"namespace/repo"`.
/// If `range` is `Some`, only that byte range is streamed.
pub async fn xet_stream_file(
    client: &HFClient,
    repo_type_plural: &str,
    repo_id: &str,
    revision: &str,
    filename: &str,
    range: Option<Range<u64>>,
) -> HFResult<impl Stream<Item = HFResult<Bytes>>> {
    let (xet_hash, file_size) = fetch_xet_file_id(client, repo_type_plural, repo_id, revision, filename).await?;
    let conn = fetch_xet_read_token(client, repo_type_plural, repo_id, revision).await?;

    let session = XetSessionBuilder::new()
        .build()
        .map_err(|e| HFError::xet(XetOperation::Session, e))?;

    let group = session
        .new_download_stream_group()
        .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?
        .with_endpoint(conn.cas_url.clone())
        .with_token_info(conn.access_token.clone(), conn.exp)
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;

    let file_info = XetFileInfo::new(xet_hash, file_size);
    let mut stream = group
        .download_stream(file_info, range)
        .await
        .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;
    stream.start();

    Ok(futures::stream::unfold(stream, |mut stream| async move {
        match stream.next().await {
            Ok(Some(bytes)) => Some((Ok(bytes), stream)),
            Ok(None) => None,
            Err(e) => Some((Err(HFError::xet(XetOperation::StreamDownload, e)), stream)),
        }
    }))
}

async fn fetch_xet_file_id(
    client: &HFClient,
    repo_type_plural: &str,
    repo_id: &str,
    revision: &str,
    filename: &str,
) -> HFResult<(String, u64)> {
    let url = format!("{}/api/{}/{}/paths-info/{}", client.endpoint(), repo_type_plural, repo_id, revision);
    let body = PathsInfoRequest {
        paths: vec![filename],
        expand: false,
    };
    let response = client
        .http_client()
        .post(&url)
        .headers(client.auth_headers())
        .json(&body)
        .send()
        .await?;
    let response = client.check_response(response, Some(repo_id), NotFoundContext::Repo).await?;
    let entries: Vec<PathsInfoEntry> = response.json().await?;
    let entry = entries
        .into_iter()
        .find(|e| e.path == filename)
        .ok_or_else(|| HFError::malformed_response_at("paths-info: file not in response", url.clone()))?;
    let hash = entry
        .xet_hash
        .ok_or_else(|| HFError::malformed_response_at("paths-info: file is not xet-backed", url.clone()))?;
    let size = entry
        .size
        .ok_or_else(|| HFError::malformed_response_at("paths-info: missing size", url))?;
    Ok((hash, size))
}

async fn fetch_xet_read_token(
    client: &HFClient,
    repo_type_plural: &str,
    repo_id: &str,
    revision: &str,
) -> HFResult<XetReadTokenResponse> {
    let url = format!("{}/api/{}/{}/xet-read-token/{}", client.endpoint(), repo_type_plural, repo_id, revision);
    let response = client.http_client().get(&url).headers(client.auth_headers()).send().await?;
    let response = client.check_response(response, Some(repo_id), NotFoundContext::Repo).await?;
    let token: XetReadTokenResponse = response.json().await?;
    Ok(token)
}
