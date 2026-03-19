use crate::api::tokio::ApiError;
use reqwest::header::HeaderMap;
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::Path;

const UPLOAD_CHUNK_SIZE: usize = 512;

pub(crate) struct UploadInfo {
    pub sha256: [u8; 32],
    pub size: u64,
    /// First 512 bytes for content-type detection by the server.
    pub sample: Vec<u8>,
}

impl UploadInfo {
    pub async fn from_path(path: &Path) -> Result<UploadInfo, ApiError> {
        let data = tokio::fs::read(path).await?;
        let size = data.len() as u64;
        let sample = data[..std::cmp::min(UPLOAD_CHUNK_SIZE, data.len())].to_vec();
        let sha256: [u8; 32] = Sha256::digest(&data).into();
        Ok(UploadInfo {
            sha256,
            size,
            sample,
        })
    }

    pub fn sha256_hex(&self) -> String {
        hex_encode(&self.sha256)
    }

    pub fn sample_base64(&self) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(&self.sample)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) struct LfsBatchResponse {
    pub transfer: String,
    pub objects: Vec<LfsObject>,
}

pub(crate) struct LfsObject {
    pub oid: String,
    pub size: u64,
}

/// POST /{repo_url}.git/info/lfs/objects/batch
///
/// Sends objects with their OID and size, requesting "upload" operation.
/// Offers ["basic", "multipart", "xet"] transfer protocols.
/// Returns the server-chosen transfer and per-object info.
pub(crate) async fn post_lfs_batch_info(
    client: &Client,
    endpoint: &str,
    repo_url: &str,
    headers: &HeaderMap,
    upload_infos: &[&UploadInfo],
) -> Result<LfsBatchResponse, ApiError> {
    let url = format!("{endpoint}/{repo_url}.git/info/lfs/objects/batch");

    let objects: Vec<serde_json::Value> = upload_infos
        .iter()
        .map(|info| {
            serde_json::json!({
                "oid": info.sha256_hex(),
                "size": info.size,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "operation": "upload",
        "transfers": ["basic", "multipart", "xet"],
        "objects": objects,
        "hash_algo": "sha256",
    });

    let response = client
        .post(&url)
        .headers(headers.clone())
        .header("Content-Type", "application/vnd.git-lfs+json")
        .header("Accept", "application/vnd.git-lfs+json")
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    let body: serde_json::Value = response.json().await?;

    let transfer = body["transfer"]
        .as_str()
        .unwrap_or("basic")
        .to_string();

    let objects = body["objects"]
        .as_array()
        .ok_or_else(|| ApiError::InvalidApiResponse("missing objects in LFS batch response".into()))?
        .iter()
        .map(|obj| LfsObject {
            oid: obj["oid"].as_str().unwrap_or_default().to_string(),
            size: obj["size"].as_u64().unwrap_or_default(),
        })
        .collect();

    Ok(LfsBatchResponse { transfer, objects })
}
