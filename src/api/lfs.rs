use crate::api::tokio::ApiError;
use reqwest::header::HeaderMap;
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Semaphore;

const UPLOAD_CHUNK_SIZE: usize = 512;
const HASH_READ_BUF_SIZE: usize = 64 * 1024;

pub(crate) struct UploadInfo {
    pub sha256: [u8; 32],
    pub size: u64,
    /// First 512 bytes for content-type detection by the server.
    pub sample: Vec<u8>,
}

impl UploadInfo {
    /// Compute upload info (SHA256, size, sample) for a file.
    /// All I/O runs in a single spawn_blocking task to avoid per-chunk scheduling overhead.
    pub async fn from_path(path: &Path) -> Result<UploadInfo, ApiError> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || Self::compute_sync(&path))
            .await
            .map_err(|e| ApiError::InvalidApiResponse(format!("spawn_blocking join error: {e}")))?
    }

    fn compute_sync(path: &Path) -> Result<UploadInfo, ApiError> {
        use std::io::{Read, Seek};

        let mut file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();

        // Read first 512 bytes for sample
        let sample_len = std::cmp::min(UPLOAD_CHUNK_SIZE, size as usize);
        let mut sample = vec![0u8; sample_len];
        file.read_exact(&mut sample)?;

        // Stream SHA256 computation in 64KB chunks
        file.rewind()?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; HASH_READ_BUF_SIZE];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        Ok(UploadInfo {
            sha256: hasher.finalize().into(),
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

/// Maximum number of concurrent file hashing tasks.
const MAX_CONCURRENT_HASH_TASKS: usize = 8;

/// Compute UploadInfo for multiple files concurrently, limiting parallelism
/// with a semaphore to avoid overwhelming the blocking thread pool.
pub(crate) async fn compute_upload_infos(
    paths: &[(usize, PathBuf)],
) -> Result<Vec<(usize, UploadInfo)>, ApiError> {
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_HASH_TASKS));
    let mut handles = Vec::with_capacity(paths.len());

    for (idx, path) in paths {
        let idx = *idx;
        let path = path.clone();
        let permit = semaphore.clone().acquire_owned().await?;
        handles.push(tokio::spawn(async move {
            let result = UploadInfo::from_path(&path).await;
            drop(permit);
            result.map(|info| (idx, info))
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        results.push(handle.await.map_err(|e| {
            ApiError::InvalidApiResponse(format!("task join error: {e}"))
        })??);
    }
    Ok(results)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[allow(dead_code)]
pub(crate) struct LfsBatchResponse {
    pub transfer: String,
    pub objects: Vec<LfsObject>,
}

#[allow(dead_code)]
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
