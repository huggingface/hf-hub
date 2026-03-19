use crate::api::lfs::UploadInfo;
use crate::api::tokio::ApiError;
use reqwest::header::HeaderMap;
use reqwest::Client;
use serde::Deserialize;
use std::path::PathBuf;

const PREUPLOAD_CHUNK_SIZE: usize = 256;

/// What to do in a commit.
pub enum CommitOperation {
    /// Add a file to the repo.
    Add(CommitOperationAdd),
    /// Delete a file from the repo.
    Delete(CommitOperationDelete),
    /// Copy a file within the repo.
    Copy(CommitOperationCopy),
}

/// Add a file to the repository.
pub struct CommitOperationAdd {
    /// The path in the repository where the file will be placed.
    pub path_in_repo: String,
    /// The local path to the file to upload.
    pub local_path: PathBuf,
    // Internal fields set during the upload pipeline
    pub(crate) upload_info: Option<UploadInfo>,
    pub(crate) upload_mode: Option<UploadMode>,
    pub(crate) should_ignore: bool,
    pub(crate) remote_oid: Option<String>,
}

impl CommitOperationAdd {
    /// Create a new add operation.
    pub fn new(path_in_repo: String, local_path: PathBuf) -> Self {
        Self {
            path_in_repo,
            local_path,
            upload_info: None,
            upload_mode: None,
            should_ignore: false,
            remote_oid: None,
        }
    }
}

/// Delete a file from the repository.
pub struct CommitOperationDelete {
    /// The path of the file to delete in the repository.
    pub path_in_repo: String,
}

/// Copy a file within the repository.
pub struct CommitOperationCopy {
    /// Source path in the repository.
    pub src_path_in_repo: String,
    /// Destination path in the repository.
    pub dest_path_in_repo: String,
    /// Optional source revision (defaults to the commit's revision).
    pub src_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UploadMode {
    Regular,
    Lfs,
}

/// Response from the create commit endpoint.
#[derive(Debug, Deserialize)]
pub struct CommitResponse {
    /// The commit hash of the new commit.
    #[serde(rename = "commitOid")]
    pub commit_id: String,
    /// The URL of the new commit.
    #[serde(rename = "commitUrl")]
    pub commit_url: String,
}

/// POST /api/{repo_type}s/{repo_id}/preupload/{revision}
///
/// Sends file paths, sizes, and SHA256 samples in chunks of 256.
/// Server returns "regular" or "lfs" per file, plus shouldIgnore and oid.
/// Mutates each CommitOperationAdd with its upload_mode.
pub(crate) async fn fetch_upload_modes(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    additions: &mut [CommitOperationAdd],
    create_pr: bool,
) -> Result<(), ApiError> {
    // Process in chunks of PREUPLOAD_CHUNK_SIZE
    for chunk_start in (0..additions.len()).step_by(PREUPLOAD_CHUNK_SIZE) {
        let chunk_end = std::cmp::min(chunk_start + PREUPLOAD_CHUNK_SIZE, additions.len());
        let chunk = &additions[chunk_start..chunk_end];

        let files: Vec<serde_json::Value> = chunk
            .iter()
            .map(|add| {
                let info = add.upload_info.as_ref().expect("upload_info must be set before fetch_upload_modes");
                serde_json::json!({
                    "path": add.path_in_repo,
                    "size": info.size,
                    "sample": info.sample_base64(),
                })
            })
            .collect();

        let mut url = format!("{endpoint}/api/{repo_type}/{repo_id}/preupload/{revision}");
        if create_pr {
            url.push_str("?create_pr=1");
        }

        let payload = serde_json::json!({ "files": files });

        let response = client
            .post(&url)
            .headers(headers.clone())
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;

        let body: serde_json::Value = response.json().await?;
        let response_files = body["files"]
            .as_array()
            .ok_or_else(|| ApiError::InvalidApiResponse("missing files in preupload response".into()))?;

        for (i, file_resp) in response_files.iter().enumerate() {
            let idx = chunk_start + i;
            let upload_mode = match file_resp["uploadMode"].as_str() {
                Some("lfs") => UploadMode::Lfs,
                _ => UploadMode::Regular,
            };
            additions[idx].upload_mode = Some(upload_mode);
            additions[idx].should_ignore = file_resp["shouldIgnore"].as_bool().unwrap_or(false);
            additions[idx].remote_oid = file_resp["oid"].as_str().map(|s| s.to_string());
        }
    }

    Ok(())
}

/// POST /api/{repo_type}s/{repo_id}/commit/{revision}
///
/// Sends the final commit payload with:
/// - regular files inline (base64 content)
/// - LFS files as pointer metadata (after xet upload)
/// - deletions and copies as operations
pub(crate) async fn create_commit_request(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    commit_message: &str,
    commit_description: Option<&str>,
    operations: Vec<CommitOperation>,
    create_pr: bool,
    parent_commit: Option<&str>,
) -> Result<CommitResponse, ApiError> {
    let mut url = format!("{endpoint}/api/{repo_type}/{repo_id}/commit/{revision}");
    if create_pr {
        url.push_str("?create_pr=1");
    }

    let mut ops_json = Vec::new();

    for op in &operations {
        match op {
            CommitOperation::Add(add) => {
                if add.should_ignore {
                    continue;
                }
                match add.upload_mode.as_ref() {
                    Some(UploadMode::Lfs) => {
                        let info = add.upload_info.as_ref()
                            .expect("upload_info must be set for LFS files");
                        // LFS pointer metadata
                        let mut lfs_info = serde_json::json!({
                            "key": "lfsFile",
                            "value": {
                                "path": add.path_in_repo,
                                "algo": "sha256",
                                "oid": info.sha256_hex(),
                                "size": info.size,
                            }
                        });
                        // If we have a remote_oid from xet upload, include it
                        if let Some(ref remote_oid) = add.remote_oid {
                            lfs_info["value"]["xetHash"] = serde_json::Value::String(remote_oid.clone());
                        }
                        ops_json.push(lfs_info);
                    }
                    _ => {
                        // Regular file: inline base64 content
                        let content = tokio::fs::read(&add.local_path).await?;
                        use base64::Engine;
                        let encoded = base64::engine::general_purpose::STANDARD.encode(&content);
                        ops_json.push(serde_json::json!({
                            "key": "file",
                            "value": {
                                "content": encoded,
                                "path": add.path_in_repo,
                                "encoding": "base64",
                            }
                        }));
                    }
                }
            }
            CommitOperation::Delete(del) => {
                ops_json.push(serde_json::json!({
                    "key": "deletedFile",
                    "value": {
                        "path": del.path_in_repo,
                    }
                }));
            }
            CommitOperation::Copy(copy) => {
                let mut value = serde_json::json!({
                    "src": copy.src_path_in_repo,
                    "dest": copy.dest_path_in_repo,
                });
                if let Some(ref rev) = copy.src_revision {
                    value["srcRevision"] = serde_json::Value::String(rev.clone());
                }
                ops_json.push(serde_json::json!({
                    "key": "copiedFile",
                    "value": value,
                }));
            }
        }
    }

    let mut payload = serde_json::json!({
        "summary": commit_message,
        "operations": ops_json,
    });

    if let Some(desc) = commit_description {
        payload["description"] = serde_json::Value::String(desc.to_string());
    }
    if let Some(parent) = parent_commit {
        payload["parentCommit"] = serde_json::Value::String(parent.to_string());
    }

    let response = client
        .post(&url)
        .headers(headers.clone())
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(ApiError::HubApiError {
            status: status.as_u16(),
            message: body,
        });
    }

    let commit_response: CommitResponse = response.json().await?;
    Ok(commit_response)
}
