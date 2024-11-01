use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use lazy_static::lazy_static;
use log::warn;
use regex::Regex;
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::fs::{read_to_string, File};
use tokio::io::{self, AsyncRead, AsyncReadExt, BufReader};

use crate::api::tokio::upload::lfs::lfs_upload;
use crate::api::tokio::{ApiError, ApiRepo, HfBadResponse};

use super::commit_info::{CommitInfo, InvalidHfIdError};

const CHUNK_SIZE: usize = 8192; // 8KB chunks for streaming
const SAMPLE_SIZE: usize = 1024; // 1KB sample size

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadMode {
    Lfs,
    Regular,
}

#[derive(Debug, Serialize)]
struct PreuploadFile {
    path: String,
    sample: String,
    size: u64,
}

#[derive(Debug, Serialize)]
struct PreuploadRequest {
    files: Vec<PreuploadFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    git_ignore: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreuploadResponseFile {
    path: String,
    upload_mode: String,
    should_ignore: bool,
    oid: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PreuploadResponse {
    files: Vec<PreuploadResponseFile>,
}

pub struct UploadInfo {
    pub size: u64,
    pub sample: Vec<u8>,
    pub sha256: Vec<u8>,
}

impl Debug for UploadInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadInfo")
            .field("size", &self.size)
            .field("sample", &hex::encode(&self.sample))
            .field("sha256", &hex::encode(&self.sha256))
            .finish()
    }
}

async fn process_stream<R>(mut reader: R, size: u64) -> io::Result<UploadInfo>
where
    R: AsyncRead + Unpin,
{
    let mut sample = vec![0u8; SAMPLE_SIZE.min(size as usize)];
    reader.read_exact(&mut sample).await?;

    let mut hasher = Sha256::new();
    hasher.update(&sample); // Hash the sample bytes too
    let mut total_bytes = sample.len() as u64; // Start with `sample` size
    let mut buffer = vec![0u8; CHUNK_SIZE];

    loop {
        let bytes_read = reader.read(&mut buffer).await?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
        total_bytes += bytes_read as u64;
    }

    Ok(UploadInfo {
        size: total_bytes,
        sample,
        sha256: hasher.finalize().to_vec(),
    })
}

impl UploadInfo {
    pub async fn from_file(path: &Path) -> io::Result<Self> {
        let file = File::open(path).await?;
        let metadata = file.metadata().await?;
        let size = metadata.len();

        let reader = BufReader::with_capacity(CHUNK_SIZE, file);
        process_stream(reader, size).await
    }

    pub async fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        process_stream(cursor, bytes.len() as u64).await
    }
}

#[derive(Debug)]
pub struct CommitOperationAdd {
    pub path_in_repo: String,
    pub upload_info: UploadInfo,
    pub upload_mode: UploadMode,
    pub should_ignore: bool,
    pub remote_oid: Option<String>,
    // Store the source for streaming
    pub(crate) source: UploadSource,
}

/// Represents different sources for upload data.
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
///
/// let file_source = UploadSource::File(PathBuf::from("path/to/file.txt"));
/// let bytes_source = UploadSource::Bytes(vec![1, 2, 3, 4]);
/// let empty_source = UploadSource::Emptied;
/// ```
pub enum UploadSource {
    /// Contains a file path from which to read the upload data
    File(PathBuf),
    /// Contains the upload data directly as a byte vector
    Bytes(Vec<u8>),
    /// Represents a state where the upload source has been consumed or cleared
    Emptied,
}

impl Debug for UploadSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(arg0) => f.debug_tuple("File").field(arg0).finish(),
            Self::Bytes(arg0) => f
                .debug_tuple("Bytes")
                .field(&format!("{} bytes", arg0.len()))
                .finish(),
            Self::Emptied => write!(f, "Emptied"),
        }
    }
}

impl From<&Path> for UploadSource {
    fn from(value: &Path) -> Self {
        Self::File(value.into())
    }
}

impl From<PathBuf> for UploadSource {
    fn from(value: PathBuf) -> Self {
        Self::File(value)
    }
}

impl From<Vec<u8>> for UploadSource {
    fn from(value: Vec<u8>) -> Self {
        Self::Bytes(value)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommitData {
    commit_url: String,
    commit_oid: String,
    #[serde(default)]
    pull_request_url: Option<String>,
}

impl CommitOperationAdd {
    pub async fn from_file(path_in_repo: String, file_path: &Path) -> io::Result<Self> {
        let upload_info = UploadInfo::from_file(file_path).await?;
        Ok(Self {
            path_in_repo,
            upload_info,
            upload_mode: UploadMode::Regular,
            should_ignore: false,
            remote_oid: None,
            source: file_path.into(),
        })
    }

    pub async fn from_bytes(path_in_repo: String, bytes: Vec<u8>) -> io::Result<Self> {
        let upload_info = UploadInfo::from_bytes(&bytes).await?;
        Ok(Self {
            path_in_repo,
            upload_info,
            upload_mode: UploadMode::Regular,
            should_ignore: false,
            remote_oid: None,
            source: UploadSource::Bytes(bytes),
        })
    }
    pub async fn from_upload_source(
        path_in_repo: String,
        upload_source: UploadSource,
    ) -> io::Result<Self> {
        match upload_source {
            UploadSource::Emptied => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "upload source was empty.".to_string(),
            )),
            UploadSource::Bytes(bytes) => CommitOperationAdd::from_bytes(path_in_repo, bytes).await,
            UploadSource::File(file) => CommitOperationAdd::from_file(path_in_repo, &file).await,
        }
    }

    /// Return the OID of the local file.
    ///
    ///    This OID is then compared to `self._remote_oid` to check if the file has changed compared to the remote one.
    ///    If the file did not change, we won't upload it again to prevent empty commits.
    ///
    ///    For LFS files, the OID corresponds to the SHA256 of the file content (used a LFS ref).
    ///
    ///    For regular files, the OID corresponds to the SHA1 of the file content.
    ///
    ///    Note: this is slightly different to git OID computation since the oid of an LFS file is usually the git-SHA1 of the
    ///          pointer file content (not the actual file content). However, using the SHA256 is enough to detect changes
    ///          and more convenient client-side.
    pub fn local_oid(&self) -> Option<String> {
        match self.upload_mode {
            UploadMode::Lfs => Some(hex::encode(&self.upload_info.sha256)),
            UploadMode::Regular => {
                let data = match &self.source {
                    UploadSource::Bytes(b) => b,
                    UploadSource::Emptied => return None,
                    UploadSource::File(f) => &fs::read(f).unwrap(),
                };
                Some(git_hash(data))
            }
        }
    }
}

lazy_static! {
    static ref REGEX_COMMIT_OID: Regex = Regex::new(r"[A-Fa-f0-9]{5,40}").unwrap();
}

#[derive(Debug)]
pub enum CommitOperation {
    Add(CommitOperationAdd),
}

impl From<CommitOperationAdd> for CommitOperation {
    fn from(value: CommitOperationAdd) -> Self {
        Self::Add(value)
    }
}

#[derive(Debug, Error)]
pub enum CommitError {
    #[error("no commit message passed")]
    NoMessage,
    #[error("invalid OID for parent commit")]
    InvalidOid,
    #[error("failed to parse huggingface ID: {0}")]
    InvalidHuggingFaceId(#[from] InvalidHfIdError),
    #[error("error from HF api: {0}")]
    Api(#[from] ApiError),
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
}

impl ApiRepo {
    /// Creates a commit in the given repo, deleting & uploading files as needed.
    ///
    /// # Arguments
    ///
    /// * `operations` - Vector of operations to include in the commit (Add, Delete, Copy)
    /// * `commit_message` - The summary (first line) of the commit
    /// * `commit_description` - Optional description of the commit
    /// * `revision` - The git revision to commit from (defaults to "main")
    /// * `create_pr` - Whether to create a Pull Request
    /// * `num_threads` - Number of concurrent threads for uploading files
    /// * `parent_commit` - The OID/SHA of the parent commit
    ///
    /// # Returns
    ///
    /// Returns CommitInfo containing information about the newly created commit
    pub async fn create_commit(
        &self,
        operations: Vec<CommitOperation>,
        commit_message: String,
        commit_description: Option<String>,
        create_pr: Option<bool>,
        num_threads: Option<usize>,
        parent_commit: Option<String>,
    ) -> Result<CommitInfo, CommitError> {
        // Validate inputs
        if commit_message.is_empty() {
            return Err(CommitError::NoMessage);
        }

        if let Some(parent) = &parent_commit {
            if !REGEX_COMMIT_OID.is_match(parent) {
                return Err(CommitError::InvalidOid);
            }
        }

        log::trace!(
            "create_commit got {} operations: {:?}",
            operations.len(),
            operations
        );

        let commit_description = commit_description.unwrap_or_default();
        let create_pr = create_pr.unwrap_or(false);
        let num_threads = num_threads.unwrap_or(5);

        // Warn on overwriting operations
        warn_on_overwriting_operations(&operations);

        // Split operations by type
        let additions: Vec<_> = operations
            .into_iter()
            .map(|op| match op {
                CommitOperation::Add(add) => add,
            })
            .collect();

        // todo copy
        // let copies: Vec<_> = operations
        //     .iter()
        //     .filter_map(|op| match op {
        //         // todo one day
        //         // CommitOperation::Copy(copy) => Some(copy),
        //         _ => None,
        //     })
        //     .collect();
        // let deletions = operations.len() - additions.len() - copies.len();

        log::debug!(
            "About to commit to the hub: {} addition(s), {} copie(s) and {} deletion(s).",
            additions.len(),
            0,
            0 // copies.len(),
              // deletions
        );

        // TODO Validate README.md metadata if present

        // Pre-upload LFS files
        let additions = self
            .preupload_lfs_files(additions, Some(create_pr), Some(num_threads), None)
            .await
            .map_err(CommitError::Api)?;

        // re-collect into operations, after lfs upload.
        let operations: Vec<CommitOperation> = additions.into_iter().map(|a| a.into()).collect();
        log::trace!(
            "after preuploading lfs files, have {} operations: {:?}",
            operations.len(),
            operations
        );
        // Remove no-op operations
        let operations: Vec<_> = operations
            .into_iter()
            .filter(|op| match op {
                CommitOperation::Add(add) => {
                    if let (Some(remote_oid), Some(local_oid)) = (&add.remote_oid, &add.local_oid())
                    {
                        if remote_oid == local_oid {
                            log::debug!(
                                "Skipping upload for '{}' as the file has not changed.",
                                add.path_in_repo
                            );
                            return false;
                        }
                    }
                    true
                }
            })
            .collect();

        if operations.is_empty() {
            log::warn!(
                "No files have been modified since last commit. Skipping to prevent empty commit."
            );
            // Return latest commit info
            let info = self.repo_info().await?;
            let sha = info
                .sha()
                .ok_or(CommitError::Api(ApiError::InvalidResponse(
                    "no SHA returned from repo info".to_string(),
                )))?
                .to_string();
            return Ok(CommitInfo::new(
                &format!("{}/{}/commit/{}", self.api.endpoint, self.repo.repo_id, sha),
                &commit_description,
                &commit_message,
                sha,
            )?);
        }

        // Prepare and send commit
        // TODO add copy
        // let files_to_copy = self.fetch_files_to_copy(&copies, &revision).await?;

        let commit_payload = prepare_commit_payload(
            &operations,
            // TODO: &files_to_copy,
            &commit_message,
            &commit_description,
            parent_commit.as_deref(),
        );
        log::trace!(
            "commit payload: {}",
            serde_json::to_string(&commit_payload).unwrap()
        );

        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/x-ndjson".parse().unwrap());

        let url = format!(
            "{}/api/{}s/{}/commit/{}",
            self.api.endpoint,
            self.repo.repo_type.to_string(),
            self.repo.url(),
            self.repo.revision
        );
        let mut params = HashMap::new();
        if create_pr {
            params.insert("create_pr", "1");
        }

        let serialized_payload: Vec<u8> = payload_as_ndjson(commit_payload).flatten().collect();

        let response = self
            .api
            .client
            .post(&url)
            .headers(headers)
            .query(&params)
            .body(serialized_payload)
            .send()
            .await
            .map_err(ApiError::from)?
            .maybe_hf_err()
            .await
            .map_err(ApiError::from)?;

        let commit_data: CommitData = response.json().await.map_err(|e| {
            CommitError::Api(ApiError::InvalidResponse(format!(
                "Failed to parse json from commit API: {e}"
            )))
        })?;
        let mut commit_info = CommitInfo::new(
            &commit_data.commit_url,
            &commit_description,
            &commit_message,
            commit_data.commit_oid,
        )
        .map_err(|e| {
            CommitError::Api(ApiError::InvalidResponse(format!(
                "Bad commit data returned from API: {e}"
            )))
        })?;

        if create_pr {
            if let Some(pr_url) = commit_data.pull_request_url {
                commit_info.set_pr_info(&pr_url).map_err(|_| {
                    CommitError::Api(ApiError::InvalidResponse(format!(
                        "Invalid PR URL {pr_url}"
                    )))
                })?;
            }
        }

        Ok(commit_info)
    }
    /// Pre-upload LFS files to S3 in preparation for a future commit.
    pub async fn preupload_lfs_files(
        &self,
        mut additions: Vec<CommitOperationAdd>,
        create_pr: Option<bool>,
        num_threads: Option<usize>,
        gitignore_content: Option<String>,
    ) -> Result<Vec<CommitOperationAdd>, ApiError> {
        // Set default values
        let create_pr = create_pr.unwrap_or(false);
        let num_threads = num_threads.unwrap_or(5);

        // Check for gitignore content in additions if not provided
        let mut gitignore_content = gitignore_content;
        if gitignore_content.is_none() {
            for addition in &additions {
                if addition.path_in_repo == ".gitignore" {
                    gitignore_content = Some(read_to_string(addition.path_in_repo.clone()).await?);
                    break;
                }
            }
        }

        // Fetch upload modes for new files
        self.fetch_and_apply_upload_modes(&mut additions, create_pr, gitignore_content)
            .await?;

        // Filter LFS files
        let (lfs_files, small_files): (Vec<_>, Vec<_>) = additions
            .into_iter()
            .partition(|addition| addition.upload_mode == UploadMode::Lfs);

        // Filter out ignored files
        let mut new_lfs_additions_to_upload = Vec::new();
        let mut ignored_count = 0;
        for addition in lfs_files {
            if addition.should_ignore {
                ignored_count += 1;
                log::debug!(
                    "Skipping upload for LFS file '{}' (ignored by gitignore file).",
                    addition.path_in_repo
                );
            } else {
                new_lfs_additions_to_upload.push(addition);
            }
        }

        if ignored_count > 0 {
            log::info!(
                "Skipped upload for {} LFS file(s) (ignored by gitignore file).",
                ignored_count
            );
        }

        // Upload LFS files
        let uploaded_lfs_files = self
            .upload_lfs_files(new_lfs_additions_to_upload, num_threads)
            .await?;
        Ok(small_files.into_iter().chain(uploaded_lfs_files).collect())
    }

    /// Requests the Hub "preupload" endpoint to determine whether each input file should be uploaded as a regular git blob
    /// or as git LFS blob. Input `additions` are mutated in-place with the upload mode.
    pub async fn fetch_and_apply_upload_modes(
        &self,
        additions: &mut Vec<CommitOperationAdd>,
        create_pr: bool,
        gitignore_content: Option<String>,
    ) -> Result<(), ApiError> {
        // Process in chunks of 256
        for chunk in additions.chunks_mut(256) {
            let files: Vec<PreuploadFile> = chunk
                .iter()
                .map(|op| PreuploadFile {
                    path: op.path_in_repo.clone(),
                    sample: BASE64.encode(&op.upload_info.sample),
                    size: op.upload_info.size,
                })
                .collect();

            let payload = PreuploadRequest {
                files,
                git_ignore: gitignore_content.clone(),
            };

            let mut url = self.preupload_url();

            if create_pr {
                url.push_str("?create_pr=1");
            }

            let preupload_info: PreuploadResponse = self
                .api
                .client
                .post(&url)
                .json(&payload)
                .send()
                .await?
                .maybe_hf_err()
                .await?
                .json()
                .await?;

            // Update the operations with the response information
            for file_info in preupload_info.files {
                if let Some(op) = chunk
                    .iter_mut()
                    .find(|op| op.path_in_repo == file_info.path)
                {
                    op.upload_mode = match file_info.upload_mode.as_str() {
                        "lfs" => UploadMode::Lfs,
                        "regular" => UploadMode::Regular,
                        m => {
                            return Err(ApiError::InvalidResponse(format!(
                                "Bad upload mode {m} returned from preupload info."
                            )))
                        }
                    };
                    op.should_ignore = file_info.should_ignore;
                    op.remote_oid = file_info.oid;
                }
            }
        }

        // Handle empty files
        for addition in additions.iter_mut() {
            if addition.upload_info.size == 0 {
                addition.upload_mode = UploadMode::Regular;
            }
        }

        Ok(())
    }

    /// Uploads the content of `additions` to the Hub using the large file storage protocol.
    /// Relevant external documentation:
    ///     - LFS Batch API: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md

    async fn upload_lfs_files(
        &self,
        additions: Vec<CommitOperationAdd>,
        num_threads: usize,
    ) -> Result<Vec<CommitOperationAdd>, ApiError> {
        // Step 1: Retrieve upload instructions from LFS batch endpoint
        let mut batch_objects = Vec::new();

        // Process in chunks of 256 files
        for chunk in additions.chunks(256) {
            let mut batch_info = self.post_lfs_batch_info(chunk).await?;
            let errs: Vec<_> = batch_info
                .iter_mut()
                .flat_map(|b| b.error.take().map(|e| (b.oid.clone(), e)))
                .collect();
            if !errs.is_empty() {
                return Err(ApiError::InvalidResponse(
                    errs.into_iter()
                        .map(|e| {
                            format!(
                                "Encountered error for file with OID {}: `{}`)",
                                e.0, e.1.message
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                ));
            }
            batch_objects.extend(batch_info);
        }

        // Create mapping of OID to addition operation
        let mut oid2addop: HashMap<String, CommitOperationAdd> = additions
            .into_iter()
            .map(|op| {
                let oid = hex::encode(&op.upload_info.sha256);
                (oid, op)
            })
            .collect();

        // Step 2: Filter out already uploaded files
        let filtered_actions: Vec<_> = batch_objects
            .into_iter()
            .filter(|action| {
                if action.actions.is_none() {
                    if let Some(op) = oid2addop.get(&action.oid) {
                        log::debug!(
                            "Content of file {} is already present upstream - skipping upload.",
                            op.path_in_repo
                        );
                    }
                    false
                } else {
                    true
                }
            })
            .collect();

        if filtered_actions.is_empty() {
            log::debug!("No LFS files to upload.");
            return Ok(oid2addop.into_values().collect());
        }

        let s3_client = reqwest::Client::new();

        // Step 3: Upload files concurrently
        let endpoint = self.api.endpoint.clone();
        let upload_futures: Vec<_> = filtered_actions
            .into_iter()
            .map(|batch_action| {
                let operation = oid2addop.remove(&batch_action.oid).unwrap();
                lfs_upload(
                    self.api.client.clone(),
                    s3_client.clone(),
                    operation,
                    batch_action,
                    endpoint.clone(),
                )
            })
            .collect();

        log::debug!(
            "Uploading {} LFS files to the Hub using up to {} threads concurrently",
            upload_futures.len(),
            num_threads
        );

        // Use tokio::spawn to handle concurrent uploads
        let handles: Vec<_> = upload_futures
            .into_iter()
            .map(|future| tokio::spawn(future))
            .collect();

        let mut operations: Vec<_> = oid2addop.drain().map(|(_k, v)| v).collect();
        for handle in handles {
            log::trace!("joining handle..");
            operations.push(handle.await.map_err(|e| {
                ApiError::IoError(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    format!("failed to join lfs upload thread {e}"),
                ))
            })??);
        }

        log::debug!("Uploaded {} LFS files to the Hub.", operations.len(),);

        Ok(operations)
    }
}

/// Computes the git-sha1 hash of the given bytes, using the same algorithm as git.
///
/// This is equivalent to running `git hash-object`. See https://git-scm.com/docs/git-hash-object
/// for more details.
///
/// Note: this method is valid for regular files. For LFS files, the proper git hash is supposed to be computed on the
/// pointer file content, not the actual file content. However, for simplicity, we directly compare the sha256 of
/// the LFS file content when we want to compare LFS files.
///
/// # Returns
///
/// The git-hash of `data` as a hexadecimal string.
///
/// # Example
///
/// ```
/// let hash = git_hash(b"Hello, World!");
/// assert_eq!(hash, "b45ef6fec89518d314f546fd6c3025367b721684");
/// ```
pub fn git_hash(data: &[u8]) -> String {
    let mut hasher = Sha1::new();

    // Add header
    hasher.update(b"blob ");
    hasher.update(data.len().to_string().as_bytes());
    hasher.update(b"\0");

    // Add data
    hasher.update(data);

    // Convert to hex string
    format!("{:x}", hasher.finalize())
}

/// Warn user when a list of operations is expected to overwrite itself in a single
/// commit.
///
/// Rules:
/// - If a filepath is updated by multiple `CommitOperationAdd` operations, a warning
///   message is triggered.
/// - If a filepath is updated at least once by a `CommitOperationAdd` and then deleted
///   by a `CommitOperationDelete`, a warning is triggered.
/// - If a `CommitOperationDelete` deletes a filepath that is then updated by a
///   `CommitOperationAdd`, no warning is triggered. This is usually useless (no need to
///   delete before upload) but can happen if a user deletes an entire folder and then
///   add new files to it.
fn warn_on_overwriting_operations(operations: &[CommitOperation]) {
    let mut nb_additions_per_path: HashMap<String, u32> = HashMap::new();
    for operation in operations {
        // i know it's irrefutable, but later we're gonna add more operations, so we'll if-let or match then.
        let CommitOperation::Add(CommitOperationAdd { path_in_repo, .. }) = operation;
        {
            if nb_additions_per_path.contains_key(path_in_repo) {
                warn!(
                    "About to update multiple times the same file in the same commit: '{path_in_repo}'. This can cause undesired inconsistencies in your repo."
                );
            }

            *nb_additions_per_path
                .entry(path_in_repo.clone())
                .or_insert(0) += 1
        }
    }
}

#[derive(Serialize, Debug)]
struct HeaderValue {
    summary: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_commit: Option<String>,
}

#[derive(Serialize, Debug)]
struct FileValue {
    content: String,
    path: String,
    encoding: String,
}

#[derive(Serialize, Debug)]
struct LfsFileValue {
    path: String,
    algo: String,
    oid: String,
    size: u64,
}

// todo add
// #[derive(Serialize)]
// struct DeletedValue {
//     path: String,
// }

#[derive(Serialize, Debug)]
#[serde(tag = "key", content = "value")]
#[serde(rename_all = "camelCase")]
enum CommitPayloadItem {
    Header(HeaderValue),
    File(FileValue),
    LfsFile(LfsFileValue),
    // todo add
    // DeletedFile(DeletedValue),
    // DeletedFolder(DeletedValue),
}

fn prepare_commit_payload(
    operations: &[CommitOperation],
    // TODO: add copy functionality
    // files_to_copy: &[],
    commit_message: &str,
    commit_description: &str,
    parent_commit: Option<&str>,
) -> Vec<CommitPayloadItem> {
    let mut payload = Vec::new();

    // 1. Send header item with commit metadata
    payload.push(CommitPayloadItem::Header(HeaderValue {
        summary: commit_message.to_string(),
        description: commit_description.to_string(),
        parent_commit: parent_commit.map(String::from),
    }));

    let mut nb_ignored_files = 0;

    // 2. Send operations, one per line
    for operation in operations {
        match operation {
            // 2.a and 2.b: Adding files (regular or LFS)
            CommitOperation::Add(add_op) => {
                if add_op.should_ignore {
                    log::debug!(
                        "Skipping file '{}' in commit (ignored by gitignore file).",
                        add_op.path_in_repo
                    );
                    nb_ignored_files += 1;
                    continue;
                }

                match &add_op.upload_mode {
                    UploadMode::Regular => {
                        let content = match &add_op.source {
                            UploadSource::Bytes(bytes) => BASE64.encode(bytes),
                            UploadSource::File(path) => {
                                BASE64.encode(std::fs::read(path).unwrap()) // TODO: proper error handling
                            }
                            UploadSource::Emptied => continue,
                        };

                        payload.push(CommitPayloadItem::File(FileValue {
                            content,
                            path: add_op.path_in_repo.clone(),
                            encoding: "base64".to_string(),
                        }));
                    }
                    UploadMode::Lfs => {
                        payload.push(CommitPayloadItem::LfsFile(LfsFileValue {
                            path: add_op.path_in_repo.clone(),
                            algo: "sha256".to_string(),
                            oid: hex::encode(&add_op.upload_info.sha256),
                            size: add_op.upload_info.size,
                        }));
                    }
                }
            } // TODO: Add other operations when implemented
        }
    }

    if nb_ignored_files > 0 {
        log::info!(
            "Skipped {} file(s) in commit (ignored by gitignore file).",
            nb_ignored_files
        );
    }

    payload
}

fn payload_as_ndjson(payload: Vec<CommitPayloadItem>) -> impl Iterator<Item = Vec<u8>> {
    payload.into_iter().flat_map(|item| {
        let mut json = serde_json::to_vec(&item).unwrap();
        json.push(b'\n');
        vec![json]
    })
}
