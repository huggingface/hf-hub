//! Bucket handles and file operations for Hugging Face Hub buckets.
//!
//! Buckets are flat file stores that live under a namespace such as
//! `"user/my-bucket"`. Start by creating an [`HFBucket`] handle with
//! [`HFClient::bucket`], then use the handle to inspect metadata, browse the
//! tree, upload or download files, or delete entries.
//!
//! This module exposes both high-level helpers and the lower-level batch API:
//!
//! - Use [`HFBucket::info`], [`HFBucket::list_tree`], and [`HFBucket::get_paths_info`] to inspect bucket contents.
//! - Use [`HFBucket::upload_files`] and [`HFBucket::download_files`] for common transfer workflows.
//! - Use [`HFBucket::batch`] when you already have xet hashes and want direct control over add, delete, or copy
//!   operations.
//! - Use [`sync`] for one-way directory mirroring between a local folder and a bucket prefix.

pub mod sync;

use std::fmt;
use std::path::PathBuf;

use futures::Stream;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult, NotFoundContext};
use crate::progress::{DownloadEvent, EmitEvent, Progress, UploadEvent};
use crate::retry;

const BUCKET_BATCH_CHUNK_SIZE: usize = 1000;
const BUCKET_PATHS_INFO_BATCH_SIZE: usize = 1000;

/// A handle for a single bucket on the Hugging Face Hub.
///
/// `HFBucket` is created via [`HFClient::bucket`] and binds together the client,
/// owner (namespace), and bucket name. All bucket-scoped API operations are methods
/// on this type.
///
/// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let bucket = client.bucket("my-org", "my-bucket");
/// assert_eq!(bucket.bucket_id(), "my-org/my-bucket");
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFBucket {
    pub(crate) hf_client: HFClient,
    owner: String,
    name: String,
}

impl fmt::Debug for HFBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFBucket")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .finish()
    }
}

impl HFBucket {
    /// Construct a new bucket handle. Prefer [`HFClient::bucket`] in most cases.
    pub fn new(client: HFClient, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The bucket owner (user or organization namespace).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The bucket name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` bucket identifier used in Hub API calls.
    pub fn bucket_id(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }

    /// Get metadata about this bucket.
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}`
    pub async fn info(&self) -> HFResult<BucketInfo> {
        let bucket_id = self.bucket_id();
        let url = self.hf_client.bucket_api_url(&bucket_id);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;

        let response = self
            .hf_client
            .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
            .await?;
        Ok(response.json().await?)
    }

    /// List files and directories in this bucket.
    ///
    /// This is the main API for browsing bucket contents. Use `prefix` to scope
    /// the listing to a subdirectory-like path, and `recursive` to choose
    /// between a shallow listing and a full traversal.
    ///
    /// For targeted lookups of a known set of paths, prefer
    /// [`HFBucket::get_paths_info`].
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}/tree[/{prefix}]` (paginated)
    pub fn list_tree(
        &self,
        params: ListBucketTreeParams,
    ) -> HFResult<impl Stream<Item = HFResult<BucketTreeEntry>> + '_> {
        let bucket_id = self.bucket_id();
        let mut url_str = format!("{}/api/buckets/{}/tree", self.hf_client.endpoint(), bucket_id);
        if let Some(ref prefix) = params.prefix {
            url_str = format!("{}/{}", url_str, prefix);
        }

        let url = Url::parse(&url_str)?;

        let mut query = vec![];
        if params.recursive == Some(true) {
            query.push(("recursive".to_string(), "true".to_string()));
        }

        Ok(self.hf_client.paginate(url, query, None))
    }

    /// Get info about specific paths in this bucket.
    ///
    /// This is useful when you already know which paths you care about and do
    /// not want to stream the full tree. Requests are automatically chunked in
    /// batches of 1000 paths.
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/paths-info`
    pub async fn get_paths_info(&self, paths: &[String]) -> HFResult<Vec<BucketTreeEntry>> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/api/buckets/{}/paths-info", self.hf_client.endpoint(), bucket_id);

        let headers = self.hf_client.auth_headers();
        let mut all_entries = Vec::new();
        for chunk in paths.chunks(BUCKET_PATHS_INFO_BATCH_SIZE) {
            let body = serde_json::json!({ "paths": chunk });

            let response = retry::retry(self.hf_client.retry_config(), || {
                self.hf_client
                    .http_client()
                    .post(&url)
                    .headers(headers.clone())
                    .json(&body)
                    .send()
            })
            .await?;

            let response = self
                .hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
            let entries: Vec<BucketTreeEntry> = response.json().await?;
            all_entries.extend(entries);
        }

        Ok(all_entries)
    }

    /// Get metadata for a single file in this bucket via a HEAD request.
    ///
    /// Endpoint: `HEAD /buckets/{bucket_id}/resolve/{path}`
    pub async fn get_file_metadata(&self, remote_path: &str) -> HFResult<BucketFileMetadata> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/buckets/{}/resolve/{}", self.hf_client.endpoint(), bucket_id, remote_path);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.no_redirect_client().head(&url).headers(headers.clone()).send()
        })
        .await?;

        let response = self
            .hf_client
            .check_response(
                response,
                Some(&bucket_id),
                NotFoundContext::Entry {
                    path: remote_path.to_string(),
                },
            )
            .await?;

        let size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let xet_hash = response
            .headers()
            .get("x-xet-hash")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        Ok(BucketFileMetadata { size, xet_hash })
    }

    /// Execute batch file operations (add, delete, copy) on this bucket.
    ///
    /// This is the low-level file mutation API. `add` operations only register
    /// metadata on the bucket side; the file contents must already have been
    /// uploaded to xet so each entry has a valid [`BucketAddFile::xet_hash`].
    ///
    /// For simpler upload and download flows, prefer
    /// [`HFBucket::upload_files`] and [`HFBucket::download_files`].
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/batch` (NDJSON)
    pub async fn batch(&self, params: BatchBucketFilesParams) -> HFResult<()> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/api/buckets/{}/batch", self.hf_client.endpoint(), bucket_id);

        let mut lines: Vec<String> = Vec::new();

        for add in &params.add {
            let mut obj = serde_json::json!({
                "type": "addFile",
                "path": add.path,
                "xetHash": add.xet_hash,
                "size": add.size,
            });
            if let Some(mtime) = add.mtime {
                obj["mtime"] = serde_json::Value::Number(mtime.into());
            }
            if let Some(ref ct) = add.content_type {
                obj["contentType"] = serde_json::Value::String(ct.clone());
            }
            lines.push(serde_json::to_string(&obj)?);
        }

        for path in &params.delete {
            let obj = serde_json::json!({
                "type": "deleteFile",
                "path": path,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        for copy in &params.copy {
            let obj = serde_json::json!({
                "type": "copyFile",
                "path": copy.path,
                "xetHash": copy.xet_hash,
                "sourceRepoType": copy.source_repo_type,
                "sourceRepoId": copy.source_repo_id,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        let headers = self.hf_client.auth_headers();
        for chunk in lines.chunks(BUCKET_BATCH_CHUNK_SIZE) {
            let body = chunk.join("\n") + "\n";

            let response = retry::retry(self.hf_client.retry_config(), || {
                self.hf_client
                    .http_client()
                    .post(&url)
                    .headers(headers.clone())
                    .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                    .body(body.clone())
                    .send()
            })
            .await?;

            self.hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
        }

        Ok(())
    }

    /// Delete files from this bucket by path.
    ///
    /// This is a convenience wrapper around [`HFBucket::batch`] that sends only
    /// `deleteFile` operations.
    pub async fn delete_files(&self, paths: &[String]) -> HFResult<()> {
        let params = BatchBucketFilesParams {
            add: vec![],
            delete: paths.to_vec(),
            copy: vec![],
        };
        self.batch(params).await
    }

    /// Upload local files to the bucket.
    ///
    /// Each tuple maps `(local_path, remote_path)`. File contents are uploaded
    /// to xet first, then registered in the bucket via the batch endpoint.
    ///
    /// Pass a [`Progress`] handler to receive aggregate upload progress events.
    pub async fn upload_files(
        &self,
        files: &[(std::path::PathBuf, String)],
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let total_bytes: u64 = files
            .iter()
            .filter_map(|(p, _)| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        progress.emit(UploadEvent::Start {
            total_files: files.len(),
            total_bytes,
        });

        let xet_files: Vec<(String, crate::repository::AddSource)> = files
            .iter()
            .map(|(local_path, remote_path)| {
                (remote_path.clone(), crate::repository::AddSource::File(local_path.clone()))
            })
            .collect();

        let xet_infos = self.xet_upload(&xet_files, progress).await?;

        let add_files: Vec<BucketAddFile> = files
            .iter()
            .zip(xet_infos.iter())
            .map(|((local_path, remote_path), xet_info)| {
                let metadata = std::fs::metadata(local_path).ok();
                let size = metadata.as_ref().map(|m| m.len()).or(xet_info.file_size).unwrap_or(0);
                let mtime = metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());

                BucketAddFile {
                    path: remote_path.clone(),
                    xet_hash: xet_info.hash.clone(),
                    size,
                    mtime,
                    content_type: None,
                }
            })
            .collect();

        let batch_params = BatchBucketFilesParams {
            add: add_files,
            ..Default::default()
        };
        self.batch(batch_params).await?;

        progress.emit(UploadEvent::Complete);
        Ok(())
    }

    /// Download files from the bucket to local paths.
    ///
    /// Each tuple maps `(remote_path, local_path)`. The method first resolves
    /// xet metadata with [`HFBucket::get_paths_info`], then downloads the file
    /// contents through xet. Directory entries are rejected.
    ///
    /// Pass a [`Progress`] handler to receive aggregate download progress
    /// events.
    pub async fn download_files(&self, params: BucketDownloadFilesParams, progress: &Option<Progress>) -> HFResult<()> {
        if params.files.is_empty() {
            return Ok(());
        }

        let remote_paths: Vec<String> = params.files.iter().map(|(r, _)| r.clone()).collect();
        let entries = self.get_paths_info(&remote_paths).await?;

        let entry_map: std::collections::HashMap<String, BucketTreeEntry> = entries
            .into_iter()
            .map(|e| {
                let path = match &e {
                    BucketTreeEntry::File { path, .. } => path.clone(),
                    BucketTreeEntry::Directory { path, .. } => path.clone(),
                };
                (path, e)
            })
            .collect();

        let mut xet_batch_files = Vec::new();
        let mut total_bytes: u64 = 0;

        for (remote_path, local_path) in &params.files {
            match entry_map.get(remote_path) {
                Some(BucketTreeEntry::File { xet_hash, size, .. }) => {
                    total_bytes += size;
                    xet_batch_files.push(crate::xet::XetBatchFile {
                        hash: xet_hash.clone(),
                        file_size: *size,
                        path: local_path.clone(),
                        filename: remote_path.clone(),
                    });
                },
                Some(BucketTreeEntry::Directory { path, .. }) => {
                    return Err(HFError::InvalidParameter(format!("Cannot download directory entry: {path}")));
                },
                None => {
                    return Err(HFError::EntryNotFound {
                        path: remote_path.clone(),
                        repo_id: self.bucket_id(),
                        context: None,
                    });
                },
            }
        }

        progress.emit(DownloadEvent::Start {
            total_files: xet_batch_files.len(),
            total_bytes,
        });

        self.xet_download_batch(&xet_batch_files, progress).await?;

        progress.emit(DownloadEvent::Complete);
        Ok(())
    }
}

/// Parameters for creating a new bucket on the Hub.
///
/// Used with [`HFClient::create_bucket`].
#[derive(Debug, Clone, TypedBuilder)]
pub struct CreateBucketParams {
    /// Namespace (user or organization) that owns the bucket.
    #[builder(setter(into))]
    pub namespace: String,
    /// Bucket name within the namespace.
    #[builder(setter(into))]
    pub name: String,
    /// Whether the bucket should be private. Defaults to `false`.
    #[builder(default = false)]
    pub private: bool,
    /// Enterprise resource group ID (optional).
    #[builder(default, setter(into, strip_option))]
    pub resource_group_id: Option<String>,
    /// If `true`, do not error when the bucket already exists. Defaults to `false`.
    #[builder(default = false)]
    pub exist_ok: bool,
}

/// Parameters for listing files in a bucket tree.
///
/// Used with [`HFBucket::list_tree`].
#[derive(Debug, Clone, Default, TypedBuilder)]
pub struct ListBucketTreeParams {
    /// Filter results to entries under this prefix.
    #[builder(default, setter(into, strip_option))]
    pub prefix: Option<String>,
    /// If `true`, list entries recursively under the prefix.
    #[builder(default, setter(strip_option))]
    pub recursive: Option<bool>,
}

/// Parameters for batch operations on bucket files.
///
/// Used with [`HFBucket::batch`].
/// Operations are chunked at 1000 entries per request.
#[derive(Debug, Clone, Default, TypedBuilder)]
pub struct BatchBucketFilesParams {
    /// Files to add (register) in the bucket.
    #[builder(default)]
    pub add: Vec<BucketAddFile>,
    /// Paths of files to delete from the bucket.
    #[builder(default)]
    pub delete: Vec<String>,
    /// Files to copy (server-side) into the bucket.
    #[builder(default)]
    pub copy: Vec<BucketCopyFile>,
}

/// A file to register in a bucket via the batch endpoint.
///
/// Represents an `addFile` entry in the NDJSON batch payload.
/// The file content must have already been uploaded to xet to obtain the `xet_hash`.
#[derive(Debug, Clone)]
pub struct BucketAddFile {
    /// Destination path in the bucket.
    pub path: String,
    /// Xet content hash from a prior upload.
    pub xet_hash: String,
    /// File size in bytes.
    pub size: u64,
    /// Last modification time as a Unix timestamp (seconds).
    pub mtime: Option<u64>,
    /// MIME content type (e.g. `"text/plain"`, `"application/octet-stream"`).
    pub content_type: Option<String>,
}

/// A server-side copy operation for the batch endpoint.
///
/// Represents a `copyFile` entry in the NDJSON batch payload.
/// Copies are performed by xet hash — no data transfer occurs.
#[derive(Debug, Clone)]
pub struct BucketCopyFile {
    /// Destination path in the bucket.
    pub path: String,
    /// Xet content hash to copy.
    pub xet_hash: String,
    /// Source repo type (e.g. `"bucket"`, `"model"`).
    pub source_repo_type: String,
    /// Source repo or bucket ID (e.g. `"user/my-bucket"`).
    pub source_repo_id: String,
}

/// Parameters for downloading files from a bucket.
///
/// Used with [`HFBucket::download_files`].
#[derive(Debug, Clone, TypedBuilder)]
pub struct BucketDownloadFilesParams {
    /// List of `(remote_path, local_path)` pairs to download.
    pub files: Vec<(String, PathBuf)>,
}

/// Metadata about a bucket on the Hugging Face Hub.
///
/// Returned by [`HFBucket::info`] and
/// [`HFClient::list_buckets`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketInfo {
    /// Full bucket identifier, e.g. `"namespace/bucket_name"`.
    pub id: String,
    /// Whether the bucket is private.
    pub private: bool,
    /// ISO 8601 creation timestamp.
    #[serde(rename = "createdAt")]
    pub created_at: String,
    /// Total size of all files in bytes.
    pub size: u64,
    /// Number of files in the bucket.
    #[serde(rename = "totalFiles")]
    pub total_files: u64,
}

/// URL returned after creating a bucket.
///
/// Returned by [`HFClient::create_bucket`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketUrl {
    /// Full URL to the bucket on the Hub.
    pub url: String,
}

/// A file or directory entry in a bucket tree listing.
///
/// Returned by [`HFBucket::list_tree`] and
/// [`HFBucket::get_paths_info`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BucketTreeEntry {
    /// A file entry with content hash and size.
    File {
        /// Path within the bucket.
        path: String,
        /// File size in bytes.
        size: u64,
        /// Xet content-addressable hash.
        #[serde(rename = "xetHash")]
        xet_hash: String,
        /// Last modification time (ISO 8601), if available.
        mtime: Option<String>,
        /// Upload timestamp (ISO 8601), if available.
        uploaded_at: Option<String>,
    },
    /// A directory entry.
    Directory {
        /// Directory path within the bucket.
        path: String,
        /// Upload timestamp (ISO 8601), if available.
        uploaded_at: Option<String>,
    },
}

/// Metadata for a single file in a bucket, retrieved via HEAD request.
///
/// Returned by [`HFBucket::get_file_metadata`].
#[derive(Debug, Clone)]
pub struct BucketFileMetadata {
    /// File size in bytes.
    pub size: u64,
    /// Xet content-addressable hash.
    pub xet_hash: String,
}

impl HFClient {
    /// Create an [`HFBucket`] handle for a bucket.
    ///
    /// The returned handle is cheap to clone and can be reused across multiple
    /// bucket-scoped operations.
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucket {
        HFBucket::new(self.clone(), owner, name)
    }

    /// Create a new bucket on the Hub.
    ///
    /// When [`CreateBucketParams::exist_ok`] is `true`, an existing bucket is
    /// treated as success and a [`BucketUrl`] is synthesized locally.
    ///
    /// Endpoint: `POST /api/buckets/{namespace}/{name}`
    pub async fn create_bucket(&self, params: CreateBucketParams) -> HFResult<BucketUrl> {
        let url = format!("{}/api/buckets/{}/{}", self.endpoint(), params.namespace, params.name);

        let mut body = serde_json::json!({});
        if params.private {
            body["private"] = serde_json::Value::Bool(true);
        }
        if let Some(ref rg) = params.resource_group_id {
            body["resourceGroupId"] = serde_json::Value::String(rg.clone());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        let bucket_id = format!("{}/{}", params.namespace, params.name);

        if response.status().as_u16() == 409 && params.exist_ok {
            return Ok(BucketUrl {
                url: format!("{}/buckets/{}", self.endpoint(), bucket_id),
            });
        }

        let response = self
            .check_response(response, Some(&bucket_id), NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a bucket from the Hub.
    ///
    /// Endpoint: `DELETE /api/buckets/{bucket_id}`
    pub async fn delete_bucket(&self, bucket_id: &str, missing_ok: bool) -> HFResult<()> {
        let url = self.bucket_api_url(bucket_id);

        let headers = self.auth_headers();
        let response =
            retry::retry(self.retry_config(), || self.http_client().delete(&url).headers(headers.clone()).send())
                .await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(bucket_id), NotFoundContext::Bucket).await?;
        Ok(())
    }

    /// List buckets in a namespace.
    ///
    /// Streams bucket metadata for all buckets owned by the given user or
    /// organization.
    ///
    /// Endpoint: `GET /api/buckets/{namespace}` (paginated)
    pub fn list_buckets(&self, namespace: &str) -> HFResult<impl Stream<Item = HFResult<BucketInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/buckets/{}", self.endpoint(), namespace))?;
        Ok(self.paginate(url, vec![], None))
    }

    /// Move (rename) a bucket.
    ///
    /// Both `from_id` and `to_id` use the `"owner/name"` bucket identifier
    /// format.
    ///
    /// Endpoint: `POST /api/repos/move` with `type: "bucket"`
    pub async fn move_bucket(&self, from_id: &str, to_id: &str) -> HFResult<()> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let body = serde_json::json!({
            "fromRepo": from_id,
            "toRepo": to_id,
            "type": "bucket",
        });

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        self.check_response(response, None, NotFoundContext::Generic).await?;
        Ok(())
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn create_bucket(&self, params: CreateBucketParams) -> HFResult<BucketUrl>;
        fn delete_bucket(&self, bucket_id: &str, missing_ok: bool) -> HFResult<()>;
        fn move_bucket(&self, from_id: &str, to_id: &str) -> HFResult<()>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_buckets(&self, namespace: &str) -> BucketInfo;
    }
}

sync_api! {
    impl HFBucket -> HFBucketSync {
        fn info(&self) -> HFResult<BucketInfo>;
        fn get_file_metadata(&self, remote_path: &str) -> HFResult<BucketFileMetadata>;
        fn get_paths_info(&self, paths: &[String]) -> HFResult<Vec<BucketTreeEntry>>;
        fn batch(&self, params: BatchBucketFilesParams) -> HFResult<()>;
        fn upload_files(&self, files: &[(std::path::PathBuf, String)], progress: &Option<Progress>) -> HFResult<()>;
        fn download_files(&self, params: BucketDownloadFilesParams, progress: &Option<Progress>) -> HFResult<()>;
        fn delete_files(&self, paths: &[String]) -> HFResult<()>;
    }
}

sync_api_stream! {
    impl HFBucket -> HFBucketSync {
        fn list_tree(&self, params: ListBucketTreeParams) -> BucketTreeEntry;
    }
}

#[cfg(test)]
mod tests {
    use super::HFBucket;

    #[test]
    fn test_bucket_accessors() {
        let client = crate::HFClient::builder().build().unwrap();
        let bucket = HFBucket::new(client, "my-org", "my-bucket");

        assert_eq!(bucket.owner(), "my-org");
        assert_eq!(bucket.name(), "my-bucket");
        assert_eq!(bucket.bucket_id(), "my-org/my-bucket");
    }
}
