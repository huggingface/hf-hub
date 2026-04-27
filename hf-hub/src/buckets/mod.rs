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

use bon::bon;
use futures::Stream;
#[cfg(feature = "blocking")]
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult, NotFoundContext};
use crate::progress::{DownloadEvent, EmitEvent, Progress, UploadEvent};
use crate::retry;

const BUCKET_BATCH_CHUNK_SIZE: usize = 1000;
const BUCKET_PATHS_INFO_BATCH_SIZE: usize = 1000;

pub(crate) mod _handle {
    use super::HFClient;

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
        pub(super) owner: String,
        pub(super) name: String,
    }
}

pub(crate) use _handle::HFBucket;

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
}

#[bon]
impl HFBucket {
    /// Get metadata about this bucket.
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}`.
    #[builder(finish_fn = send)]
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
    /// This is the main API for browsing bucket contents. For targeted lookups of a known set of
    /// paths, prefer [`HFBucket::get_paths_info`].
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}/tree[/{prefix}]` (paginated).
    ///
    /// # Parameters
    ///
    /// - `prefix`: filter results to entries under this prefix.
    /// - `recursive`: if `Some(true)`, list entries recursively under the prefix.
    #[builder(finish_fn = send)]
    pub fn list_tree(
        &self,
        #[builder(into)] prefix: Option<String>,
        recursive: Option<bool>,
    ) -> HFResult<impl Stream<Item = HFResult<BucketTreeEntry>> + '_> {
        let bucket_id = self.bucket_id();
        let mut url_str = format!("{}/api/buckets/{}/tree", self.hf_client.endpoint(), bucket_id);
        if let Some(ref prefix) = prefix {
            url_str = format!("{}/{}", url_str, prefix);
        }

        let url = Url::parse(&url_str)?;

        let mut query = vec![];
        if recursive == Some(true) {
            query.push(("recursive".to_string(), "true".to_string()));
        }

        Ok(self.hf_client.paginate(url, query, None))
    }

    /// Get info about specific paths in this bucket.
    ///
    /// This is useful when you already know which paths you care about and do not want to stream
    /// the full tree. Requests are automatically chunked in batches of 1000 paths.
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/paths-info`.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths to inspect.
    #[builder(finish_fn = send)]
    pub async fn get_paths_info(&self, paths: Vec<String>) -> HFResult<Vec<BucketTreeEntry>> {
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
    /// Endpoint: `HEAD /buckets/{bucket_id}/resolve/{path}`.
    ///
    /// # Parameters
    ///
    /// - `remote_path` (required): file path within the bucket.
    #[builder(finish_fn = send)]
    pub async fn get_file_metadata(&self, #[builder(into)] remote_path: String) -> HFResult<BucketFileMetadata> {
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
    /// This is the low-level file mutation API. `add` operations only register metadata on the
    /// bucket side; the file contents must already have been uploaded to xet so each entry has a
    /// valid [`BucketAddFile::xet_hash`].
    ///
    /// For simpler upload and download flows, prefer [`HFBucket::upload_files`] and
    /// [`HFBucket::download_files`].
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/batch` (NDJSON). Operations are chunked at 1000
    /// entries per request.
    ///
    /// # Parameters
    ///
    /// - `add`: files to add (register) in the bucket.
    /// - `delete`: paths of files to delete from the bucket.
    /// - `copy`: files to copy (server-side) into the bucket.
    #[builder(finish_fn = send)]
    #[allow(clippy::should_implement_trait)]
    pub async fn batch(
        &self,
        #[builder(default)] add: Vec<BucketAddFile>,
        #[builder(default)] delete: Vec<String>,
        #[builder(default)] copy: Vec<BucketCopyFile>,
    ) -> HFResult<()> {
        let bucket_id = self.bucket_id();
        let url = format!("{}/api/buckets/{}/batch", self.hf_client.endpoint(), bucket_id);

        let mut lines: Vec<String> = Vec::new();

        for add in &add {
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

        for path in &delete {
            let obj = serde_json::json!({
                "type": "deleteFile",
                "path": path,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        for copy in &copy {
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
    /// Convenience wrapper around [`HFBucket::batch`] that sends only `deleteFile` operations.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths to delete from the bucket.
    #[builder(finish_fn = send)]
    pub async fn delete_files(&self, paths: Vec<String>) -> HFResult<()> {
        self.batch().delete(paths).send().await
    }

    /// Upload local files to the bucket.
    ///
    /// Each tuple maps `(local_path, remote_path)`. File contents are uploaded to xet first, then
    /// registered in the bucket via the batch endpoint.
    ///
    /// # Parameters
    ///
    /// - `files` (required): list of `(local_path, remote_path)` pairs.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send)]
    pub async fn upload_files(&self, files: Vec<(PathBuf, String)>, progress: Option<Progress>) -> HFResult<()> {
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

        let xet_infos = self.xet_upload(&xet_files, &progress).await?;

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

        self.batch().add(add_files).send().await?;

        progress.emit(UploadEvent::Complete);
        Ok(())
    }

    /// Download files from the bucket to local paths.
    ///
    /// Each tuple maps `(remote_path, local_path)`. The method first resolves xet metadata, then
    /// downloads the file contents through xet. Directory entries are rejected.
    ///
    /// # Parameters
    ///
    /// - `files` (required): list of `(remote_path, local_path)` pairs.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send)]
    pub async fn download_files(&self, files: Vec<(String, PathBuf)>, progress: Option<Progress>) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let remote_paths: Vec<String> = files.iter().map(|(r, _)| r.clone()).collect();
        let entries = self.get_paths_info().paths(remote_paths).send().await?;

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

        for (remote_path, local_path) in &files {
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

        self.xet_download_batch(&xet_batch_files, &progress).await?;

        progress.emit(DownloadEvent::Complete);
        Ok(())
    }
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
}

#[bon]
impl HFClient {
    /// Create a new bucket on the Hub. Endpoint: `POST /api/buckets/{namespace}/{name}`.
    ///
    /// When `exist_ok` is `true`, an existing bucket is treated as success and a [`BucketUrl`] is
    /// synthesized locally.
    ///
    /// # Parameters
    ///
    /// - `namespace` (required): namespace (user or organization) that owns the bucket.
    /// - `name` (required): bucket name within the namespace.
    /// - `private` (default `false`): whether the bucket should be private.
    /// - `resource_group_id`: enterprise resource group ID.
    /// - `exist_ok` (default `false`): if `true`, do not error when the bucket already exists.
    #[builder(finish_fn = send)]
    pub async fn create_bucket(
        &self,
        #[builder(into)] namespace: String,
        #[builder(into)] name: String,
        #[builder(default)] private: bool,
        #[builder(into)] resource_group_id: Option<String>,
        #[builder(default)] exist_ok: bool,
    ) -> HFResult<BucketUrl> {
        let url = format!("{}/api/buckets/{}/{}", self.endpoint(), namespace, name);

        let mut body = serde_json::json!({});
        if private {
            body["private"] = serde_json::Value::Bool(true);
        }
        if let Some(ref rg) = resource_group_id {
            body["resourceGroupId"] = serde_json::Value::String(rg.clone());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        let bucket_id = format!("{}/{}", namespace, name);

        if response.status().as_u16() == 409 && exist_ok {
            return Ok(BucketUrl {
                url: format!("{}/buckets/{}", self.endpoint(), bucket_id),
            });
        }

        let response = self
            .check_response(response, Some(&bucket_id), NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a bucket from the Hub. Endpoint: `DELETE /api/buckets/{bucket_id}`.
    ///
    /// # Parameters
    ///
    /// - `bucket_id` (required): bucket ID in `"owner/name"` format.
    /// - `missing_ok` (default `false`): if `true`, do not error when the bucket does not exist.
    #[builder(finish_fn = send)]
    pub async fn delete_bucket(
        &self,
        #[builder(into)] bucket_id: String,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        let url = self.bucket_api_url(&bucket_id);

        let headers = self.auth_headers();
        let response =
            retry::retry(self.retry_config(), || self.http_client().delete(&url).headers(headers.clone()).send())
                .await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(&bucket_id), NotFoundContext::Bucket).await?;
        Ok(())
    }

    /// List buckets in a namespace.
    ///
    /// Streams bucket metadata for all buckets owned by the given user or organization.
    ///
    /// Endpoint: `GET /api/buckets/{namespace}` (paginated).
    ///
    /// # Parameters
    ///
    /// - `namespace` (required): user or organization namespace to list buckets for.
    #[builder(finish_fn = send)]
    pub fn list_buckets(
        &self,
        #[builder(into)] namespace: String,
    ) -> HFResult<impl Stream<Item = HFResult<BucketInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/buckets/{}", self.endpoint(), namespace))?;
        Ok(self.paginate(url, vec![], None))
    }

    /// Move (rename) a bucket.
    ///
    /// Endpoint: `POST /api/repos/move` with `type: "bucket"`.
    ///
    /// # Parameters
    ///
    /// - `from_id` (required): current bucket ID in `"owner/name"` format.
    /// - `to_id` (required): new bucket ID in `"owner/name"` format.
    #[builder(finish_fn = send)]
    pub async fn move_bucket(&self, #[builder(into)] from_id: String, #[builder(into)] to_id: String) -> HFResult<()> {
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

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::create_bucket`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn create_bucket(
        &self,
        #[builder(into)] namespace: String,
        #[builder(into)] name: String,
        #[builder(default)] private: bool,
        #[builder(into)] resource_group_id: Option<String>,
        #[builder(default)] exist_ok: bool,
    ) -> HFResult<BucketUrl> {
        self.runtime.block_on(
            self.inner
                .create_bucket()
                .namespace(namespace)
                .name(name)
                .private(private)
                .maybe_resource_group_id(resource_group_id)
                .exist_ok(exist_ok)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::delete_bucket`].
    #[builder(finish_fn = send)]
    pub fn delete_bucket(
        &self,
        #[builder(into)] bucket_id: String,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_bucket().bucket_id(bucket_id).missing_ok(missing_ok).send())
    }

    /// Blocking counterpart of [`HFClient::list_buckets`]. Collects the stream into a
    /// `Vec<BucketInfo>`.
    #[builder(finish_fn = send)]
    pub fn list_buckets(&self, #[builder(into)] namespace: String) -> HFResult<Vec<BucketInfo>> {
        self.runtime.block_on(async move {
            let stream = self.inner.list_buckets().namespace(namespace).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::move_bucket`].
    #[builder(finish_fn = send)]
    pub fn move_bucket(&self, #[builder(into)] from_id: String, #[builder(into)] to_id: String) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.move_bucket().from_id(from_id).to_id(to_id).send())
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFBucketSync {
    /// Blocking counterpart of [`HFBucket::info`].
    #[builder(finish_fn = send)]
    pub fn info(&self) -> HFResult<BucketInfo> {
        self.runtime.block_on(self.inner.info().send())
    }

    /// Blocking counterpart of [`HFBucket::list_tree`]. Collects the stream into a
    /// `Vec<BucketTreeEntry>`.
    #[builder(finish_fn = send)]
    pub fn list_tree(
        &self,
        #[builder(into)] prefix: Option<String>,
        recursive: Option<bool>,
    ) -> HFResult<Vec<BucketTreeEntry>> {
        self.runtime.block_on(async move {
            let stream = self.inner.list_tree().maybe_prefix(prefix).maybe_recursive(recursive).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFBucket::get_paths_info`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn get_paths_info(&self, paths: Vec<String>) -> HFResult<Vec<BucketTreeEntry>> {
        self.runtime.block_on(self.inner.get_paths_info().paths(paths).send())
    }

    /// Blocking counterpart of [`HFBucket::get_file_metadata`].
    #[builder(finish_fn = send)]
    pub fn get_file_metadata(&self, #[builder(into)] remote_path: String) -> HFResult<BucketFileMetadata> {
        self.runtime
            .block_on(self.inner.get_file_metadata().remote_path(remote_path).send())
    }

    /// Blocking counterpart of [`HFBucket::batch`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    #[allow(clippy::should_implement_trait)]
    pub fn batch(
        &self,
        #[builder(default)] add: Vec<BucketAddFile>,
        #[builder(default)] delete: Vec<String>,
        #[builder(default)] copy: Vec<BucketCopyFile>,
    ) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.batch().add(add).delete(delete).copy(copy).send())
    }

    /// Blocking counterpart of [`HFBucket::delete_files`].
    #[builder(finish_fn = send)]
    pub fn delete_files(&self, paths: Vec<String>) -> HFResult<()> {
        self.runtime.block_on(self.inner.delete_files().paths(paths).send())
    }

    /// Blocking counterpart of [`HFBucket::upload_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn upload_files(&self, files: Vec<(PathBuf, String)>, progress: Option<Progress>) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.upload_files().files(files).maybe_progress(progress).send())
    }

    /// Blocking counterpart of [`HFBucket::download_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn download_files(&self, files: Vec<(String, PathBuf)>, progress: Option<Progress>) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.download_files().files(files).maybe_progress(progress).send())
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
