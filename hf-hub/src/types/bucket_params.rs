use std::path::PathBuf;

use typed_builder::TypedBuilder;

use crate::types::progress::Progress;

/// Parameters for creating a new bucket on the Hub.
///
/// Used with [`HFClient::create_bucket`](crate::client::HFClient::create_bucket).
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
/// Used with [`HFBucket::list_tree`](crate::bucket::HFBucket::list_tree).
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
/// Used with [`HFBucket::batch`](crate::bucket::HFBucket::batch).
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
/// Used with [`HFBucket::download_files`](crate::bucket::HFBucket::download_files).
#[derive(Debug, Clone, TypedBuilder)]
pub struct BucketDownloadFilesParams {
    /// List of `(remote_path, local_path)` pairs to download.
    pub files: Vec<(String, PathBuf)>,
}

/// Direction for a bucket sync operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// Local directory -> bucket (upload).
    Upload,
    /// Bucket -> local directory (download).
    Download,
}

/// Parameters for syncing files between a local directory and a bucket.
///
/// Used with [`HFBucket::sync`](crate::bucket::HFBucket::sync).
#[derive(Clone, TypedBuilder)]
pub struct BucketSyncParams {
    /// Local directory path.
    pub local_path: PathBuf,
    /// Sync direction.
    pub direction: SyncDirection,
    /// Optional prefix within the bucket (subdirectory).
    #[builder(default, setter(into, strip_option))]
    pub prefix: Option<String>,
    /// Delete destination files not present in source.
    #[builder(default = false)]
    pub delete: bool,
    /// Only compare sizes, ignore modification times.
    #[builder(default = false)]
    pub ignore_times: bool,
    /// Only compare modification times, ignore sizes.
    #[builder(default = false)]
    pub ignore_sizes: bool,
    /// Only sync files that already exist at destination.
    #[builder(default = false)]
    pub existing: bool,
    /// Skip files that already exist at destination.
    #[builder(default = false)]
    pub ignore_existing: bool,
    /// Include patterns (fnmatch/glob-style).
    #[builder(default)]
    pub include: Vec<String>,
    /// Exclude patterns (fnmatch/glob-style).
    #[builder(default)]
    pub exclude: Vec<String>,
    /// Include skip operations in the returned plan.
    #[builder(default = false)]
    pub verbose: bool,
    /// Progress handler for upload/download tracking.
    #[builder(default)]
    pub progress: Progress,
}
