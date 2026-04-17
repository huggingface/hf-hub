use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
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
pub enum BucketSyncDirection {
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
    pub direction: BucketSyncDirection,
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

/// Metadata about a bucket on the Hugging Face Hub.
///
/// Returned by [`HFBucket::info`](crate::bucket::HFBucket::info) and
/// [`HFClient::list_buckets`](crate::client::HFClient::list_buckets).
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketUrl {
    /// Full URL to the bucket on the Hub.
    pub url: String,
}

/// A file or directory entry in a bucket tree listing.
///
/// Returned by [`HFBucket::list_tree`](crate::bucket::HFBucket::list_tree) and
/// [`HFBucket::get_paths_info`](crate::bucket::HFBucket::get_paths_info).
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
/// Returned by [`HFBucket::get_file_metadata`](crate::bucket::HFBucket::get_file_metadata).
#[derive(Debug, Clone)]
pub struct BucketFileMetadata {
    /// File size in bytes.
    pub size: u64,
    /// Xet content-addressable hash.
    pub xet_hash: String,
}

/// Action to perform for a single file during sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketSyncAction {
    Upload,
    Download,
    Delete,
    Skip,
}

/// A single operation within a sync plan.
#[derive(Debug, Clone)]
pub struct BucketSyncOperation {
    /// What action to take.
    pub action: BucketSyncAction,
    /// Relative file path (forward-slash separated).
    pub path: String,
    /// File size in bytes, if known.
    pub size: Option<u64>,
    /// Human-readable reason for this action (e.g. "new file", "size differs", "identical").
    pub reason: String,
}

/// The computed sync plan — describes what will happen (or has happened) during a sync.
///
/// Returned by [`HFBucket::sync`](crate::bucket::HFBucket::sync).
#[derive(Debug, Clone)]
pub struct BucketSyncPlan {
    /// Sync direction that produced this plan.
    pub direction: BucketSyncDirection,
    /// All operations in the plan.
    pub operations: Vec<BucketSyncOperation>,
    /// Bucket tree entries for download operations, keyed by relative path.
    /// Used internally during execution to avoid re-fetching metadata.
    pub(crate) download_entries: HashMap<String, BucketTreeEntry>,
}

impl BucketSyncPlan {
    pub fn uploads(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Upload)
            .count()
    }

    pub fn downloads(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Download)
            .count()
    }

    pub fn deletes(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Delete)
            .count()
    }

    pub fn skips(&self) -> usize {
        self.operations.iter().filter(|op| op.action == BucketSyncAction::Skip).count()
    }

    /// Total bytes to transfer (upload + download operations).
    pub fn transfer_bytes(&self) -> u64 {
        self.operations
            .iter()
            .filter(|op| op.action == BucketSyncAction::Upload || op.action == BucketSyncAction::Download)
            .filter_map(|op| op.size)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plan(ops: Vec<(BucketSyncAction, Option<u64>)>) -> BucketSyncPlan {
        BucketSyncPlan {
            direction: BucketSyncDirection::Upload,
            operations: ops
                .into_iter()
                .enumerate()
                .map(|(i, (action, size))| BucketSyncOperation {
                    action,
                    path: format!("file_{i}.txt"),
                    size,
                    reason: "test".to_string(),
                })
                .collect(),
            download_entries: HashMap::new(),
        }
    }

    #[test]
    fn test_plan_counts() {
        let plan = make_plan(vec![
            (BucketSyncAction::Upload, Some(100)),
            (BucketSyncAction::Upload, Some(200)),
            (BucketSyncAction::Download, Some(300)),
            (BucketSyncAction::Delete, Some(50)),
            (BucketSyncAction::Skip, None),
        ]);
        assert_eq!(plan.uploads(), 2);
        assert_eq!(plan.downloads(), 1);
        assert_eq!(plan.deletes(), 1);
        assert_eq!(plan.skips(), 1);
    }

    #[test]
    fn test_transfer_bytes() {
        let plan = make_plan(vec![
            (BucketSyncAction::Upload, Some(100)),
            (BucketSyncAction::Download, Some(300)),
            (BucketSyncAction::Delete, Some(50)),
            (BucketSyncAction::Skip, None),
        ]);
        assert_eq!(plan.transfer_bytes(), 400);
    }

    #[test]
    fn test_empty_plan() {
        let plan = make_plan(vec![]);
        assert_eq!(plan.uploads(), 0);
        assert_eq!(plan.downloads(), 0);
        assert_eq!(plan.deletes(), 0);
        assert_eq!(plan.skips(), 0);
        assert_eq!(plan.transfer_bytes(), 0);
    }
}
