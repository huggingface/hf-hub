pub mod sync;
use serde::{Deserialize, Serialize};
pub use sync::*;

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
