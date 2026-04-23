use std::path::PathBuf;
use std::time::SystemTime;

use super::repo::RepoType;

/// Information about a single cached file on disk, including its blob and access times.
pub struct CachedFileInfo {
    pub file_name: String,
    pub file_path: PathBuf,
    pub blob_path: PathBuf,
    pub size_on_disk: u64,
    pub blob_last_accessed: SystemTime,
    pub blob_last_modified: SystemTime,
}

/// Information about a single cached revision of a repository, including the snapshot path and its files.
pub struct CachedRevisionInfo {
    pub commit_hash: String,
    pub snapshot_path: PathBuf,
    pub files: Vec<CachedFileInfo>,
    pub size_on_disk: u64,
    pub refs: Vec<String>,
    pub last_modified: SystemTime,
}

/// Information about a single cached repository, aggregating its revisions and disk usage.
pub struct CachedRepoInfo {
    pub repo_id: String,
    pub repo_type: RepoType,
    pub repo_path: PathBuf,
    pub revisions: Vec<CachedRevisionInfo>,
    pub nb_files: usize,
    pub size_on_disk: u64,
    pub last_accessed: SystemTime,
    pub last_modified: SystemTime,
}

/// Snapshot of the local Hugging Face cache directory.
///
/// Returned by [`HFClient::scan_cache`](crate::client::HFClient::scan_cache); aggregates every
/// cached repository along with total disk usage and any warnings encountered during scanning.
pub struct HFCacheInfo {
    pub cache_dir: PathBuf,
    pub repos: Vec<CachedRepoInfo>,
    pub size_on_disk: u64,
    pub warnings: Vec<String>,
}
