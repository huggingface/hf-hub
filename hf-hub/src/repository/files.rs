//! Repository file listing, metadata, download, and upload APIs.
//!
//! Common entry points:
//!
//! - Use [`HFRepository::list_files`] when you only need file paths.
//! - Use [`HFRepository::list_tree`] when you need file and directory entries, or [`HFRepository::get_paths_info`] /
//!   [`HFRepository::get_file_metadata`] for targeted lookups.
//! - Use [`HFRepository::download_file`] to write one file to disk, [`HFRepository::download_file_stream`] to stream
//!   bytes, [`HFRepository::download_file_to_bytes`] to collect a file into memory, and
//!   [`HFRepository::snapshot_download`] to fetch a whole revision.
//! - Use [`HFRepository::upload_file`] and [`HFRepository::upload_folder`] for convenience, or
//!   [`HFRepository::create_commit`] when you need an explicit set of add/delete operations in one commit.
//!
//! The public types in this module are shared across the listing, download, and
//! upload helpers implemented on [`HFRepository`].

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

#[allow(unused_imports)] // used by intra-doc links
use super::HFRepository;
use crate::constants;
use crate::progress::Progress;

/// LFS metadata attached to a repository file, when the file is stored in Git LFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobLfsInfo {
    /// Original file size in bytes, when reported by the Hub.
    pub size: Option<u64>,
    /// SHA-256 object id of the LFS payload.
    pub sha256: Option<String>,
    /// Size in bytes of the LFS pointer file stored in git.
    pub pointer_size: Option<u64>,
}

/// Summary of the last commit that touched a tree entry, included when expanded metadata is requested.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastCommitInfo {
    /// Commit SHA, when available.
    pub id: Option<String>,
    /// Commit title/summary line.
    pub title: Option<String>,
    /// Commit timestamp in ISO 8601 format, when available.
    pub date: Option<String>,
}

/// Metadata returned from a HEAD request on a file's resolve URL.
///
/// Produced by [`HFRepository::get_file_metadata`]
/// and used internally by snapshot downloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadataInfo {
    /// Path of the file within the repository.
    pub filename: String,
    /// ETag of the file content (normalized, with weak prefix and quotes stripped).
    pub etag: String,
    /// Commit hash the revision resolved to (from the `X-Repo-Commit` header).
    pub commit_hash: String,
    /// Xet content hash if the file is stored in Xet (from the `X-Xet-Hash` header).
    pub xet_hash: Option<String>,
    /// File size in bytes. Falls back to `0` if neither `X-Linked-Size` nor `Content-Length`
    /// is present on the response.
    pub file_size: u64,
}

/// File or directory entry returned by repository tree/listing APIs.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RepoTreeEntry {
    /// A file entry in the repository tree.
    File {
        /// Object id reported by the Hub for this entry.
        oid: String,
        /// File size in bytes.
        size: u64,
        /// Repository-relative path.
        path: String,
        /// LFS metadata, when the file is LFS-backed.
        lfs: Option<BlobLfsInfo>,
        /// Last-commit summary, only when expanded metadata is requested.
        #[serde(default, rename = "lastCommit")]
        last_commit: Option<LastCommitInfo>,
    },
    /// A directory entry in the repository tree.
    Directory {
        /// Object id reported by the Hub for this entry.
        oid: String,
        /// Repository-relative path.
        path: String,
    },
}

/// Response body returned after creating a commit.
///
/// Includes URLs for the commit and any PR that was opened, along with the commit OID
/// when present. Returned by [`HFRepository::create_commit`]
/// and related upload/delete helpers.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitInfo {
    /// URL of the created commit on the Hub, when available.
    pub commit_url: Option<String>,
    /// Commit message recorded for the operation.
    pub commit_message: Option<String>,
    /// Commit description/body, when provided.
    pub commit_description: Option<String>,
    /// Commit SHA, when returned by the API.
    pub commit_oid: Option<String>,
    /// Pull-request URL, when `create_pr` was enabled and a PR was opened.
    pub pr_url: Option<String>,
    /// Pull-request number, when `create_pr` was enabled and a PR was opened.
    pub pr_num: Option<u64>,
}

/// File mutation included in [`HFRepository::create_commit`].
#[derive(Debug, Clone)]
pub enum CommitOperation {
    /// Add or replace a file in the repository.
    Add {
        /// Destination path within the repository.
        path_in_repo: String,
        /// Source of the uploaded contents.
        source: AddSource,
    },
    /// Delete a file from the repository.
    Delete {
        /// Repository-relative path to remove.
        path_in_repo: String,
    },
}

impl CommitOperation {
    /// Create an add operation backed by a local file path.
    pub fn add_file(path_in_repo: impl Into<String>, source: impl Into<PathBuf>) -> Self {
        CommitOperation::Add {
            path_in_repo: path_in_repo.into(),
            source: AddSource::File(source.into()),
        }
    }

    /// Create an add operation backed by in-memory bytes.
    pub fn add_bytes(path_in_repo: impl Into<String>, source: impl Into<Vec<u8>>) -> Self {
        CommitOperation::Add {
            path_in_repo: path_in_repo.into(),
            source: AddSource::Bytes(source.into()),
        }
    }

    /// Create a delete operation for a repository path.
    pub fn delete(path_in_repo: impl Into<String>) -> Self {
        CommitOperation::Delete {
            path_in_repo: path_in_repo.into(),
        }
    }
}

/// Source of content for a [`CommitOperation::Add`] operation.
#[derive(Debug, Clone)]
pub enum AddSource {
    /// Read file contents from this local path when creating the commit.
    File(PathBuf),
    /// Use these bytes as the file contents.
    Bytes(Vec<u8>),
}

/// Parameters for listing files in a repository.
///
/// Used with [`HFRepository::list_files`].
#[derive(Default, TypedBuilder)]
pub struct RepoListFilesParams {
    /// Git revision to list files from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for listing the tree of entries in a repository.
///
/// Used with [`HFRepository::list_tree`].
#[derive(Default, TypedBuilder)]
pub struct RepoListTreeParams {
    /// Git revision to list the tree from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Whether to list files recursively in subdirectories.
    #[builder(default)]
    pub recursive: bool,
    /// Whether to include expanded metadata (size, LFS info) for each entry.
    #[builder(default)]
    pub expand: bool,
    /// Maximum number of tree entries to return.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for fetching metadata about a single file in a repository.
///
/// Used with [`HFRepository::get_file_metadata`].
#[derive(TypedBuilder)]
pub struct RepoGetFileMetadataParams {
    /// Path of the file within the repository.
    #[builder(setter(into))]
    pub filepath: String,
    /// Git revision to query. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for fetching info about a set of paths in a repository.
///
/// Used with [`HFRepository::get_paths_info`].
#[derive(TypedBuilder)]
pub struct RepoGetPathsInfoParams {
    /// List of file paths within the repository to retrieve info for.
    pub paths: Vec<String>,
    /// Git revision to query. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for downloading a single file from a repository.
///
/// Used with [`HFRepository::download_file`].
#[derive(TypedBuilder)]
pub struct RepoDownloadFileParams {
    /// Path of the file to download within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Local directory to download the file into. When set, the file is saved with its repo path structure.
    #[builder(default, setter(strip_option))]
    pub local_dir: Option<PathBuf>,
    /// Git revision to download from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// If `true`, re-download the file even if a cached copy exists.
    #[builder(default, setter(strip_option))]
    pub force_download: Option<bool>,
    /// If `true`, only return the file if it is already cached locally; never make a network request.
    #[builder(default, setter(strip_option))]
    pub local_files_only: Option<bool>,
    /// Optional progress handler for tracking download progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for streaming a file download from a repository.
///
/// Used with [`HFRepository::download_file_stream`]
/// and [`HFRepository::download_file_to_bytes`].
#[derive(TypedBuilder)]
pub struct RepoDownloadFileStreamParams {
    /// Path of the file to stream within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Git revision to stream from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Byte range to request (HTTP Range header). Useful for partial downloads.
    #[builder(default, setter(strip_option))]
    pub range: Option<std::ops::Range<u64>>,
}

/// Alias of [`RepoDownloadFileStreamParams`] for
/// [`HFRepository::download_file_to_bytes`].
pub type RepoDownloadFileToBytesParams = RepoDownloadFileStreamParams;
/// Builder alias for [`RepoDownloadFileToBytesParams`].
pub type RepoDownloadFileToBytesParamsBuilder = RepoDownloadFileStreamParamsBuilder;

/// Parameters for downloading a full repository snapshot.
///
/// Used with [`HFRepository::snapshot_download`].
#[derive(Default, TypedBuilder)]
pub struct RepoSnapshotDownloadParams {
    /// Git revision to download. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Glob patterns for files to include in the download. Only matching files are downloaded.
    #[builder(default, setter(strip_option))]
    pub allow_patterns: Option<Vec<String>>,
    /// Glob patterns for files to exclude from the download.
    #[builder(default, setter(strip_option))]
    pub ignore_patterns: Option<Vec<String>>,
    /// Local directory to download the snapshot into.
    #[builder(default, setter(strip_option))]
    pub local_dir: Option<PathBuf>,
    /// If `true`, re-download all files even if cached copies exist.
    #[builder(default, setter(strip_option))]
    pub force_download: Option<bool>,
    /// If `true`, only return files already cached locally; never make network requests.
    #[builder(default, setter(strip_option))]
    pub local_files_only: Option<bool>,
    /// Maximum number of concurrent file downloads.
    #[builder(default, setter(strip_option))]
    pub max_workers: Option<usize>,
    /// Optional progress handler for tracking download progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for uploading a single file to a repository.
///
/// Used with [`HFRepository::upload_file`].
#[derive(TypedBuilder)]
pub struct RepoUploadFileParams {
    /// Source of the file content to upload (bytes or file path).
    pub source: AddSource,
    /// Destination path within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to upload to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the upload.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Expected parent commit SHA. The upload fails if the branch head has moved past this commit.
    #[builder(default, setter(into, strip_option))]
    pub parent_commit: Option<String>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for uploading a local folder to a repository.
///
/// Used with [`HFRepository::upload_folder`].
#[derive(TypedBuilder)]
pub struct RepoUploadFolderParams {
    /// Local folder path to upload.
    #[builder(setter(into))]
    pub folder_path: PathBuf,
    /// Destination directory within the repository. Defaults to the repo root.
    #[builder(default, setter(into, strip_option))]
    pub path_in_repo: Option<String>,
    /// Git revision (branch) to upload to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the upload.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Glob patterns for files to include from the local folder.
    #[builder(default, setter(strip_option))]
    pub allow_patterns: Option<Vec<String>>,
    /// Glob patterns for files to exclude from the local folder.
    #[builder(default, setter(strip_option))]
    pub ignore_patterns: Option<Vec<String>>,
    /// Glob patterns for remote files to delete as part of the upload commit.
    #[builder(default, setter(strip_option))]
    pub delete_patterns: Option<Vec<String>>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for deleting a single file from a repository.
///
/// Used with [`HFRepository::delete_file`].
#[derive(TypedBuilder)]
pub struct RepoDeleteFileParams {
    /// Path of the file to delete within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to delete from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the deletion.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
}

/// Parameters for deleting a folder from a repository.
///
/// Used with [`HFRepository::delete_folder`].
#[derive(TypedBuilder)]
pub struct RepoDeleteFolderParams {
    /// Path of the folder to delete within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to delete from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the deletion.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
}

/// Parameters for creating a commit composed of multiple file operations.
///
/// Used with [`HFRepository::create_commit`]. Prefer the upload/delete
/// convenience helpers when you only need a single common operation.
#[derive(TypedBuilder)]
pub struct RepoCreateCommitParams {
    /// List of file operations to include in the commit.
    pub operations: Vec<CommitOperation>,
    /// Commit message.
    #[builder(setter(into))]
    pub commit_message: String,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// Git revision (branch) to commit to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Expected parent commit SHA. The commit fails if the branch head has moved past this commit.
    #[builder(default, setter(into, strip_option))]
    pub parent_commit: Option<String>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

pub(super) fn extract_etag(response: &reqwest::Response) -> Option<String> {
    let headers = response.headers();
    let raw = headers
        .get(constants::HEADER_X_LINKED_ETAG)
        .or_else(|| headers.get(reqwest::header::ETAG))
        .and_then(|v| v.to_str().ok())?;
    let normalized = raw.strip_prefix("W/").unwrap_or(raw);
    Some(normalized.trim_matches('"').to_string())
}

pub(super) fn extract_commit_hash(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(constants::HEADER_X_REPO_COMMIT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

pub(crate) fn extract_file_size(response: &reqwest::Response) -> Option<u64> {
    let headers = response.headers();
    headers
        .get(constants::HEADER_X_LINKED_SIZE)
        .or_else(|| headers.get(reqwest::header::CONTENT_LENGTH))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
}

pub(crate) fn extract_xet_hash(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(constants::HEADER_X_XET_HASH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Check if a path matches any of the given glob patterns using the `globset` crate.
pub(super) fn matches_any_glob(patterns: &[String], path: &str) -> bool {
    patterns.iter().any(|p| {
        globset::Glob::new(p)
            .ok()
            .map(|g| g.compile_matcher().is_match(path))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::RepoTreeEntry;

    #[test]
    fn test_repo_tree_entry_deserialize_file() {
        let json = r#"{"type":"file","oid":"abc123","size":100,"path":"test.txt"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::File { path, size, .. } => {
                assert_eq!(path, "test.txt");
                assert_eq!(size, 100);
            },
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_repo_tree_entry_deserialize_directory() {
        let json = r#"{"type":"directory","oid":"def456","path":"src"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::Directory { path, .. } => {
                assert_eq!(path, "src");
            },
            _ => panic!("Expected Directory variant"),
        }
    }
}
