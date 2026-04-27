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

#[allow(unused_imports)] // used by intra-doc links
use super::HFRepository;
use crate::constants;

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
