use std::path::PathBuf;

use serde::Deserialize;

/// Author entry attached to a commit, as returned by the commit history endpoint.
///
/// All fields are optional because the Hub only surfaces the identifying fields it has
/// (a linked Hub user, or the raw git name/email).
#[derive(Debug, Clone, Deserialize)]
pub struct CommitAuthor {
    pub user: Option<String>,
    pub name: Option<String>,
    pub email: Option<String>,
}

/// A single commit entry returned by the commit history endpoint.
///
/// Returned by [`HFRepository::list_commits`](crate::repository::HFRepository::list_commits).
#[derive(Debug, Clone, Deserialize)]
pub struct GitCommitInfo {
    pub id: String,
    pub authors: Vec<CommitAuthor>,
    pub date: Option<String>,
    pub title: String,
    pub message: String,
    #[serde(default)]
    pub parents: Vec<String>,
}

/// A single git ref (branch, tag, convert, or pull-request ref) and the commit it points to.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefInfo {
    pub name: String,
    #[serde(rename = "ref")]
    pub git_ref: String,
    pub target_commit: String,
}

/// All git refs on a repository — branches, tags, converts, and pull-request refs.
///
/// Returned by [`HFRepository::list_refs`](crate::repository::HFRepository::list_refs).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefs {
    pub branches: Vec<GitRefInfo>,
    pub tags: Vec<GitRefInfo>,
    #[serde(default)]
    pub converts: Vec<GitRefInfo>,
    #[serde(default, rename = "pullRequests")]
    pub pull_requests: Vec<GitRefInfo>,
}

/// Response body returned after creating a commit.
///
/// Includes URLs for the commit and any PR that was opened, along with the commit OID
/// when present. Returned by [`HFRepository::create_commit`](crate::repository::HFRepository::create_commit)
/// and related upload/delete helpers.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitInfo {
    pub commit_url: Option<String>,
    pub commit_message: Option<String>,
    pub commit_description: Option<String>,
    pub commit_oid: Option<String>,
    pub pr_url: Option<String>,
    pub pr_num: Option<u64>,
}

/// A single entry in a commit diff
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffEntry {
    pub path: Option<String>,
    pub old_path: Option<String>,
    pub status: Option<String>,
}

/// Describes a file mutation in a commit
#[derive(Debug, Clone)]
pub enum CommitOperation {
    /// Upload a file (from path or bytes)
    Add { path_in_repo: String, source: AddSource },
    /// Delete a file or folder
    Delete { path_in_repo: String },
}

/// Source of content for an add operation
#[derive(Debug, Clone)]
pub enum AddSource {
    File(PathBuf),
    Bytes(Vec<u8>),
}
