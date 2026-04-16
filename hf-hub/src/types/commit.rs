use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct CommitAuthor {
    pub user: Option<String>,
    pub name: Option<String>,
    pub email: Option<String>,
}

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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefInfo {
    pub name: String,
    #[serde(rename = "ref")]
    pub git_ref: String,
    pub target_commit: String,
}

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
