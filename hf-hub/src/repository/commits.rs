//! Commit history, refs, and revision comparison helpers for repositories.
//!
//! This module groups three related workflows:
//!
//! - Use [`HFRepository::list_commits`] to walk commit history and [`HFRepository::list_refs`] to inspect branches,
//!   tags, convert refs, and optional pull-request refs.
//! - Use [`HFRepository::get_commit_diff`] for the Hub's non-raw compare payload as text.
//! - Use [`HFRepository::get_raw_diff`] for the full raw diff text, or [`HFRepository::get_raw_diff_stream`] to parse
//!   that raw diff incrementally into [`HFFileDiff`] entries.
//!
//! The same module also hosts branch and tag creation/deletion helpers because
//! they operate on the same repository revision namespace.

use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
use serde::Deserialize;
use typed_builder::TypedBuilder;
use url::Url;

use super::HFRepository;
use super::diff::{self, HFFileDiff};
use crate::error::HFResult;
use crate::{constants, retry};

/// Author entry attached to a commit, as returned by the commit history endpoint.
///
/// All fields are optional because the Hub only surfaces the identifying fields it has
/// (a linked Hub user, or the raw git name/email).
#[derive(Debug, Clone, Deserialize)]
pub struct CommitAuthor {
    /// Hub username, when the commit author is linked to a Hub account.
    pub user: Option<String>,
    /// Git author name as recorded on the commit.
    pub name: Option<String>,
    /// Git author email as recorded on the commit.
    pub email: Option<String>,
}

/// A single commit entry returned by the commit history endpoint.
///
/// Returned by [`HFRepository::list_commits`].
#[derive(Debug, Clone, Deserialize)]
pub struct GitCommitInfo {
    /// Full commit SHA.
    pub id: String,
    /// Commit authors as returned by the Hub.
    pub authors: Vec<CommitAuthor>,
    /// Commit timestamp in ISO 8601 format, when available.
    pub date: Option<String>,
    /// Commit title/summary line.
    pub title: String,
    /// Full commit message.
    pub message: String,
    /// Parent commit SHAs.
    #[serde(default)]
    pub parents: Vec<String>,
}

/// A single git ref (branch, tag, convert, or pull-request ref) and the commit it points to.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefInfo {
    /// Short ref name such as `"main"` or `"v1.0.0"`.
    pub name: String,
    /// Full git ref name such as `"refs/heads/main"`.
    #[serde(rename = "ref")]
    pub git_ref: String,
    /// Commit SHA the ref currently points to.
    pub target_commit: String,
}

/// All git refs on a repository — branches, tags, converts, and pull-request refs.
///
/// Returned by [`HFRepository::list_refs`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefs {
    /// Branch refs on the repository.
    pub branches: Vec<GitRefInfo>,
    /// Tag refs on the repository.
    pub tags: Vec<GitRefInfo>,
    /// Convert refs exposed by the Hub, when present.
    #[serde(default)]
    pub converts: Vec<GitRefInfo>,
    /// Pull-request refs, only populated when requested.
    #[serde(default, rename = "pullRequests")]
    pub pull_requests: Vec<GitRefInfo>,
}

/// A single file entry in the Hub's non-raw compare payload.
///
/// This type is useful if you want to deserialize the response body returned by
/// [`HFRepository::get_commit_diff`] yourself.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffEntry {
    /// Destination path for the changed file, when present.
    pub path: Option<String>,
    /// Previous path for renames or moves, when present.
    pub old_path: Option<String>,
    /// Hub-provided status string for the change.
    pub status: Option<String>,
}

/// Parameters for listing commits on a repository revision.
///
/// Used with [`HFRepository::list_commits`].
#[derive(Default, TypedBuilder)]
pub struct RepoListCommitsParams {
    /// Git revision (branch, tag, or commit SHA) to list commits from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Maximum number of commits to return.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for listing refs (branches, tags, ...) on a repository.
///
/// Used with [`HFRepository::list_refs`].
#[derive(Default, TypedBuilder)]
pub struct RepoListRefsParams {
    /// Whether to include pull request refs in the listing.
    #[builder(default)]
    pub include_pull_requests: bool,
}

/// Parameters for fetching the Hub's non-raw compare payload between a revision
/// and its parent.
///
/// Used with [`HFRepository::get_commit_diff`].
#[derive(TypedBuilder)]
pub struct RepoGetCommitDiffParams {
    /// Revision to compare against the parent (branch, tag, or commit SHA).
    #[builder(setter(into))]
    pub compare: String,
}

/// Parameters for fetching the raw git diff between a revision and its parent.
///
/// Used with [`HFRepository::get_raw_diff`]
/// and [`HFRepository::get_raw_diff_stream`].
#[derive(TypedBuilder)]
pub struct RepoGetRawDiffParams {
    /// Revision to compare against the parent (branch, tag, or commit SHA).
    #[builder(setter(into))]
    pub compare: String,
}

/// Parameters for creating a branch on a repository.
///
/// Used with [`HFRepository::create_branch`].
#[derive(TypedBuilder)]
pub struct RepoCreateBranchParams {
    /// Name of the branch to create.
    #[builder(setter(into))]
    pub branch: String,
    /// Revision to branch from. Defaults to the current main branch head.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for deleting a branch on a repository.
///
/// Used with [`HFRepository::delete_branch`].
#[derive(TypedBuilder)]
pub struct RepoDeleteBranchParams {
    /// Name of the branch to delete.
    #[builder(setter(into))]
    pub branch: String,
}

/// Parameters for creating a tag on a repository.
///
/// Used with [`HFRepository::create_tag`].
#[derive(TypedBuilder)]
pub struct RepoCreateTagParams {
    /// Name of the tag to create.
    #[builder(setter(into))]
    pub tag: String,
    /// Revision to tag. Defaults to the current main branch head.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Annotation message for the tag.
    #[builder(default, setter(into, strip_option))]
    pub message: Option<String>,
}

/// Parameters for deleting a tag on a repository.
///
/// Used with [`HFRepository::delete_tag`].
#[derive(TypedBuilder)]
pub struct RepoDeleteTagParams {
    /// Name of the tag to delete.
    #[builder(setter(into))]
    pub tag: String,
}

impl HFRepository {
    /// Stream commit history for the repository at a given revision.
    ///
    /// Returns `HFResult<impl Stream<Item = HFResult<GitCommitInfo>>>`. Use `limit` to limit
    /// the total number of commits yielded.
    pub fn list_commits(
        &self,
        params: RepoListCommitsParams,
    ) -> HFResult<impl Stream<Item = HFResult<GitCommitInfo>> + '_> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str =
            format!("{}/commits/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);
        let url = Url::parse(&url_str)?;
        Ok(self.hf_client.paginate(url, vec![], params.limit))
    }

    /// Fetch all branches, tags, and optionally pull request refs for the
    /// repository.
    ///
    /// Pull-request refs are omitted unless
    /// [`RepoListRefsParams::include_pull_requests`] is enabled.
    ///
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/refs
    pub async fn list_refs(&self, params: RepoListRefsParams) -> HFResult<GitRefs> {
        let url = format!("{}/refs", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()));
        let mut query: Vec<(&str, String)> = Vec::new();
        if params.include_pull_requests {
            query.push(("include_prs", "1".into()));
        }

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&query)
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch the Hub's non-raw compare payload as text.
    ///
    /// This returns the response body from the standard `/compare/{compare}`
    /// endpoint. Use [`HFRepository::get_raw_diff`] if you need raw git-style
    /// diff text, or [`HFRepository::get_raw_diff_stream`] if you want parsed
    /// [`HFFileDiff`] entries.
    ///
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/compare/{compare}
    pub async fn get_commit_diff(&self, params: RepoGetCommitDiffParams) -> HFResult<String> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.text().await?)
    }

    /// Fetch the raw diff payload between two revisions as a string.
    ///
    /// Prefer [`HFRepository::get_raw_diff_stream`] when you want file-level
    /// metadata without buffering the entire diff response in memory.
    ///
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/compare/{compare}?raw=true
    pub async fn get_raw_diff(&self, params: RepoGetRawDiffParams) -> HFResult<String> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&[("raw", "true")])
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.text().await?)
    }

    /// Fetch the raw diff between two revisions as a parsed stream of [`HFFileDiff`] entries.
    ///
    /// Each item in the returned stream is one parsed diff entry. Parse errors
    /// are logged as warnings and yielded as `Err` items.
    ///
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/compare/{compare}?raw=true
    pub async fn get_raw_diff_stream(
        &self,
        params: RepoGetRawDiffParams,
    ) -> HFResult<impl Stream<Item = HFResult<HFFileDiff>> + '_> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&[("raw", "true")])
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        let byte_stream = response.bytes_stream().map(|r| r.map_err(std::io::Error::other));
        Ok(diff::stream_raw_diff(byte_stream).map_err(Into::into))
    }

    /// Create a new branch, optionally starting from a specific revision.
    /// Endpoint: POST /api/{repo_type}s/{repo_id}/branch/{branch}
    pub async fn create_branch(&self, params: RepoCreateBranchParams) -> HFResult<()> {
        let url =
            format!("{}/branch/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.branch);

        let mut body = serde_json::Map::new();
        if let Some(ref revision) = params.revision {
            body.insert("startingPoint".into(), serde_json::Value::String(revision.clone()));
        }

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a branch from the repository.
    /// Endpoint: DELETE /api/{repo_type}s/{repo_id}/branch/{branch}
    pub async fn delete_branch(&self, params: RepoDeleteBranchParams) -> HFResult<()> {
        let url =
            format!("{}/branch/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.branch);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().delete(&url).headers(headers.clone()).send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Create a lightweight or annotated tag, optionally at a specific revision.
    /// Endpoint: POST /api/{repo_type}s/{repo_id}/tag/{revision}
    pub async fn create_tag(&self, params: RepoCreateTagParams) -> HFResult<()> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = format!("{}/tag/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);

        let mut body = serde_json::json!({ "tag": params.tag });
        if let Some(ref message) = params.message {
            body["message"] = serde_json::Value::String(message.clone());
        }

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a tag from the repository.
    /// Endpoint: DELETE /api/{repo_type}s/{repo_id}/tag/{tag}
    pub async fn delete_tag(&self, params: RepoDeleteTagParams) -> HFResult<()> {
        let url = format!("{}/tag/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.tag);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().delete(&url).headers(headers.clone()).send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }
}

sync_api! {
    impl HFRepository -> HFRepositorySync {
        fn list_refs(&self, params: RepoListRefsParams) -> HFResult<GitRefs>;
        fn get_commit_diff(&self, params: RepoGetCommitDiffParams) -> HFResult<String>;
        fn get_raw_diff(&self, params: RepoGetRawDiffParams) -> HFResult<String>;
        fn create_branch(&self, params: RepoCreateBranchParams) -> HFResult<()>;
        fn delete_branch(&self, params: RepoDeleteBranchParams) -> HFResult<()>;
        fn create_tag(&self, params: RepoCreateTagParams) -> HFResult<()>;
        fn delete_tag(&self, params: RepoDeleteTagParams) -> HFResult<()>;
    }
}

sync_api_stream! {
    impl HFRepository -> HFRepositorySync {
        fn list_commits(&self, params: RepoListCommitsParams) -> GitCommitInfo;
    }
}

sync_api_async_stream! {
    impl HFRepository -> HFRepositorySync {
        fn get_raw_diff_stream(&self, params: RepoGetRawDiffParams) -> HFFileDiff;
    }
}
