use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
use url::Url;

use crate::constants;
use crate::diff::HFFileDiff;
use crate::error::Result;
use crate::repository::HFRepository;
use crate::types::{
    GitCommitInfo, GitRefs, RepoCreateBranchParams, RepoCreateTagParams, RepoDeleteBranchParams, RepoDeleteTagParams,
    RepoGetCommitDiffParams, RepoGetRawDiffParams, RepoListCommitsParams, RepoListRefsParams,
};

impl HFRepository {
    /// Stream commit history for the repository at a given revision.
    ///
    /// Returns `Result<impl Stream<Item = Result<GitCommitInfo>>>`. Use `limit` to limit
    /// the total number of commits yielded.
    pub fn list_commits(
        &self,
        params: &RepoListCommitsParams,
    ) -> Result<impl Stream<Item = Result<GitCommitInfo>> + '_> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str =
            format!("{}/commits/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);
        let url = Url::parse(&url_str)?;
        Ok(self.hf_client.paginate(url, vec![], params.limit))
    }

    /// Fetch all branches, tags, and optionally pull request refs for the repository.
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/refs
    pub async fn list_refs(&self, params: &RepoListRefsParams) -> Result<GitRefs> {
        let url = format!("{}/refs", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()));
        let mut query: Vec<(&str, String)> = Vec::new();
        if params.include_pull_requests {
            query.push(("include_prs", "1".into()));
        }

        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .query(&query)
            .send()
            .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch a structured diff between two revisions (HEAD..compare or a commit SHA).
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/compare/{compare}
    pub async fn get_commit_diff(&self, params: &RepoGetCommitDiffParams) -> Result<String> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.text().await?)
    }

    /// Fetch the raw unified diff between two revisions as a string.
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/compare/{compare}?raw=true
    pub async fn get_raw_diff(&self, params: &RepoGetRawDiffParams) -> Result<String> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .query(&[("raw", "true")])
            .send()
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
        params: &RepoGetRawDiffParams,
    ) -> Result<impl Stream<Item = Result<HFFileDiff>> + '_> {
        let url =
            format!("{}/compare/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.compare);

        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .query(&[("raw", "true")])
            .send()
            .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        let byte_stream = response.bytes_stream().map(|r| r.map_err(std::io::Error::other));
        Ok(crate::diff::stream_raw_diff(byte_stream).map_err(Into::into))
    }

    /// Create a new branch, optionally starting from a specific revision.
    /// Endpoint: POST /api/{repo_type}s/{repo_id}/branch/{branch}
    pub async fn create_branch(&self, params: &RepoCreateBranchParams) -> Result<()> {
        let url =
            format!("{}/branch/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.branch);

        let mut body = serde_json::Map::new();
        if let Some(ref revision) = params.revision {
            body.insert("startingPoint".into(), serde_json::Value::String(revision.clone()));
        }

        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a branch from the repository.
    /// Endpoint: DELETE /api/{repo_type}s/{repo_id}/branch/{branch}
    pub async fn delete_branch(&self, params: &RepoDeleteBranchParams) -> Result<()> {
        let url =
            format!("{}/branch/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.branch);

        let response = self
            .hf_client
            .http_client()
            .delete(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Create a lightweight or annotated tag, optionally at a specific revision.
    /// Endpoint: POST /api/{repo_type}s/{repo_id}/tag/{revision}
    pub async fn create_tag(&self, params: &RepoCreateTagParams) -> Result<()> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = format!("{}/tag/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);

        let mut body = serde_json::json!({ "tag": params.tag });
        if let Some(ref message) = params.message {
            body["message"] = serde_json::Value::String(message.clone());
        }

        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a tag from the repository.
    /// Endpoint: DELETE /api/{repo_type}s/{repo_id}/tag/{tag}
    pub async fn delete_tag(&self, params: &RepoDeleteTagParams) -> Result<()> {
        let url = format!("{}/tag/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.tag);

        let response = self
            .hf_client
            .http_client()
            .delete(&url)
            .headers(self.hf_client.auth_headers())
            .send()
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
        fn list_refs(&self, params: &RepoListRefsParams) -> Result<GitRefs>;
        fn get_commit_diff(&self, params: &RepoGetCommitDiffParams) -> Result<String>;
        fn get_raw_diff(&self, params: &RepoGetRawDiffParams) -> Result<String>;
        fn create_branch(&self, params: &RepoCreateBranchParams) -> Result<()>;
        fn delete_branch(&self, params: &RepoDeleteBranchParams) -> Result<()>;
        fn create_tag(&self, params: &RepoCreateTagParams) -> Result<()>;
        fn delete_tag(&self, params: &RepoDeleteTagParams) -> Result<()>;
    }
}

sync_api_stream! {
    impl HFRepository -> HFRepositorySync {
        fn list_commits(&self, params: &RepoListCommitsParams) -> GitCommitInfo;
    }
}

sync_api_async_stream! {
    impl HFRepository -> HFRepositorySync {
        fn get_raw_diff_stream(&self, params: &RepoGetRawDiffParams) -> HFFileDiff;
    }
}
