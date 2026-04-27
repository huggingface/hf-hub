use bon::bon;
use futures::stream::{Stream, StreamExt};
use reqwest::Url;

use super::files::{extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash};
use super::{FileMetadataInfo, HFRepository, RepoTreeEntry};
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

#[bon]
impl HFRepository {
    /// Return a flat list of file paths in the repository at the given revision.
    ///
    /// This is a convenience wrapper around a recursive [`HFRepository::list_tree`]
    /// call that drops directory entries and returns only file paths.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision (branch, tag, or commit SHA). Defaults to the main branch.
    #[builder(finish_fn = send)]
    pub async fn list_files(&self, #[builder(into)] revision: Option<String>) -> HFResult<Vec<String>> {
        let stream = self.list_tree().maybe_revision(revision).recursive(true).send()?;
        futures::pin_mut!(stream);

        let mut files = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, .. } = entry {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// Stream file and directory entries in the repository tree.
    ///
    /// Returns `HFResult<impl Stream<Item = HFResult<RepoTreeEntry>>>`.
    ///
    /// Use [`HFRepository::list_files`] when you only need paths, or
    /// [`HFRepository::get_paths_info`] when you already know the exact paths
    /// you want to inspect.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision to list. Defaults to the main branch.
    /// - `recursive` (default `false`): traverse subdirectories.
    /// - `expand` (default `false`): include per-file metadata such as size, LFS info, and last-commit summaries.
    /// - `limit`: cap the total number of entries yielded.
    #[builder(finish_fn = send)]
    pub fn list_tree(
        &self,
        #[builder(into)] revision: Option<String>,
        #[builder(default)] recursive: bool,
        #[builder(default)] expand: bool,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<RepoTreeEntry>> + '_> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str = format!("{}/tree/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);
        let url = Url::parse(&url_str)?;

        let mut query: Vec<(String, String)> = Vec::new();
        if recursive {
            query.push(("recursive".into(), "true".into()));
        }
        if expand {
            query.push(("expand".into(), "true".into()));
        }

        Ok(self.hf_client.paginate(url, query, limit))
    }

    /// Get info about specific paths in a repository.
    ///
    /// Prefer this over [`HFRepository::list_tree`] when you already know the
    /// small set of paths you want to inspect.
    ///
    /// Endpoint: `POST /api/{repo_type}s/{repo_id}/paths-info/{revision}`.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths in the repository to fetch info for.
    /// - `revision`: Git revision. Defaults to the main branch.
    #[builder(finish_fn = send)]
    pub async fn get_paths_info(
        &self,
        paths: Vec<String>,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url =
            format!("{}/paths-info/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);

        let body = serde_json::json!({ "paths": paths });

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
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Entry { path: paths.join(", ") })
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch metadata for a single file via a HEAD request on its resolve URL.
    ///
    /// Returns the resolved commit hash, ETag, file size, and (if the file is Xet-backed)
    /// the Xet content hash — without downloading the file contents.
    ///
    /// Endpoint: `HEAD {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filepath}`.
    ///
    /// # Parameters
    ///
    /// - `filepath` (required): path of the file to inspect within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    #[builder(finish_fn = send)]
    pub async fn get_file_metadata(
        &self,
        #[builder(into)] filepath: String,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<FileMetadataInfo> {
        let filename = filepath.clone();
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &repo_path, revision, &filename);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().head(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Entry { path: filename.clone() })
            .await?;

        let etag =
            extract_etag(&response).ok_or_else(|| HFError::Other(format!("Missing ETag header for {filename}")))?;
        let commit_hash = extract_commit_hash(&response)
            .ok_or_else(|| HFError::Other(format!("Missing X-Repo-Commit header for {filename}")))?;
        let xet_hash = extract_xet_hash(&response);
        let file_size = extract_file_size(&response).unwrap_or_else(|| {
            tracing::warn!(
                file = %filename,
                "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0"
            );
            0
        });

        Ok(FileMetadataInfo {
            filename,
            etag,
            commit_hash,
            xet_hash,
            file_size,
        })
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync {
    /// Blocking counterpart of [`HFRepository::list_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn list_files(&self, #[builder(into)] revision: Option<String>) -> HFResult<Vec<String>> {
        self.runtime.block_on(self.inner.list_files().maybe_revision(revision).send())
    }

    /// Blocking counterpart of [`HFRepository::list_tree`]. Returns the collected stream as a
    /// `Vec<RepoTreeEntry>`.
    #[builder(finish_fn = send)]
    pub fn list_tree(
        &self,
        #[builder(into)] revision: Option<String>,
        #[builder(default)] recursive: bool,
        #[builder(default)] expand: bool,
        limit: Option<usize>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_tree()
                .maybe_revision(revision)
                .recursive(recursive)
                .expand(expand)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFRepository::get_paths_info`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send)]
    pub fn get_paths_info(
        &self,
        paths: Vec<String>,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        self.runtime
            .block_on(self.inner.get_paths_info().paths(paths).maybe_revision(revision).send())
    }

    /// Blocking counterpart of [`HFRepository::get_file_metadata`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send)]
    pub fn get_file_metadata(
        &self,
        #[builder(into)] filepath: String,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<FileMetadataInfo> {
        self.runtime.block_on(
            self.inner
                .get_file_metadata()
                .filepath(filepath)
                .maybe_revision(revision)
                .send(),
        )
    }
}
