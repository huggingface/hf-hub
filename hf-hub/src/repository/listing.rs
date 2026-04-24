use futures::stream::{Stream, StreamExt};
use reqwest::Url;

use super::files::{extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash};
use super::{
    FileMetadataInfo, HFRepository, RepoGetFileMetadataParams, RepoGetPathsInfoParams, RepoListFilesParams,
    RepoListTreeParams, RepoTreeEntry,
};
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

impl HFRepository {
    /// Return a flat list of file paths in the repository at the given revision.
    ///
    /// This is a convenience wrapper around a recursive [`HFRepository::list_tree`]
    /// call that drops directory entries and returns only file paths.
    pub async fn list_files(&self, params: RepoListFilesParams) -> HFResult<Vec<String>> {
        let revision = params.revision.clone();
        let stream = self.list_tree(RepoListTreeParams {
            revision,
            recursive: true,
            expand: false,
            limit: None,
        })?;
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
    /// Returns `HFResult<impl Stream<Item = HFResult<RepoTreeEntry>>>`. Set
    /// `recursive` to traverse subdirectories, and `expand` to include
    /// per-file metadata such as size, LFS info, and last-commit summaries.
    ///
    /// Use [`HFRepository::list_files`] when you only need paths, or
    /// [`HFRepository::get_paths_info`] when you already know the exact paths
    /// you want to inspect.
    pub fn list_tree(&self, params: RepoListTreeParams) -> HFResult<impl Stream<Item = HFResult<RepoTreeEntry>> + '_> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str = format!("{}/tree/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);
        let url = Url::parse(&url_str)?;

        let mut query: Vec<(String, String)> = Vec::new();
        if params.recursive {
            query.push(("recursive".into(), "true".into()));
        }
        if params.expand {
            query.push(("expand".into(), "true".into()));
        }

        Ok(self.hf_client.paginate(url, query, params.limit))
    }

    /// Get info about specific paths in a repository.
    ///
    /// Prefer this over [`HFRepository::list_tree`] when you already know the
    /// small set of paths you want to inspect.
    ///
    /// Endpoint: POST /api/{repo_type}s/{repo_id}/paths-info/{revision}
    pub async fn get_paths_info(&self, params: RepoGetPathsInfoParams) -> HFResult<Vec<RepoTreeEntry>> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url =
            format!("{}/paths-info/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);

        let body = serde_json::json!({
            "paths": params.paths,
        });

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
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.paths.join(", "),
                },
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch metadata for a single file via a HEAD request on its resolve URL.
    ///
    /// Returns the resolved commit hash, ETag, file size, and (if the file is Xet-backed)
    /// the Xet content hash — without downloading the file contents.
    ///
    /// Endpoint: HEAD {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filepath}
    pub async fn get_file_metadata(&self, params: RepoGetFileMetadataParams) -> HFResult<FileMetadataInfo> {
        let filename = params.filepath.clone();
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
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

sync_api! {
    impl HFRepository -> HFRepositorySync {
        fn list_files(&self, params: RepoListFilesParams) -> HFResult<Vec<String>>;
        fn get_paths_info(&self, params: RepoGetPathsInfoParams) -> HFResult<Vec<RepoTreeEntry>>;
        fn get_file_metadata(&self, params: RepoGetFileMetadataParams) -> HFResult<FileMetadataInfo>;
    }
}

sync_api_stream! {
    impl HFRepository -> HFRepositorySync {
        fn list_tree(&self, params: RepoListTreeParams) -> RepoTreeEntry;
    }
}
