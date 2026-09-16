//! Repository content listing and metadata lookups.
//!
//! Builders on [`HFRepository`] for inspecting what's in a repo without downloading file
//! contents:
//!
//! - [`HFRepository::list_tree`] — paginated stream of [`RepoTreeEntry`] (files and directories), optionally recursive
//!   and prefix-filtered.
//! - [`HFRepository::get_paths_info`] — batched lookup of metadata for a known set of paths.
//! - [`HFRepository::get_file_metadata`] — HEAD-based metadata for one file (size, ETag, commit hash, xet hash).

use bon::bon;
use futures::stream::Stream;
use reqwest::Url;

#[cfg(not(target_family = "wasm"))]
use super::files::{extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash};
use super::{FileMetadataInfo, HFRepository, RepoTreeEntry, RepoType};
use crate::client::encode_ref;
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Stream file and directory entries in the repository tree.
    ///
    /// Returns `HFResult<impl Stream<Item = HFResult<RepoTreeEntry>>>`.
    ///
    /// Use [`HFRepository::get_paths_info`] when you already know the exact paths
    /// you want to inspect.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision to list. Defaults to the main branch.
    /// - `path_in_repo`: repository-relative subdirectory to list. Defaults to the repo root. Leading, trailing, and
    ///   consecutive `/` separators are ignored; path segments are URL-encoded automatically.
    /// - `recursive` (default `false`): traverse subdirectories.
    /// - `expand` (default `false`): include per-file metadata such as size, LFS info, and last-commit summaries.
    /// - `limit`: cap the total number of entries yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        /// Git revision to list. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Repository-relative subdirectory to list. Defaults to the repo root.
        /// Leading, trailing, and consecutive `/` separators are ignored;
        /// path segments are URL-encoded automatically.
        #[builder(into)]
        path_in_repo: Option<String>,
        /// Traverse subdirectories.
        #[builder(default)]
        recursive: bool,
        /// Include per-file metadata such as size, LFS info, and last-commit summaries.
        #[builder(default)]
        expand: bool,
        /// Cap the total number of entries yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<RepoTreeEntry>> + '_> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str = format!(
            "{}/tree/{}",
            self.hf_client.api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );
        let mut url = Url::parse(&url_str)?;
        if let Some(path) = path_in_repo.as_deref() {
            crate::client::append_path_segments(&mut url, path)?;
        }

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
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_paths_info(
        &self,
        /// Paths in the repository to fetch info for.
        paths: Vec<String>,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = format!(
            "{}/paths-info/{}",
            self.hf_client.api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );

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
    /// For LFS- and Xet-backed files the Hub sets `X-Repo-Commit`, `X-Linked-Etag`,
    /// `X-Linked-Size` and `X-Xet-Hash` on the 302 to the CDN, not on the CDN response,
    /// so native targets stop at the first absolute redirect and read them off it.
    /// The browser owns redirect handling on wasm and only exposes the final response's
    /// headers, so there this dispatches to `paths-info` plus a revision lookup instead:
    /// two requests rather than one, and `location` is the resolve URL rather than the CDN
    /// URL the redirect would have named.
    ///
    /// # Parameters
    ///
    /// - `filepath` (required): path of the file to inspect within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_file_metadata(
        &self,
        /// Path of the file to inspect within the repository.
        #[builder(into)]
        filepath: String,
        /// Git revision. Defaults to the main branch.
        revision: Option<&str>,
    ) -> HFResult<FileMetadataInfo> {
        let revision = revision.unwrap_or(constants::DEFAULT_REVISION);

        #[cfg(target_family = "wasm")]
        return self.file_metadata_via_paths_info(filepath, revision).await;

        #[cfg(not(target_family = "wasm"))]
        self.file_metadata_via_head(filepath, revision).await
    }

    /// HEAD the resolve URL and read the metadata off the response headers, stopping at the
    /// first absolute redirect so the Hub's headers survive.
    #[cfg(not(target_family = "wasm"))]
    async fn file_metadata_via_head(&self, filename: String, revision: &str) -> HFResult<FileMetadataInfo> {
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(self.repo_type.url_prefix(), &repo_path, revision, &filename)?;

        let headers = self.hf_client.auth_headers();
        let response = self.hf_client.head_with_relative_redirects(&url, &headers).await?;
        let response = if response.status().is_redirection() {
            response
        } else {
            self.hf_client
                .check_response(
                    response,
                    Some(&repo_path),
                    crate::error::NotFoundContext::Entry { path: filename.clone() },
                )
                .await?
        };

        let etag = extract_etag(&response).ok_or_else(|| {
            HFError::malformed_response_at(format!("missing ETag header for {filename}"), url.to_string())
        })?;
        let commit_hash = extract_commit_hash(&response).ok_or_else(|| {
            HFError::malformed_response_at(format!("missing X-Repo-Commit header for {filename}"), url.to_string())
        })?;
        let xet_hash = extract_xet_hash(&response);
        let file_size = extract_file_size(&response).unwrap_or_else(|| {
            tracing::warn!(
                file = %filename,
                "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0"
            );
            0
        });
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| response.url().join(value).ok())
            .map(|resolved| resolved.to_string())
            .or_else(|| Some(response.url().to_string()));

        Ok(FileMetadataInfo {
            filename,
            etag,
            commit_hash,
            xet_hash,
            file_size,
            location,
        })
    }

    /// Assemble the same metadata from non-redirecting JSON endpoints, for targets that
    /// cannot see a redirect's headers. `paths-info` carries everything but the revision's
    /// commit, which the repo info endpoint supplies.
    ///
    /// Only wasm dispatches here, but it stays compiled everywhere so the native test suite
    /// can cover it — CI never executes wasm tests.
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    async fn file_metadata_via_paths_info(&self, filename: String, revision: &str) -> HFResult<FileMetadataInfo> {
        #[derive(serde::Deserialize)]
        struct RevisionSha {
            sha: String,
        }

        let entries = self
            .get_paths_info()
            .paths(vec![filename.clone()])
            .revision(revision.to_string())
            .send()
            .await?;
        let entry = entries
            .into_iter()
            .find(|entry| matches!(entry, RepoTreeEntry::File { path, .. } if *path == filename));
        let (oid, size, lfs, xet_hash) = match entry {
            Some(RepoTreeEntry::File {
                oid,
                size,
                lfs,
                xet_hash,
                ..
            }) => (oid, size, lfs, xet_hash),
            _ => {
                return Err(HFError::EntryNotFound {
                    path: filename,
                    repo_id: self.repo_path(),
                    context: None,
                });
            },
        };

        let commit_hash = self.fetch_repo_info::<RevisionSha>(Some(revision.to_string()), None).await?.sha;

        Ok(FileMetadataInfo {
            // The Hub serves `X-Linked-Etag` (the LFS object's sha256) for LFS-backed files
            // and the git blob oid otherwise; `paths-info` reports both separately.
            etag: lfs.and_then(|lfs| lfs.sha256).unwrap_or(oid),
            commit_hash,
            xet_hash,
            file_size: size,
            location: Some(self.hf_client.download_url(
                self.repo_type.url_prefix(),
                &self.repo_path(),
                revision,
                &filename,
            )?),
            filename,
        })
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
use futures::stream::StreamExt as _;

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::list_tree`]. Returns the collected stream as a
    /// `Vec<RepoTreeEntry>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] path_in_repo: Option<String>,
        #[builder(default)] recursive: bool,
        #[builder(default)] expand: bool,
        limit: Option<usize>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_tree()
                .maybe_revision(revision)
                .maybe_path_in_repo(path_in_repo)
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
    #[builder(finish_fn = send, derive(Debug, Clone))]
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
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_file_metadata(
        &self,
        #[builder(into)] filepath: String,
        revision: Option<&str>,
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

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use crate::test_support::mock_hub;

    const XET_REDIRECT: &str = "HTTP/1.1 302 Found\r\nLocation: {endpoint}/cdn\r\nX-Repo-Commit: deadbeef\r\nX-Linked-Etag: \"realsha\"\r\nX-Linked-Size: 42\r\nX-Xet-Hash: abc123\r\nETag: \"hubetag\"\r\nContent-Length: 1102\r\nConnection: close\r\n\r\n";
    const CDN_OK: &str = "HTTP/1.1 200 OK\r\nETag: \"cdnetag\"\r\nContent-Length: 42\r\nConnection: close\r\n\r\n";

    #[tokio::test]
    async fn file_metadata_reads_headers_from_the_cdn_redirect() {
        let (client, server) = mock_hub(&[
            ("HEAD /owner/repo/resolve/main/model.bin HTTP/1.1", XET_REDIRECT),
            ("HEAD /cdn HTTP/1.1", CDN_OK),
        ])
        .await;
        let metadata = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("model.bin")
            .send()
            .await;
        server.abort();

        let metadata = metadata.unwrap();
        assert_eq!(metadata.xet_hash.as_deref(), Some("abc123"));
        assert_eq!(metadata.commit_hash, "deadbeef");
        assert_eq!(metadata.etag, "realsha");
        assert_eq!(metadata.file_size, 42);
        assert!(metadata.location.unwrap().ends_with("/cdn"));
    }

    #[tokio::test]
    async fn file_metadata_reads_headers_from_a_direct_response() {
        let (client, server) = mock_hub(&[(
            "HEAD /owner/repo/resolve/main/config.json HTTP/1.1",
            "HTTP/1.1 200 OK\r\nX-Repo-Commit: deadbeef\r\nETag: \"abc\"\r\nContent-Length: 570\r\nConnection: close\r\n\r\n",
        )])
        .await;
        let metadata = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("config.json")
            .send()
            .await;
        server.abort();

        let metadata = metadata.unwrap();
        assert!(metadata.xet_hash.is_none());
        assert_eq!(metadata.commit_hash, "deadbeef");
        assert_eq!(metadata.etag, "abc");
        assert_eq!(metadata.file_size, 570);
        assert!(metadata.location.unwrap().ends_with("/owner/repo/resolve/main/config.json"));
    }

    #[tokio::test]
    async fn file_metadata_surfaces_missing_file_errors() {
        let (client, server) = mock_hub(&[]).await;
        let result = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("missing.bin")
            .send()
            .await;
        server.abort();
        assert!(
            matches!(result, Err(crate::HFError::EntryNotFound { repo_id, path, .. }) if repo_id == "owner/repo" && path == "missing.bin")
        );
    }

    // The wasm dispatch is exercised here because CI never runs wasm tests; the helper
    // itself is target-independent.
    #[tokio::test]
    async fn paths_info_fallback_matches_the_header_path_for_an_lfs_file() {
        let (client, server) = mock_hub(&[
            (
                "POST /api/models/owner/repo/paths-info/main HTTP/1.1",
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 133\r\nConnection: close\r\n\r\n[{\"type\":\"file\",\"oid\":\"blobsha\",\"size\":42,\"lfs\":{\"oid\":\"realsha\",\"size\":42,\"pointerSize\":134},\"xetHash\":\"abc123\",\"path\":\"model.bin\"}]",
            ),
            (
                "GET /api/models/owner/repo/revision/main HTTP/1.1",
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"sha\":\"deadbeef\"}",
            ),
        ])
        .await;
        let metadata = client
            .model("owner", "repo")
            .file_metadata_via_paths_info("model.bin".to_string(), "main")
            .await;
        server.abort();

        let metadata = metadata.unwrap();
        // Same values the redirect-header path produces for the equivalent file.
        assert_eq!(metadata.xet_hash.as_deref(), Some("abc123"));
        assert_eq!(metadata.commit_hash, "deadbeef");
        assert_eq!(metadata.etag, "realsha");
        assert_eq!(metadata.file_size, 42);
    }

    #[tokio::test]
    async fn paths_info_fallback_uses_the_blob_oid_for_a_plain_file() {
        let (client, server) = mock_hub(&[
            (
                "POST /api/models/owner/repo/paths-info/main HTTP/1.1",
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 65\r\nConnection: close\r\n\r\n[{\"type\":\"file\",\"oid\":\"blobsha\",\"size\":570,\"path\":\"config.json\"}]",
            ),
            (
                "GET /api/models/owner/repo/revision/main HTTP/1.1",
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"sha\":\"deadbeef\"}",
            ),
        ])
        .await;
        let metadata = client
            .model("owner", "repo")
            .file_metadata_via_paths_info("config.json".to_string(), "main")
            .await;
        server.abort();

        let metadata = metadata.unwrap();
        assert_eq!(metadata.etag, "blobsha");
        assert!(metadata.xet_hash.is_none());
        assert_eq!(metadata.file_size, 570);
    }

    #[tokio::test]
    async fn paths_info_fallback_surfaces_missing_file_errors() {
        let (client, server) = mock_hub(&[(
            "POST /api/models/owner/repo/paths-info/main HTTP/1.1",
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n[]",
        )])
        .await;
        let result = client
            .model("owner", "repo")
            .file_metadata_via_paths_info("missing.bin".to_string(), "main")
            .await;
        server.abort();
        assert!(
            matches!(result, Err(crate::HFError::EntryNotFound { repo_id, path, .. }) if repo_id == "owner/repo" && path == "missing.bin")
        );
    }
}
