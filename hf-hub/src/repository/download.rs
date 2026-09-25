//! Repository file and snapshot download builders.
//!
//! Builders on [`HFRepository`] for fetching file contents:
//!
//! - [`HFRepository::download_file`] — download one file to the cache or a local directory.
//! - [`HFRepository::download_file_stream`] — stream a file (or byte range) without buffering.
//! - [`HFRepository::download_file_to_bytes`] — read a file (or byte range) into memory.
//! - [`HFRepository::snapshot_download`] — download all files at a revision, optionally filtered by allow/ignore globs
//!   matched against repo-relative paths.
//!
//! Range parameters use Rust `std::ops::Range<u64>` semantics (start-inclusive, end-exclusive).
//! See each builder's docs for the exact path / range / glob format rules.

#[cfg(not(target_family = "wasm"))]
use std::{
    collections::HashSet,
    io::Write,
    path::{Path, PathBuf},
};

use bon::bon;
#[cfg(not(target_family = "wasm"))]
use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
#[cfg(not(target_family = "wasm"))]
use reqwest::header::IF_NONE_MATCH;
#[cfg(not(target_family = "wasm"))]
use serde::Deserialize;

use super::files::extract_file_size;
#[cfg(not(target_family = "wasm"))]
use super::{
    FileMetadataInfo,
    files::{extract_commit_hash, extract_etag, extract_xet_hash, matches_any_glob},
};
use super::{HFRepository, RepoTreeEntry, RepoType};
#[cfg(not(target_family = "wasm"))]
use crate::cache::storage as cache;
use crate::error::{HFError, HFResult};
use crate::progress::{DownloadEvent, EmitEvent, FileProgress, FileStatus, Progress};
use crate::{constants, retry};

/// Boxed byte stream returned by [`HFRepository::download_file_stream`].
///
/// `Send + Unpin` on native targets; on wasm the browser `reqwest` backend
/// produces `!Send` streams, so the `Send` bound is dropped.
#[cfg(not(target_family = "wasm"))]
pub type HFByteStream = Box<dyn Stream<Item = HFResult<bytes::Bytes>> + Send + Unpin>;
#[cfg(target_family = "wasm")]
pub type HFByteStream = Box<dyn Stream<Item = HFResult<bytes::Bytes>> + Unpin>;

/// Internal options struct used by the file download helpers.
#[cfg(not(target_family = "wasm"))]
struct DownloadFileParams {
    filename: String,
    local_dir: Option<PathBuf>,
    revision: Option<String>,
    force_download: bool,
    local_files_only: bool,
    progress: Option<Progress>,
    /// Emit `DownloadEvent::Start` for this file. Off inside `snapshot_download`, whose own
    /// `Start` already covers every file.
    announce_start: bool,
}

/// Internal options struct used by the streaming download helpers.
struct DownloadFileStreamParams {
    filename: String,
    revision: Option<String>,
    range: Option<std::ops::Range<u64>>,
    progress: Option<Progress>,
}

/// Internal options struct used by `snapshot_download_impl`.
#[cfg(not(target_family = "wasm"))]
struct SnapshotDownloadParams {
    revision: Option<String>,
    allow_patterns: Option<Vec<String>>,
    ignore_patterns: Option<Vec<String>>,
    local_dir: Option<PathBuf>,
    force_download: bool,
    local_files_only: bool,
    max_workers: Option<usize>,
    progress: Option<Progress>,
}

impl<T: RepoType> HFRepository<T> {
    #[cfg(not(target_family = "wasm"))]
    async fn download_file_impl(&self, params: DownloadFileParams) -> HFResult<PathBuf> {
        let result = self.download_file_inner(&params).await;
        if result.is_ok() {
            params.progress.emit(DownloadEvent::Complete);
        }
        result
    }

    #[cfg(not(target_family = "wasm"))]
    async fn download_file_inner(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        if params.local_dir.is_some() {
            self.download_file_to_local_dir(params).await
        } else {
            if !self.hf_client.cache_enabled() {
                return Err(HFError::CacheNotEnabled);
            }
            self.download_file_to_cache(params).await
        }
    }

    // Determine whether the file is xet-backed and learn its size.
    //
    // Native: HEAD the resolve URL and read `X-Xet-Hash` /
    // `Content-Length` / `X-Linked-Size` from the response headers.
    //
    // Wasm: the resolve URL 302-redirects to a CAS blob URL, and the Fetch
    // spec only surfaces the final response's headers when following
    // redirects (`redirect: 'manual'` doesn't help — it yields an opaque
    // response with no readable headers). So on wasm we dispatch via
    // `paths-info`, a non-redirecting JSON endpoint that returns the same
    // metadata in the body.
    #[cfg(not(target_family = "wasm"))]
    async fn resolve_xet_hash_and_size(
        &self,
        revision: &str,
        filename: &str,
    ) -> HFResult<(Option<String>, Option<u64>)> {
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(self.repo_type.url_prefix(), &repo_path, revision, filename)?;
        let headers = self.hf_client.auth_headers();
        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().head(&url).headers(headers.clone()).send()
        })
        .await?;
        let head_response = self
            .hf_client
            .check_response(
                head_response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: filename.to_string(),
                },
            )
            .await?;
        Ok((extract_xet_hash(&head_response), extract_file_size(&head_response)))
    }

    #[cfg(target_family = "wasm")]
    async fn resolve_xet_hash_and_size(
        &self,
        revision: &str,
        filename: &str,
    ) -> HFResult<(Option<String>, Option<u64>)> {
        let entries = self
            .get_paths_info()
            .paths(vec![filename.to_string()])
            .revision(revision.to_string())
            .send()
            .await?;
        let entry = entries
            .into_iter()
            .find(|e| matches!(e, RepoTreeEntry::File { path, .. } if path == filename));
        match entry {
            Some(RepoTreeEntry::File { xet_hash, size, .. }) => Ok((xet_hash, Some(size))),
            _ => Err(HFError::EntryNotFound {
                path: filename.to_string(),
                repo_id: self.repo_path(),
                context: None,
            }),
        }
    }

    async fn download_file_stream_impl(
        &self,
        params: DownloadFileStreamParams,
    ) -> HFResult<(Option<u64>, HFByteStream)> {
        if let Some(ref range) = params.range
            && range.start >= range.end
        {
            return Err(HFError::InvalidParameter(format!(
                "range start ({}) must be less than end ({})",
                range.start, range.end
            )));
        }

        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(self.repo_type.url_prefix(), &repo_path, revision, &params.filename)?;

        let headers = self.hf_client.auth_headers();

        let (xet_hash, file_size_hint) = self.resolve_xet_hash_and_size(revision, &params.filename).await?;

        if let Some(xet_hash) = xet_hash {
            let file_size: u64 = file_size_hint.unwrap_or_else(|| {
                tracing::warn!(url = %url, "missing file size for xet file, defaulting to 0");
                0
            });

            let content_length = params.range.as_ref().map(|r| r.end.saturating_sub(r.start)).or(Some(file_size));

            let stream = self
                .xet_download_stream(revision, &xet_hash, file_size, params.range.clone())
                .await?;

            let total_bytes = content_length.unwrap_or(0);
            params.progress.emit(DownloadEvent::Start {
                total_files: 1,
                total_bytes,
            });
            let wrapped =
                wrap_stream_with_progress(Box::new(Box::pin(stream)), params.progress, params.filename, total_bytes);
            #[cfg(target_family = "wasm")]
            let wrapped = buffer_wasm_stream(wrapped);
            return Ok((content_length, wrapped));
        }

        let range_header = params
            .range
            .as_ref()
            .map(|r| format!("bytes={}-{}", r.start, r.end.saturating_sub(1)));
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self.hf_client.http_client().get(&url).headers(headers.clone());
            if let Some(ref range) = range_header {
                req = req.header(reqwest::header::RANGE, range);
            }
            req.send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        let content_length = extract_file_size(&response);
        let total_bytes = content_length.unwrap_or(0);
        let stream = response.bytes_stream().map(|r| r.map_err(HFError::from));
        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes,
        });
        let wrapped =
            wrap_stream_with_progress(Box::new(Box::pin(stream)), params.progress, params.filename, total_bytes);
        #[cfg(target_family = "wasm")]
        let wrapped = buffer_wasm_stream(wrapped);
        Ok((content_length, wrapped))
    }

    async fn download_file_to_bytes_impl(&self, params: DownloadFileStreamParams) -> HFResult<bytes::Bytes> {
        let (content_length, stream) = self.download_file_stream_impl(params).await?;
        futures::pin_mut!(stream);

        let capacity = content_length.unwrap_or(0) as usize;
        let mut buf = bytes::BytesMut::with_capacity(capacity);
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk?);
        }
        Ok(buf.freeze())
    }

    #[cfg(not(target_family = "wasm"))]
    async fn download_file_to_local_dir(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(self.repo_type.url_prefix(), &repo_path, revision, &params.filename)?;

        let headers = self.hf_client.auth_headers();
        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().head(&url).headers(headers.clone()).send()
        })
        .await?;

        let head_response = self
            .hf_client
            .check_response(
                head_response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        let file_size = extract_file_size(&head_response).unwrap_or(0);
        let has_xet_hash = head_response.headers().get(constants::HEADER_X_XET_HASH).is_some();

        if params.announce_start {
            params.progress.emit(DownloadEvent::Start {
                total_files: 1,
                total_bytes: file_size,
            });
        }

        if has_xet_hash {
            let local_dir = params.local_dir.as_ref().unwrap();
            return self
                .xet_download_to_local_dir(revision, &params.filename, local_dir, &head_response, &params.progress)
                .await;
        }

        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        let local_dir = params.local_dir.as_ref().unwrap();
        std::fs::create_dir_all(local_dir)?;

        let dest_path = local_dir.join(&params.filename);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        stream_response_to_file_with_progress(
            response,
            &dest_path,
            &params.progress,
            Some(&params.filename),
            file_size,
        )
        .await?;
        params.progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: params.filename.clone(),
                bytes_completed: file_size,
                total_bytes: file_size,
                status: FileStatus::Complete,
            }],
        });

        Ok(dest_path)
    }

    /// Resolve a file from the local cache without making network requests.
    /// Matches Python's `try_to_load_from_cache`: checks the snapshot pointer
    /// first, then consults `.no_exist` markers for negative cache hits.
    #[cfg(not(target_family = "wasm"))]
    fn resolve_from_cache_only(&self, repo_folder: &str, revision: &str, filename: &str) -> HFResult<PathBuf> {
        let cache_dir = self.hf_client.cache_dir();

        let commit_hash = if cache::is_commit_hash(revision) {
            Some(revision.to_string())
        } else {
            let ref_path = cache::ref_path(cache_dir, repo_folder, revision);
            std::fs::read_to_string(&ref_path).ok().map(|s| s.trim().to_string())
        };

        if let Some(ref hash) = commit_hash {
            let snap = cache::snapshot_path(cache_dir, repo_folder, hash, filename);
            if snap.exists() {
                return Ok(snap);
            }
            if cache::no_exist_path(cache_dir, repo_folder, hash, filename).exists() {
                return Err(HFError::EntryNotFound {
                    path: filename.to_string(),
                    repo_id: String::new(),
                    context: None,
                });
            }
        }

        Err(HFError::LocalEntryNotFound {
            path: filename.to_string(),
        })
    }

    /// Resolve the cached etag for a file by reading the symlink target in snapshots/.
    /// On Windows, where copies are used instead of symlinks, `read_link` will fail
    /// and this returns `None`, disabling conditional-request (If-None-Match) optimization.
    #[cfg(not(target_family = "wasm"))]
    fn find_cached_etag(&self, repo_folder: &str, revision: &str, filename: &str) -> Option<String> {
        let cache_dir = self.hf_client.cache_dir();

        let commit_hash = if cache::is_commit_hash(revision) {
            Some(revision.to_string())
        } else {
            let ref_path = cache::ref_path(cache_dir, repo_folder, revision);
            std::fs::read_to_string(&ref_path).ok().map(|s| s.trim().to_string())
        };

        let hash = commit_hash?;
        let snap = cache::snapshot_path(cache_dir, repo_folder, &hash, filename);
        let target = std::fs::read_link(&snap).ok()?;
        target.file_name()?.to_str().map(|s| s.to_string())
    }

    #[cfg(not(target_family = "wasm"))]
    async fn download_file_to_cache(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let cache_dir = self.hf_client.cache_dir();
        let repo_folder = cache::repo_folder_name(&self.repo_path(), self.repo_type.plural());
        let force_download = params.force_download;

        if cache::is_commit_hash(revision) && !force_download {
            let snap = cache::snapshot_path(cache_dir, &repo_folder, revision, &params.filename);
            if snap.exists() {
                announce_cached_file(params, &snap, false)?;
                return Ok(snap);
            }
        }

        if params.local_files_only {
            let path = self.resolve_from_cache_only(&repo_folder, revision, &params.filename)?;
            announce_cached_file(params, &path, false)?;
            return Ok(path);
        }

        let mut start_size: Option<u64> = None;
        let result = self
            .download_file_to_cache_network(params, revision, cache_dir, &repo_folder, force_download, &mut start_size)
            .await;

        match result {
            Err(e) if e.is_transient() && !force_download => {
                match self.resolve_from_cache_only(&repo_folder, revision, &params.filename) {
                    Ok(path) => {
                        announce_cached_file_after_transient_error(params, &path, start_size);
                        Ok(path)
                    },
                    Err(_) => Err(e),
                }
            },
            result => result,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    async fn download_file_to_cache_network(
        &self,
        params: &DownloadFileParams,
        revision: &str,
        cache_dir: &Path,
        repo_folder: &str,
        force_download: bool,
        start_size: &mut Option<u64>,
    ) -> HFResult<PathBuf> {
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(self.repo_type.url_prefix(), &repo_path, revision, &params.filename)?;

        let cached_etag = if !force_download {
            self.find_cached_etag(repo_folder, revision, &params.filename)
        } else {
            None
        };

        let mut head_headers = self.hf_client.auth_headers();
        if let Some(ref etag_val) = cached_etag
            && let Ok(hv) = reqwest::header::HeaderValue::from_str(&format!("\"{etag_val}\""))
        {
            head_headers.insert(IF_NONE_MATCH, hv);
        }

        let head_response = self.hf_client.head_with_relative_redirects(&url, &head_headers).await?;

        let status = head_response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(mark_no_exist_and_return_error(
                cache_dir,
                repo_folder,
                revision,
                &head_response,
                &repo_path,
                &params.filename,
            )
            .await);
        }

        if status == reqwest::StatusCode::NOT_MODIFIED {
            let etag = cached_etag
                .ok_or_else(|| HFError::malformed_response_at("304 Not Modified without cached ETag", url.clone()))?;
            let commit_hash = if cache::is_commit_hash(revision) {
                revision.to_string()
            } else {
                cache::read_ref(cache_dir, repo_folder, revision).await?.ok_or_else(|| {
                    HFError::malformed_response_at("304 Not Modified without cached commit hash", url.clone())
                })?
            };
            let path =
                finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await?;
            announce_cached_file(params, &path, false)?;
            return Ok(path);
        }

        let etag = extract_etag(&head_response)
            .ok_or_else(|| HFError::malformed_response_at("missing ETag header", url.clone()));
        let commit_hash = extract_commit_hash(&head_response);
        let xet_hash = extract_xet_hash(&head_response);
        let has_xet_hash = xet_hash.is_some();
        let file_size: u64 = extract_file_size(&head_response).unwrap_or_else(|| {
            tracing::warn!(url = %url, "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0");
            0
        });

        if !status.is_success() && !status.is_redirection() {
            self.hf_client
                .check_response(
                    head_response,
                    Some(&repo_path),
                    crate::error::NotFoundContext::Entry {
                        path: params.filename.clone(),
                    },
                )
                .await?;
        }

        let etag = etag?;
        let commit_hash =
            commit_hash.ok_or_else(|| HFError::malformed_response_at("missing X-Repo-Commit header", url.clone()))?;

        if params.announce_start {
            params.progress.emit(DownloadEvent::Start {
                total_files: 1,
                total_bytes: file_size,
            });
            *start_size = Some(file_size);
        }

        if has_xet_hash {
            let xet_hash =
                xet_hash.ok_or_else(|| HFError::malformed_response_at("missing X-Xet-Hash header", url.clone()))?;
            let blob = cache::blob_path(cache_dir, repo_folder, &etag);
            let _lock = if force_download || !blob.exists() {
                Some(cache::acquire_lock(cache_dir, repo_folder, &etag).await?)
            } else {
                None
            };
            if force_download || !blob.exists() {
                if let Some(parent) = blob.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                self.xet_download_to_blob(revision, &params.filename, &xet_hash, file_size, &blob, &params.progress)
                    .await?;
            } else {
                emit_file_complete(&params.progress, &params.filename, file_size);
            }

            return finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await;
        }

        let blob = cache::blob_path(cache_dir, repo_folder, &etag);
        let _lock = if force_download || !blob.exists() {
            Some(cache::acquire_lock(cache_dir, repo_folder, &etag).await?)
        } else {
            None
        };

        if blob.exists() && !force_download {
            emit_file_complete(&params.progress, &params.filename, file_size);
            return finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await;
        }

        let incomplete_path = PathBuf::from(format!("{}.incomplete", blob.display()));
        if let Some(parent) = incomplete_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let dl_headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(dl_headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;
        stream_response_to_file_with_progress(
            response,
            &incomplete_path,
            &params.progress,
            Some(&params.filename),
            file_size,
        )
        .await?;
        params.progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: params.filename.clone(),
                bytes_completed: file_size,
                total_bytes: file_size,
                status: FileStatus::Complete,
            }],
        });
        std::fs::rename(&incomplete_path, &blob)?;

        finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await
    }

    #[cfg(not(target_family = "wasm"))]
    async fn resolve_commit_hash(&self, revision: &str) -> HFResult<String> {
        if cache::is_commit_hash(revision) {
            return Ok(revision.to_string());
        }
        #[derive(Deserialize)]
        struct ShaOnly {
            sha: Option<String>,
        }
        let repo_path = self.repo_path();
        let info: ShaOnly = self.fetch_repo_info(Some(revision.to_string()), None).await?;
        info.sha.ok_or_else(|| {
            HFError::malformed_response(format!("repo info for {}@{} returned no commit sha", repo_path, revision))
        })
    }

    #[cfg(not(target_family = "wasm"))]
    async fn list_filtered_files(
        &self,
        revision: &str,
        allow_patterns: Option<&Vec<String>>,
        ignore_patterns: Option<&Vec<String>>,
    ) -> HFResult<Vec<(String, u64)>> {
        let stream = self.list_tree().revision(revision.to_string()).recursive(true).send()?;
        futures::pin_mut!(stream);

        let mut files: Vec<(String, u64)> = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, size, .. } = entry {
                files.push((path, size));
            }
        }

        if let Some(allow) = allow_patterns {
            files.retain(|(f, _)| matches_any_glob(allow, f));
        }
        if let Some(ignore) = ignore_patterns {
            files.retain(|(f, _)| !matches_any_glob(ignore, f));
        }

        Ok(files)
    }

    #[cfg(not(target_family = "wasm"))]
    async fn snapshot_download_impl(&self, params: SnapshotDownloadParams) -> HFResult<PathBuf> {
        if params.local_dir.is_none() && !self.hf_client.cache_enabled() {
            return Err(HFError::CacheNotEnabled);
        }
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let max_workers = params.max_workers.unwrap_or(8);
        let repo_folder = cache::repo_folder_name(&self.repo_path(), self.repo_type.plural());
        let cache_dir = self.hf_client.cache_dir();

        if params.local_files_only {
            let commit_hash = if cache::is_commit_hash(revision) {
                revision.to_string()
            } else {
                cache::read_ref(cache_dir, &repo_folder, revision).await?.ok_or_else(|| {
                    HFError::LocalEntryNotFound {
                        path: format!("{}/{}", repo_folder, revision),
                    }
                })?
            };
            let snapshot_dir = cache_dir.join(&repo_folder).join("snapshots").join(&commit_hash);
            if snapshot_dir.exists() {
                return Ok(snapshot_dir);
            }
            return Err(HFError::LocalEntryNotFound {
                path: format!("{}/{}", repo_folder, commit_hash),
            });
        }

        let commit_hash = self.resolve_commit_hash(revision).await?;

        let listed_files = self
            .list_filtered_files(&commit_hash, params.allow_patterns.as_ref(), params.ignore_patterns.as_ref())
            .await?;

        let force = params.force_download;

        let mut cached_files: Vec<(String, u64)> = Vec::new();
        let mut filenames: Vec<String> = Vec::with_capacity(listed_files.len());
        for (filename, size) in listed_files {
            if !force
                && params.local_dir.is_none()
                && cache::snapshot_path(cache_dir, &repo_folder, &commit_hash, &filename).exists()
            {
                cached_files.push((filename, size));
            } else {
                filenames.push(filename);
            }
        }

        let repo_path = self.repo_path();
        let repo_path_ref = &repo_path;
        let commit_hash_ref = &commit_hash;
        let mut head_futs = Vec::with_capacity(filenames.len());
        for filename in &filenames {
            let auth = self.hf_client.auth_headers();
            let filename = filename.clone();
            let repo_folder_ref = &repo_folder;
            head_futs.push(async move {
                    let url = self
                        .hf_client
                        .download_url(self.repo_type.url_prefix(), repo_path_ref, commit_hash_ref, &filename)?;
                    let resp = self.hf_client.head_with_relative_redirects(&url, &auth).await?;
                    // Per-file 404 resilience: write a .no_exist marker and skip
                    // the file rather than aborting the entire snapshot download.
                    // This matches the Python huggingface_hub library behavior.
                    // Alternative: since the file list comes from list_repo_tree
                    // on a pinned commit, a 404 here is unexpected and could be
                    // treated as an error instead.
                    if resp.status() == reqwest::StatusCode::NOT_FOUND {
                        if let Some(commit) = extract_commit_hash(&resp) {
                            let no_exist = cache::no_exist_path(cache_dir, repo_folder_ref, &commit, &filename);
                            if let Some(parent) = no_exist.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            let _ = std::fs::write(&no_exist, b"");
                        }
                        return Ok::<_, HFError>(None);
                    } else if !resp.status().is_success() && !resp.status().is_redirection() {
                        let context = Box::new(crate::error::HttpErrorContext::from_response(resp).await);
                        return Err(HFError::Http { context });
                    }
                    let etag = extract_etag(&resp).ok_or_else(|| {
                        HFError::malformed_response_at(format!("missing ETag header for {filename}"), url.clone())
                    })?;
                    let commit = extract_commit_hash(&resp).unwrap_or_else(|| commit_hash_ref.clone());
                    let xet_hash = extract_xet_hash(&resp);
                    let file_size: u64 = extract_file_size(&resp).unwrap_or_else(|| {
                        tracing::warn!(file = %filename, "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0");
                        0
                    });
                    let location = Some(resp.url().to_string());
                    Ok::<_, HFError>(Some(FileMetadataInfo {
                        filename,
                        etag,
                        commit_hash: commit,
                        xet_hash,
                        file_size,
                        location,
                    }))
            });
        }

        let file_metas: Vec<FileMetadataInfo> = futures::stream::iter(head_futs)
            .buffer_unordered(max_workers)
            .try_collect::<Vec<Option<FileMetadataInfo>>>()
            .await?
            .into_iter()
            .flatten()
            .collect();

        let total_bytes: u64 = cached_files.iter().map(|(_, size)| size).sum::<u64>()
            + file_metas.iter().map(|m| m.file_size).sum::<u64>();
        params.progress.emit(DownloadEvent::Start {
            total_files: cached_files.len() + file_metas.len(),
            total_bytes,
        });
        if !cached_files.is_empty() {
            params.progress.emit(DownloadEvent::Progress {
                files: cached_files
                    .iter()
                    .map(|(filename, size)| FileProgress {
                        filename: filename.clone(),
                        bytes_completed: *size,
                        total_bytes: *size,
                        status: FileStatus::Complete,
                    })
                    .collect(),
            });
        }

        let mut xet_metas = Vec::new();
        let mut non_xet_filenames = Vec::new();

        if let Some(ref local_dir) = params.local_dir {
            let mut local_cached = Vec::new();
            for meta in file_metas {
                let dest = local_dir.join(&meta.filename);
                if dest.exists() && !force {
                    local_cached.push(meta);
                    continue;
                }
                if meta.xet_hash.is_some() {
                    xet_metas.push(meta);
                } else {
                    non_xet_filenames.push(meta.filename);
                }
            }
            if !local_cached.is_empty() {
                params.progress.emit(DownloadEvent::Progress {
                    files: local_cached
                        .iter()
                        .map(|m| FileProgress {
                            filename: m.filename.clone(),
                            bytes_completed: m.file_size,
                            total_bytes: m.file_size,
                            status: FileStatus::Complete,
                        })
                        .collect(),
                });
            }

            let xet_batch_fut = async {
                if xet_metas.is_empty() {
                    return Ok::<_, HFError>(());
                }
                let batch_files: Vec<crate::xet::XetBatchFile> = xet_metas
                    .iter()
                    .map(|m| crate::xet::XetBatchFile {
                        hash: m.xet_hash.as_ref().unwrap().clone(),
                        file_size: m.file_size,
                        path: local_dir.join(&m.filename),
                        filename: m.filename.clone(),
                    })
                    .collect();
                self.xet_download_batch(&commit_hash, &batch_files, &params.progress).await?;
                Ok(())
            };

            let non_xet_dl_params = build_download_params(
                &repo_path,
                &non_xet_filenames,
                &commit_hash,
                params.force_download,
                Some(local_dir.clone()),
                &params.progress,
            );
            let non_xet_fut = async {
                download_concurrently(self, &non_xet_dl_params, max_workers).await?;
                Ok::<_, HFError>(())
            };

            tokio::try_join!(xet_batch_fut, non_xet_fut)?;
            params.progress.emit(DownloadEvent::Complete);
            return Ok(local_dir.clone());
        }

        // Cache mode
        let mut cached_progress: Vec<FileProgress> = Vec::new();
        let mut uncached_metas = Vec::new();
        for meta in file_metas {
            let blob = cache::blob_path(cache_dir, &repo_folder, &meta.etag);
            if blob.exists() && !force {
                cache::create_pointer_symlink(cache_dir, &repo_folder, &meta.commit_hash, &meta.filename, &meta.etag)
                    .await?;
                cached_progress.push(FileProgress {
                    filename: meta.filename.clone(),
                    bytes_completed: meta.file_size,
                    total_bytes: meta.file_size,
                    status: FileStatus::Complete,
                });
                continue;
            }
            uncached_metas.push(meta);
        }
        let (blob_owner_metas, shared_blob_metas) = split_shared_blobs(uncached_metas);
        for meta in blob_owner_metas {
            if meta.xet_hash.is_some() {
                xet_metas.push(meta);
            } else {
                non_xet_filenames.push(meta.filename);
            }
        }
        xet_metas.sort_by(|a, b| a.etag.cmp(&b.etag));
        if !cached_progress.is_empty() {
            params.progress.emit(DownloadEvent::Progress { files: cached_progress });
        }

        let xet_batch_fut = async {
            if xet_metas.is_empty() {
                return Ok::<_, HFError>(());
            }
            let mut locks = Vec::with_capacity(xet_metas.len());
            for m in &xet_metas {
                locks.push(cache::acquire_lock(cache_dir, &repo_folder, &m.etag).await?);
            }
            let batch_files: Vec<crate::xet::XetBatchFile> = xet_metas
                .iter()
                .map(|m| crate::xet::XetBatchFile {
                    hash: m.xet_hash.as_ref().unwrap().clone(),
                    file_size: m.file_size,
                    path: cache::blob_path(cache_dir, &repo_folder, &m.etag),
                    filename: m.filename.clone(),
                })
                .collect();
            self.xet_download_batch(&commit_hash, &batch_files, &params.progress).await?;
            for m in &xet_metas {
                cache::create_pointer_symlink(cache_dir, &repo_folder, &m.commit_hash, &m.filename, &m.etag).await?;
            }
            drop(locks);
            Ok(())
        };

        let non_xet_dl_params = build_download_params(
            &repo_path,
            &non_xet_filenames,
            &commit_hash,
            params.force_download,
            None,
            &params.progress,
        );
        let non_xet_fut = async {
            download_concurrently(self, &non_xet_dl_params, max_workers).await?;
            Ok::<_, HFError>(())
        };

        tokio::try_join!(xet_batch_fut, non_xet_fut)?;

        if !shared_blob_metas.is_empty() {
            for m in &shared_blob_metas {
                cache::create_pointer_symlink(cache_dir, &repo_folder, &m.commit_hash, &m.filename, &m.etag).await?;
            }
            params.progress.emit(DownloadEvent::Progress {
                files: shared_blob_metas
                    .iter()
                    .map(|m| FileProgress {
                        filename: m.filename.clone(),
                        bytes_completed: m.file_size,
                        total_bytes: m.file_size,
                        status: FileStatus::Complete,
                    })
                    .collect(),
            });
        }

        if !cache::is_commit_hash(revision) {
            cache::write_ref(cache_dir, &repo_folder, revision, &commit_hash).await?;
        }

        params.progress.emit(DownloadEvent::Complete);
        Ok(cache_dir.join(&repo_folder).join("snapshots").join(&commit_hash))
    }
}

/// Announce a file served from the local cache: `Start` (unless one was already emitted for
/// this call) and a per-file `Complete`, both sized from the file on disk.
#[cfg(not(target_family = "wasm"))]
fn announce_cached_file(params: &DownloadFileParams, path: &Path, start_emitted: bool) -> HFResult<()> {
    if params.progress.is_none() {
        return Ok(());
    }
    let size = std::fs::metadata(path)?.len();
    if params.announce_start && !start_emitted {
        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: size,
        });
    }
    emit_file_complete(&params.progress, &params.filename, size);
    Ok(())
}

/// Like `announce_cached_file`, for the fallback-to-cache path after a transient network error:
/// the file is already resolved from cache, so a metadata read failure here must not turn that
/// success into an error. Reuses the size already announced in `Start`, if any, instead of
/// re-reading it from disk, so the reported size can't drift from what `Start` promised.
#[cfg(not(target_family = "wasm"))]
fn announce_cached_file_after_transient_error(params: &DownloadFileParams, path: &Path, start_size: Option<u64>) {
    if params.progress.is_none() {
        return;
    }
    let size = if let Some(size) = start_size {
        size
    } else {
        match std::fs::metadata(path) {
            Ok(meta) => meta.len(),
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to read cached file size after transient download error; skipping progress event"
                );
                return;
            },
        }
    };
    if params.announce_start && start_size.is_none() {
        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: size,
        });
    }
    emit_file_complete(&params.progress, &params.filename, size);
}

#[cfg(not(target_family = "wasm"))]
fn emit_file_complete(progress: &Option<Progress>, filename: &str, size: u64) {
    progress.emit(DownloadEvent::Progress {
        files: vec![FileProgress {
            filename: filename.to_string(),
            bytes_completed: size,
            total_bytes: size,
            status: FileStatus::Complete,
        }],
    });
}

/// Split files into one owner per blob and the remaining files whose content (etag) an owner
/// already covers. Several filenames with identical content share one cache blob, so only the
/// owners are fetched; the rest are linked to the owner's blob afterwards.
#[cfg(not(target_family = "wasm"))]
fn split_shared_blobs(metas: Vec<FileMetadataInfo>) -> (Vec<FileMetadataInfo>, Vec<FileMetadataInfo>) {
    let mut seen_etags = HashSet::new();
    metas.into_iter().partition(|m| seen_etags.insert(m.etag.clone()))
}

#[cfg(not(target_family = "wasm"))]
async fn mark_no_exist_and_return_error(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
    response: &reqwest::Response,
    repo_id: &str,
    filename: &str,
) -> HFError {
    if let Some(commit_hash) = extract_commit_hash(response) {
        let no_exist = cache::no_exist_path(cache_dir, repo_folder, &commit_hash, filename);
        if let Some(parent) = no_exist.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&no_exist, b"");
        if !cache::is_commit_hash(revision) {
            let _ = cache::write_ref(cache_dir, repo_folder, revision, &commit_hash).await;
        }
    }
    HFError::EntryNotFound {
        path: filename.to_string(),
        repo_id: repo_id.to_string(),
        context: None,
    }
}

#[cfg(not(target_family = "wasm"))]
async fn finalize_cached_file(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
    commit_hash: &str,
    filename: &str,
    etag: &str,
) -> HFResult<PathBuf> {
    if !cache::is_commit_hash(revision) {
        cache::write_ref(cache_dir, repo_folder, revision, commit_hash).await?;
    }
    cache::create_pointer_symlink(cache_dir, repo_folder, commit_hash, filename, etag).await?;
    Ok(cache::snapshot_path(cache_dir, repo_folder, commit_hash, filename))
}

#[cfg(not(target_family = "wasm"))]
fn build_download_params(
    _repo_id: &str,
    filenames: &[String],
    commit_hash: &str,
    force_download: bool,
    local_dir: Option<PathBuf>,
    progress: &Option<Progress>,
) -> Vec<DownloadFileParams> {
    filenames
        .iter()
        .map(|filename| DownloadFileParams {
            filename: filename.clone(),
            local_dir: local_dir.clone(),
            revision: Some(commit_hash.to_string()),
            force_download,
            local_files_only: false,
            progress: progress.clone(),
            announce_start: false,
        })
        .collect()
}

#[cfg(not(target_family = "wasm"))]
async fn download_concurrently<T: RepoType>(
    api: &HFRepository<T>,
    params: &[DownloadFileParams],
    max_workers: usize,
) -> HFResult<Vec<PathBuf>> {
    let mut download_futs = Vec::with_capacity(params.len());
    for file_params in params {
        download_futs.push(api.download_file_inner(file_params));
    }
    futures::stream::iter(download_futs)
        .buffer_unordered(max_workers)
        .try_collect()
        .await
}

#[cfg(not(target_family = "wasm"))]
async fn stream_response_to_file_with_progress(
    response: reqwest::Response,
    dest: &Path,
    handler: &Option<Progress>,
    filename: Option<&str>,
    total_bytes: u64,
) -> HFResult<()> {
    let mut file = std::fs::File::create(dest)?;
    let mut stream = response.bytes_stream();
    let mut bytes_read: u64 = 0;

    if let (Some(h), Some(filename)) = (handler, filename) {
        h.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: filename.to_string(),
                bytes_completed: 0,
                total_bytes,
                status: FileStatus::Started,
            }],
        });
    }

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        bytes_read += chunk.len() as u64;

        if let (Some(h), Some(filename)) = (handler, filename) {
            h.emit(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: filename.to_string(),
                    bytes_completed: bytes_read,
                    total_bytes,
                    status: FileStatus::InProgress,
                }],
            });
        }
    }
    file.flush()?;
    Ok(())
}

/// Decouple the inner byte stream from the JS-side `ReadableStream` reader cadence on wasm.
///
/// `wasm_streams::ReadableStream::from_stream` builds the JS stream with `QueuingStrategy(0.0)`
/// (HWM=0) — the underlying Rust stream is only polled when JS calls `reader.read()`. For the xet
/// download path, slow JS-side consumption then propagates through hf-xet's term-permit semaphore
/// (`AdjustableSemaphore` in `file_reconstructor.rs`) and stalls xorb fetching. Browsers differ in
/// how aggressively they schedule the `pull` callback, so the effect is intermittent.
///
/// This pump task drains the inner stream into a small bounded channel under `spawn_local`. JS
/// `reader.read()` now pulls from the local channel (cheap) instead of gating the live xet
/// pipeline; xet keeps making progress as long as the channel has room. The channel depth bounds
/// the extra in-flight bytes — at most `depth * max_chunk_size` beyond what xet already buffers.
#[cfg(target_family = "wasm")]
pub(crate) fn buffer_wasm_stream(inner: HFByteStream) -> HFByteStream {
    use futures::SinkExt;
    use futures::channel::mpsc;

    const BUFFER_DEPTH: usize = 2;
    let (mut tx, rx) = mpsc::channel::<HFResult<bytes::Bytes>>(BUFFER_DEPTH);

    wasm_bindgen_futures::spawn_local(async move {
        let mut inner = inner;
        while let Some(item) = inner.next().await {
            let is_err = item.is_err();
            if tx.send(item).await.is_err() {
                return;
            }
            if is_err {
                return;
            }
        }
    });

    Box::new(Box::pin(rx))
}

pub(crate) fn wrap_stream_with_progress(
    stream: HFByteStream,
    progress: Option<Progress>,
    filename: String,
    total_bytes: u64,
) -> HFByteStream {
    if progress.is_none() {
        return stream;
    }
    let wrapped = futures::stream::unfold((stream, 0u64, false), move |(mut inner, bytes_completed, ended)| {
        let progress = progress.clone();
        let filename = filename.clone();
        async move {
            if ended {
                return None;
            }
            match inner.next().await {
                Some(Ok(chunk)) => {
                    let bytes_completed = bytes_completed + chunk.len() as u64;
                    progress.emit(DownloadEvent::Progress {
                        files: vec![FileProgress {
                            filename,
                            bytes_completed,
                            total_bytes,
                            status: FileStatus::InProgress,
                        }],
                    });
                    Some((Ok(chunk), (inner, bytes_completed, false)))
                },
                Some(Err(e)) => Some((Err(e), (inner, bytes_completed, true))),
                None => {
                    progress.emit(DownloadEvent::Complete);
                    None
                },
            }
        }
    });
    Box::new(Box::pin(wrapped))
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Download a single file from a repository.
    ///
    /// When `local_dir` is `Some`, the file is downloaded directly to that directory
    /// (no caching). When `local_dir` is `None`, the HF cache system is used:
    /// blobs are stored by etag and symlinked from snapshots/{commit}/{filename}.
    ///
    /// Returns the local filesystem path of the downloaded or cached file. Use
    /// [`HFRepository::download_file_stream`] or
    /// [`HFRepository::download_file_to_bytes`] when you do not want to write to
    /// disk.
    ///
    /// # Offline / cache-only lookups
    ///
    /// Set `.local_files_only(true)` to resolve strictly from the local cache
    /// without any network request — this is the replacement for the 0.x
    /// `Cache::get` API. A cache miss returns
    /// [`HFError::LocalEntryNotFound`], which is distinct from a real failure:
    /// match it to tell "not cached" apart from a genuinely missing file
    /// ([`HFError::EntryNotFound`]) or any transport error.
    ///
    /// ```no_run
    /// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
    /// use hf_hub::HFError;
    ///
    /// let repo = hf_hub::HFClient::new()?.model("openai-community", "gpt2");
    /// match repo.download_file().filename("config.json").local_files_only(true).send().await {
    ///     Ok(path) => println!("cached at {}", path.display()),
    ///     Err(HFError::LocalEntryNotFound { .. }) => println!("not in cache"),
    ///     Err(e) => return Err(e),
    /// }
    /// # Ok(()) }
    /// ```
    ///
    /// Endpoint: `GET {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filename}`.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to download within the repository.
    /// - `local_dir`: local directory to download the file into. When set, the file is saved with its repo path
    ///   structure.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `force_download` (default `false`): re-download the file even if a cached copy exists.
    /// - `local_files_only` (default `false`): only return the file if cached locally; never make a network request.
    /// - `progress`: optional progress handler.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file(
        &self,
        /// Path of the file to download within the repository.
        #[builder(into)]
        filename: String,
        /// Local directory to download the file into. When set, the file is saved with its repo path structure.
        #[builder(into)]
        local_dir: Option<PathBuf>,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Re-download the file even if a cached copy exists.
        #[builder(default)]
        force_download: bool,
        /// Only return the file if cached locally; never make a network request.
        #[builder(default)]
        local_files_only: bool,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        Box::pin(self.download_file_impl(DownloadFileParams {
            filename,
            local_dir,
            revision,
            force_download,
            local_files_only,
            progress,
            announce_start: true,
        }))
        .await
    }

    /// Download a file and return a byte stream instead of writing to disk.
    ///
    /// Returns a `(content_length, stream)` tuple. `content_length` is `Some`
    /// when the server provides a `Content-Length` header.
    ///
    /// When `range` is set, only the specified byte range is fetched.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to stream within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `range`: byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
    ///   `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
    ///   `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must be
    ///   strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
    /// - `progress`: optional progress handler. `Start` is emitted before the stream is returned; `Progress` is emitted
    ///   as the caller polls each chunk; `Complete` is emitted when the stream is exhausted.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file_stream(
        &self,
        /// Path of the file to stream within the repository.
        #[builder(into)]
        filename: String,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
        /// `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
        /// `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must
        /// be strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
        range: Option<std::ops::Range<u64>>,
        /// Progress handler. `Start` is emitted before the stream is returned; `Progress` is emitted
        /// as the caller polls each chunk; `Complete` is emitted when the stream is exhausted.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<(Option<u64>, HFByteStream)> {
        Box::pin(self.download_file_stream_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
            progress,
        }))
        .await
    }

    /// Download a file (or byte range) into memory and return the contents as [`bytes::Bytes`].
    ///
    /// This is a convenience wrapper around
    /// [`download_file_stream`](Self::download_file_stream) that collects the entire stream into
    /// a single buffer. When `range` is set, only the specified byte range is fetched.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to download within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `range`: byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
    ///   `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
    ///   `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must be
    ///   strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
    /// - `progress`: optional progress handler. Emits `Start`/`Progress`/`Complete` as the underlying stream is
    ///   drained, identically to [`download_file_stream`](Self::download_file_stream).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file_to_bytes(
        &self,
        /// Path of the file to download within the repository.
        #[builder(into)]
        filename: String,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
        /// `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
        /// `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must
        /// be strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
        range: Option<std::ops::Range<u64>>,
        /// Progress handler. Emits `Start`/`Progress`/`Complete` as the underlying stream is
        /// drained, identically to [`HFRepository::download_file_stream`].
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<bytes::Bytes> {
        Box::pin(self.download_file_to_bytes_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
            progress,
        }))
        .await
    }

    /// Download all selected files for a resolved revision.
    ///
    /// When `local_dir` is `None`, files are stored in the HF cache, and the returned path is the
    /// cache snapshot directory for the resolved commit. When `local_dir` is `Some`, files are
    /// written directly under that directory.
    ///
    /// `allow_patterns` and `ignore_patterns` use [`globset`](https://docs.rs/globset) syntax
    /// (`*`, `?`, `**`, character classes, etc.). Both are matched against each candidate file's
    /// **repository path** — forward-slash-joined and relative to the repo root, e.g.,
    /// `tokenizer.json` or `weights/model-00001-of-00003.safetensors`.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `allow_patterns`: globs selecting which repository files to download. When set, only files whose repo path
    ///   matches at least one pattern are downloaded.
    /// - `ignore_patterns`: globs of repository files to skip. Matched against the same repo paths as `allow_patterns`.
    /// - `local_dir`: local directory to download into.
    /// - `force_download` (default `false`): re-download all files even if cached.
    /// - `local_files_only` (default `false`): resolve only from the local cache.
    /// - `max_workers`: maximum concurrent file downloads (default 8).
    /// - `progress`: optional progress handler.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn snapshot_download(
        &self,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Globs selecting which repository files to download. When set, only files whose repo path
        /// matches at least one pattern are downloaded.
        allow_patterns: Option<Vec<String>>,
        /// Globs of repository files to skip. Matched against the same repo paths as `allow_patterns`.
        ignore_patterns: Option<Vec<String>>,
        /// Local directory to download into.
        #[builder(into)]
        local_dir: Option<PathBuf>,
        /// Re-download all files even if cached.
        #[builder(default)]
        force_download: bool,
        /// Resolve only from the local cache.
        #[builder(default)]
        local_files_only: bool,
        /// Maximum concurrent file downloads (default 8).
        max_workers: Option<usize>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        Box::pin(self.snapshot_download_impl(SnapshotDownloadParams {
            revision,
            allow_patterns,
            ignore_patterns,
            local_dir,
            force_download,
            local_files_only,
            max_workers,
            progress,
        }))
        .await
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::download_file`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn download_file(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] local_dir: Option<PathBuf>,
        #[builder(into)] revision: Option<String>,
        #[builder(default)] force_download: bool,
        #[builder(default)] local_files_only: bool,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .download_file()
                .filename(filename)
                .maybe_local_dir(local_dir)
                .maybe_revision(revision)
                .force_download(force_download)
                .local_files_only(local_files_only)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::download_file_to_bytes`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn download_file_to_bytes(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
        range: Option<std::ops::Range<u64>>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<bytes::Bytes> {
        self.runtime.block_on(
            self.inner
                .download_file_to_bytes()
                .filename(filename)
                .maybe_revision(revision)
                .maybe_range(range)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::snapshot_download`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn snapshot_download(
        &self,
        #[builder(into)] revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        #[builder(into)] local_dir: Option<PathBuf>,
        #[builder(default)] force_download: bool,
        #[builder(default)] local_files_only: bool,
        max_workers: Option<usize>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .snapshot_download()
                .maybe_revision(revision)
                .maybe_allow_patterns(allow_patterns)
                .maybe_ignore_patterns(ignore_patterns)
                .maybe_local_dir(local_dir)
                .force_download(force_download)
                .local_files_only(local_files_only)
                .maybe_max_workers(max_workers)
                .maybe_progress(progress)
                .send(),
        )
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    use super::split_shared_blobs;
    use crate::HFClient;
    use crate::cache::storage as cache;
    use crate::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler};
    use crate::repository::{AddSource, FileMetadataInfo};

    const REPO: &str = "acme/dups";
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[derive(Clone)]
    struct MockFile {
        path: String,
        etag: String,
        xet_hash: Option<String>,
        body: Vec<u8>,
        fail_get: bool,
    }

    impl MockFile {
        fn plain(path: &str, etag: &str, body: &[u8]) -> Self {
            Self {
                path: path.to_string(),
                etag: etag.to_string(),
                xet_hash: None,
                body: body.to_vec(),
                fail_get: false,
            }
        }
    }

    /// Loopback stand-in for the Hub: the tree listing, `HEAD`/`GET` on `resolve`, and Xet
    /// read/write tokens pointing at a `local://` CAS directory. Everything else is a 404.
    struct MockHub {
        endpoint: String,
        files: Arc<Mutex<Vec<MockFile>>>,
        requests: Arc<Mutex<Vec<String>>>,
        _cas_dir: tempfile::TempDir,
    }

    impl MockHub {
        async fn start() -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let cas_dir = tempfile::tempdir().unwrap();
            let cas_url = format!("local://{}", cas_dir.path().display());
            let files: Arc<Mutex<Vec<MockFile>>> = Arc::default();
            let requests: Arc<Mutex<Vec<String>>> = Arc::default();
            let (served_files, log) = (Arc::clone(&files), Arc::clone(&requests));
            tokio::spawn(async move {
                while let Ok((socket, _)) = listener.accept().await {
                    let (files, log, cas_url) = (Arc::clone(&served_files), Arc::clone(&log), cas_url.clone());
                    tokio::spawn(async move {
                        let mut socket = BufReader::new(socket);
                        let mut request_line = String::new();
                        if socket.read_line(&mut request_line).await.unwrap_or(0) == 0 {
                            return;
                        }
                        let mut if_none_match = None;
                        loop {
                            let mut line = String::new();
                            if socket.read_line(&mut line).await.unwrap_or(0) == 0 || line == "\r\n" {
                                break;
                            }
                            if let Some((name, value)) = line.split_once(':')
                                && name.eq_ignore_ascii_case("if-none-match")
                            {
                                if_none_match = Some(value.trim().trim_matches('"').to_string());
                            }
                        }
                        let mut parts = request_line.split_whitespace();
                        let method = parts.next().unwrap_or_default().to_string();
                        let path = parts
                            .next()
                            .unwrap_or_default()
                            .split('?')
                            .next()
                            .unwrap_or_default()
                            .to_string();
                        log.lock().unwrap().push(format!("{method} {path}"));

                        let files = files.lock().unwrap().clone();
                        let (status, headers, body) = route(&files, &cas_url, &method, &path, if_none_match.as_deref());
                        if status.starts_with("304") {
                            log.lock().unwrap().push(format!("304 {path}"));
                        }
                        let mut response = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
                        for (name, value) in headers {
                            response.push_str(&format!("{name}: {value}\r\n"));
                        }
                        response.push_str("\r\n");
                        let mut bytes = response.into_bytes();
                        if method != "HEAD" {
                            bytes.extend_from_slice(&body);
                        }
                        let _ = socket.get_mut().write_all(&bytes).await;
                    });
                }
            });
            Self {
                endpoint,
                files,
                requests,
                _cas_dir: cas_dir,
            }
        }

        fn client(&self, cache_dir: &Path) -> HFClient {
            HFClient::builder()
                .endpoint(&self.endpoint)
                .client(reqwest::Client::builder().no_proxy().build().unwrap())
                .cache_dir(cache_dir)
                .retry_max_attempts(1)
                .build()
                .unwrap()
        }

        /// Upload `body` to the local CAS and serve it at `path` as a Xet file.
        async fn add_xet_file(&self, path: &str, body: &[u8]) {
            let cache = tempfile::tempdir().unwrap();
            let infos = self
                .client(cache.path())
                .model("acme", "dups")
                .xet_upload(&[(path.to_string(), AddSource::bytes(body.to_vec()))], "main", false, &None)
                .await
                .unwrap();
            let hash = infos[0].hash.clone();
            self.files.lock().unwrap().push(MockFile {
                path: path.to_string(),
                etag: format!("sha-{hash}"),
                xet_hash: Some(hash),
                body: body.to_vec(),
                fail_get: false,
            });
        }

        fn add_file(&self, file: MockFile) {
            self.files.lock().unwrap().push(file);
        }

        fn count(&self, request: &str) -> usize {
            self.requests.lock().unwrap().iter().filter(|r| *r == request).count()
        }
    }

    fn route(
        files: &[MockFile],
        cas_url: &str,
        method: &str,
        path: &str,
        if_none_match: Option<&str>,
    ) -> (&'static str, Vec<(String, String)>, Vec<u8>) {
        let json = |body: Vec<u8>| {
            let headers = vec![
                ("Content-Type".into(), "application/json".into()),
                ("Content-Length".into(), body.len().to_string()),
            ];
            ("200 OK", headers, body)
        };
        let token_prefix = format!("/api/models/{REPO}/xet-");
        if method == "GET" && path.starts_with(&token_prefix) {
            let token = serde_json::json!({"accessToken": "token", "exp": 4_102_444_800u64, "casUrl": cas_url});
            return json(serde_json::to_vec(&token).unwrap());
        }
        if method == "GET" && path == format!("/api/models/{REPO}/tree/{COMMIT}") {
            let entries: Vec<serde_json::Value> = files
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "type": "file",
                        "oid": f.etag,
                        "size": f.body.len(),
                        "path": f.path,
                        "xetHash": f.xet_hash,
                    })
                })
                .collect();
            return json(serde_json::to_vec(&entries).unwrap());
        }
        let resolve_prefix = format!("/{REPO}/resolve/");
        if let Some(file) = path
            .strip_prefix(&resolve_prefix)
            .and_then(|rest| rest.strip_prefix(&format!("{COMMIT}/")).or_else(|| rest.strip_prefix("main/")))
            .and_then(|name| files.iter().find(|f| f.path == name))
        {
            if method == "GET" && file.fail_get {
                return ("500 Internal Server Error", vec![("Content-Length".into(), "0".into())], Vec::new());
            }
            if method == "HEAD" && if_none_match == Some(file.etag.as_str()) {
                let headers = vec![
                    ("ETag".into(), format!("\"{}\"", file.etag)),
                    ("X-Repo-Commit".into(), COMMIT.into()),
                    ("Content-Length".into(), "0".into()),
                ];
                return ("304 Not Modified", headers, Vec::new());
            }
            let mut headers = vec![
                ("ETag".into(), format!("\"{}\"", file.etag)),
                ("X-Repo-Commit".into(), COMMIT.into()),
                ("Content-Length".into(), file.body.len().to_string()),
            ];
            if let Some(hash) = &file.xet_hash {
                headers.push(("X-Xet-Hash".into(), hash.clone()));
            }
            return ("200 OK", headers, file.body.clone());
        }
        ("404 Not Found", vec![("Content-Length".into(), "0".into())], Vec::new())
    }

    #[derive(Default)]
    struct RecordingHandler(Mutex<Vec<DownloadEvent>>);

    impl ProgressHandler for RecordingHandler {
        fn on_progress(&self, event: &ProgressEvent) {
            if let ProgressEvent::Download(event) = event {
                self.0.lock().unwrap().push(event.clone());
            }
        }
    }

    impl RecordingHandler {
        /// Check the download contract documented in `crate::progress` and return the final
        /// `(bytes_completed, total_bytes)` per file.
        fn assert_download_contract(&self, expected_files: usize, expected_bytes: u64) -> HashMap<String, (u64, u64)> {
            let events = self.0.lock().unwrap().clone();
            let starts: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    DownloadEvent::Start {
                        total_files,
                        total_bytes,
                    } => Some((*total_files, *total_bytes)),
                    _ => None,
                })
                .collect();
            assert_eq!(starts, [(expected_files, expected_bytes)], "exactly one authoritative Start");
            assert!(matches!(events.first(), Some(DownloadEvent::Start { .. })), "Start comes first");
            assert!(matches!(events.last(), Some(DownloadEvent::Complete)), "Complete comes last");
            assert_eq!(events.iter().filter(|e| matches!(e, DownloadEvent::Complete)).count(), 1);

            let mut latest: HashMap<String, (u64, u64)> = HashMap::new();
            let mut completed: HashSet<String> = HashSet::new();
            let mut overall = 0u64;
            for event in &events {
                let DownloadEvent::Progress { files } = event else {
                    continue;
                };
                for f in files {
                    if let Some((done, total)) = latest.get(&f.filename) {
                        assert_eq!(*total, f.total_bytes, "per-file total changed for {}", f.filename);
                        assert!(f.bytes_completed >= *done, "per-file bytes went backwards for {}", f.filename);
                    }
                    latest.insert(f.filename.clone(), (f.bytes_completed, f.total_bytes));
                    if f.status == FileStatus::Complete {
                        assert_eq!(f.bytes_completed, f.total_bytes, "{} completed short", f.filename);
                        completed.insert(f.filename.clone());
                    }
                }
                let sum: u64 = latest.values().map(|(done, _)| done).sum();
                assert!(sum >= overall && sum <= expected_bytes, "overall bytes {sum} not monotonic within total");
                overall = sum;
            }
            assert_eq!(completed.len(), expected_files, "every file completes: {completed:?}");
            assert_eq!(overall, expected_bytes, "per-file bytes add up to Start.total_bytes");
            latest
        }
    }

    fn meta(filename: &str, etag: &str) -> FileMetadataInfo {
        FileMetadataInfo {
            filename: filename.to_string(),
            etag: etag.to_string(),
            commit_hash: COMMIT.to_string(),
            xet_hash: Some(etag.to_string()),
            file_size: 10,
            location: None,
        }
    }

    #[test]
    fn split_shared_blobs_keeps_one_owner_per_etag() {
        let (owners, shared) = split_shared_blobs(vec![
            meta("onnx/model_qint8_arm64.onnx", "q8"),
            meta("model.onnx", "full"),
            meta("onnx/model_qint8_avx512.onnx", "q8"),
            meta("onnx/model_qint8_avx512_vnni.onnx", "q8"),
        ]);
        let owners: Vec<_> = owners.iter().map(|m| m.filename.as_str()).collect();
        let shared: Vec<_> = shared.iter().map(|m| m.filename.as_str()).collect();
        assert_eq!(owners, ["onnx/model_qint8_arm64.onnx", "model.onnx"]);
        assert_eq!(shared, ["onnx/model_qint8_avx512.onnx", "onnx/model_qint8_avx512_vnni.onnx"]);
    }

    #[tokio::test]
    async fn cache_snapshot_fetches_a_shared_plain_blob_once() {
        let hub = MockHub::start().await;
        for path in ["config.json", "copies/config.json", "copies/again.json"] {
            hub.add_file(MockFile::plain(path, "shared-etag", b"same bytes"));
        }
        hub.add_file(MockFile::plain("README.md", "readme-etag", b"readme"));
        let cache = tempfile::tempdir().unwrap();
        let handler = Arc::new(RecordingHandler::default());

        let snapshot = hub
            .client(cache.path())
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        for name in ["config.json", "copies/config.json", "copies/again.json"] {
            assert_eq!(std::fs::read(snapshot.join(name)).unwrap(), b"same bytes", "{name}");
        }
        assert_eq!(std::fs::read(snapshot.join("README.md")).unwrap(), b"readme");
        let shared_gets: usize = ["config.json", "copies/config.json", "copies/again.json"]
            .iter()
            .map(|name| hub.count(&format!("GET /{REPO}/resolve/{COMMIT}/{name}")))
            .sum();
        assert_eq!(shared_gets, 1, "the shared blob must be fetched exactly once");
        handler.assert_download_contract(4, 3 * 10 + 6);
    }

    /// Xet files with identical content used to take the same cache lock once per filename, so
    /// the second acquisition blocked on the first until the lock timed out.
    #[tokio::test(flavor = "multi_thread")]
    async fn cache_snapshot_with_shared_xet_blob_downloads_it_once_and_links_every_filename() {
        let hub = MockHub::start().await;
        let names = [
            "onnx/model_qint8_arm64.onnx",
            "onnx/model_qint8_avx512.onnx",
            "onnx/model_qint8_avx512_vnni.onnx",
        ];
        for name in names {
            hub.add_xet_file(name, b"quantized weights").await;
        }
        let cache = tempfile::tempdir().unwrap();
        let handler = Arc::new(RecordingHandler::default());

        let snapshot = hub
            .client(cache.path())
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        for name in names {
            assert_eq!(std::fs::read(snapshot.join(name)).unwrap(), b"quantized weights", "{name}");
        }
        let blobs: Vec<_> = std::fs::read_dir(cache.path().join("models--acme--dups").join("blobs"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(blobs.len(), 1, "one blob, no leftover .incomplete files: {blobs:?}");
        handler.assert_download_contract(3, 3 * 17);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mixed_cache_snapshot_announces_totals_once() {
        let hub = MockHub::start().await;
        hub.add_xet_file("model.safetensors", &[7u8; 300_000]).await;
        hub.add_xet_file("tokenizer.json", b"{\"tokens\": []}").await;
        hub.add_file(MockFile::plain("config.json", "config-etag", b"{\"a\": 1}"));
        hub.add_file(MockFile::plain("README.md", "readme-etag", b"# readme"));
        let cache = tempfile::tempdir().unwrap();
        let client = hub.client(cache.path());
        let expected_bytes = 300_000 + 14 + 8 + 8;

        client
            .model("acme", "dups")
            .download_file()
            .filename("README.md")
            .revision(COMMIT)
            .send()
            .await
            .unwrap();

        let handler = Arc::new(RecordingHandler::default());
        client
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        let latest = handler.assert_download_contract(4, expected_bytes);
        assert_eq!(latest["README.md"], (8, 8), "an already-present file still reports its size");
        assert_eq!(hub.count(&format!("HEAD /{REPO}/resolve/{COMMIT}/README.md")), 1, "only the first download");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mixed_local_dir_snapshot_announces_totals_once() {
        let hub = MockHub::start().await;
        hub.add_xet_file("model.safetensors", &[3u8; 200_000]).await;
        hub.add_file(MockFile::plain("config.json", "config-etag", b"{\"a\": 1}"));
        hub.add_file(MockFile::plain("nested/notes.txt", "notes-etag", b"notes"));
        let cache = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dest.path().join("nested")).unwrap();
        std::fs::write(dest.path().join("nested/notes.txt"), b"notes").unwrap();
        let handler = Arc::new(RecordingHandler::default());

        hub.client(cache.path())
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .local_dir(dest.path())
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        let latest = handler.assert_download_contract(3, 200_000 + 8 + 5);
        assert_eq!(latest["nested/notes.txt"], (5, 5));
        assert_eq!(std::fs::read(dest.path().join("model.safetensors")).unwrap(), vec![3u8; 200_000]);
    }

    #[tokio::test]
    async fn download_file_snapshot_hit_keeps_the_contract() {
        let hub = MockHub::start().await;
        hub.add_file(MockFile::plain("config.json", "config-etag", b"{\"a\": 1}"));
        let cache = tempfile::tempdir().unwrap();
        let repo = hub.client(cache.path()).model("acme", "dups");
        repo.download_file()
            .filename("config.json")
            .revision(COMMIT)
            .send()
            .await
            .unwrap();

        let handler = Arc::new(RecordingHandler::default());
        let path = repo
            .download_file()
            .filename("config.json")
            .revision(COMMIT)
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        assert_eq!(std::fs::read(path).unwrap(), b"{\"a\": 1}");
        handler.assert_download_contract(1, 8);
        assert_eq!(hub.count(&format!("HEAD /{REPO}/resolve/{COMMIT}/config.json")), 1);
    }

    #[tokio::test]
    async fn download_file_not_modified_and_local_files_only_keep_the_contract() {
        let hub = MockHub::start().await;
        hub.add_file(MockFile::plain("config.json", "config-etag", b"{\"a\": 1}"));
        let cache = tempfile::tempdir().unwrap();
        let repo = hub.client(cache.path()).model("acme", "dups");
        repo.download_file().filename("config.json").send().await.unwrap();

        let handler = Arc::new(RecordingHandler::default());
        let path = repo
            .download_file()
            .filename("config.json")
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        assert_eq!(std::fs::read(path).unwrap(), b"{\"a\": 1}");
        assert_eq!(hub.count(&format!("304 /{REPO}/resolve/main/config.json")), 1);
        assert_eq!(hub.count(&format!("GET /{REPO}/resolve/main/config.json")), 1);
        handler.assert_download_contract(1, 8);

        let handler = Arc::new(RecordingHandler::default());
        repo.download_file()
            .filename("config.json")
            .local_files_only(true)
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();
        handler.assert_download_contract(1, 8);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn download_file_blob_hit_keeps_the_contract_without_waiting_on_the_lock() {
        let hub = MockHub::start().await;
        hub.add_file(MockFile::plain("a.txt", "shared-etag", b"same bytes"));
        hub.add_file(MockFile::plain("b.txt", "shared-etag", b"same bytes"));
        hub.add_xet_file("weights/a.bin", b"shared weights").await;
        hub.add_xet_file("weights/b.bin", b"shared weights").await;
        let xet_etag = hub.files.lock().unwrap().last().unwrap().etag.clone();
        let cache = tempfile::tempdir().unwrap();
        let repo = hub.client(cache.path()).model("acme", "dups");
        for name in ["a.txt", "weights/a.bin"] {
            repo.download_file().filename(name).revision(COMMIT).send().await.unwrap();
        }

        let _held = [
            cache::acquire_lock(cache.path(), "models--acme--dups", "shared-etag")
                .await
                .unwrap(),
            cache::acquire_lock(cache.path(), "models--acme--dups", &xet_etag)
                .await
                .unwrap(),
        ];
        for (name, body) in [("b.txt", &b"same bytes"[..]), ("weights/b.bin", &b"shared weights"[..])] {
            let handler = Arc::new(RecordingHandler::default());
            let path = repo
                .download_file()
                .filename(name)
                .revision(COMMIT)
                .progress(Arc::clone(&handler))
                .send()
                .await
                .unwrap();
            assert_eq!(std::fs::read(path).unwrap(), body, "{name}");
            handler.assert_download_contract(1, body.len() as u64);
        }
    }

    #[tokio::test]
    async fn cache_snapshot_owner_failure_links_no_shared_filename() {
        let hub = MockHub::start().await;
        let names = ["config.json", "copies/config.json", "copies/again.json"];
        for name in names {
            hub.add_file(MockFile {
                fail_get: true,
                ..MockFile::plain(name, "shared-etag", b"same bytes")
            });
        }
        let cache = tempfile::tempdir().unwrap();
        let handler = Arc::new(RecordingHandler::default());

        let result = hub
            .client(cache.path())
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .progress(Arc::clone(&handler))
            .send()
            .await;

        assert!(result.is_err(), "{result:?}");
        let snapshot = cache.path().join("models--acme--dups").join("snapshots").join(COMMIT);
        for name in names {
            assert!(std::fs::symlink_metadata(snapshot.join(name)).is_err(), "{name} was linked");
        }
        let completed: Vec<_> = handler
            .0
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                DownloadEvent::Progress { files } => Some(files.clone()),
                _ => None,
            })
            .flatten()
            .filter(|f| f.status == FileStatus::Complete)
            .map(|f| f.filename)
            .collect();
        assert!(completed.is_empty(), "{completed:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn local_dir_snapshot_writes_every_file_with_shared_content() {
        let hub = MockHub::start().await;
        let plain = ["config.json", "copies/config.json"];
        let xet = ["onnx/model_arm64.onnx", "onnx/model_avx512.onnx"];
        for name in plain {
            hub.add_file(MockFile::plain(name, "shared-etag", b"same bytes"));
        }
        for name in xet {
            hub.add_xet_file(name, b"quantized weights").await;
        }
        let cache = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        let handler = Arc::new(RecordingHandler::default());

        hub.client(cache.path())
            .model("acme", "dups")
            .snapshot_download()
            .revision(COMMIT)
            .local_dir(dest.path())
            .progress(Arc::clone(&handler))
            .send()
            .await
            .unwrap();

        handler.assert_download_contract(4, 2 * 10 + 2 * 17);
        for name in plain {
            assert_eq!(std::fs::read(dest.path().join(name)).unwrap(), b"same bytes", "{name}");
        }
        for name in xet {
            assert_eq!(std::fs::read(dest.path().join(name)).unwrap(), b"quantized weights", "{name}");
        }
    }
}
