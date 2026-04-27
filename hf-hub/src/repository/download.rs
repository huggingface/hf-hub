use std::io::Write;
use std::path::{Path, PathBuf};

use bon::bon;
use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
use reqwest::header::IF_NONE_MATCH;

use super::files::{extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash, matches_any_glob};
use super::{FileMetadataInfo, HFRepository, RepoTreeEntry, RepoType};
use crate::cache::storage as cache;
use crate::error::{HFError, HFResult};
use crate::progress::{DownloadEvent, EmitEvent, FileProgress, FileStatus, Progress};
use crate::{constants, retry};

/// Internal options struct used by the file download helpers.
struct DownloadFileParams {
    filename: String,
    local_dir: Option<PathBuf>,
    revision: Option<String>,
    force_download: Option<bool>,
    local_files_only: Option<bool>,
    progress: Option<Progress>,
}

/// Internal options struct used by the streaming download helpers.
struct DownloadFileStreamParams {
    filename: String,
    revision: Option<String>,
    range: Option<std::ops::Range<u64>>,
}

/// Internal options struct used by `snapshot_download_impl`.
struct SnapshotDownloadParams {
    revision: Option<String>,
    allow_patterns: Option<Vec<String>>,
    ignore_patterns: Option<Vec<String>>,
    local_dir: Option<PathBuf>,
    force_download: Option<bool>,
    local_files_only: Option<bool>,
    max_workers: Option<usize>,
    progress: Option<Progress>,
}

impl HFRepository {
    async fn download_file_impl(&self, params: DownloadFileParams) -> HFResult<PathBuf> {
        let result = self.download_file_inner(&params).await;
        if result.is_ok() {
            params.progress.emit(DownloadEvent::Complete);
        }
        result
    }

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

    async fn download_file_stream_impl(
        &self,
        params: DownloadFileStreamParams,
    ) -> HFResult<(Option<u64>, Box<dyn Stream<Item = std::result::Result<bytes::Bytes, HFError>> + Send + Unpin>)>
    {
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
            .download_url(Some(self.repo_type), &repo_path, revision, &params.filename);

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

        if let Some(xet_hash) = extract_xet_hash(&head_response) {
            let file_size: u64 = extract_file_size(&head_response).unwrap_or_else(|| {
                    tracing::warn!(url = %url, "missing or invalid Content-Length/X-Linked-Size header for xet file, defaulting file size to 0");
                    0
                });

            let content_length = params.range.as_ref().map(|r| r.end.saturating_sub(r.start)).or(Some(file_size));

            let stream = self
                .xet_download_stream(revision, &xet_hash, file_size, params.range.clone())
                .await?;

            return Ok((content_length, Box::new(Box::pin(stream))));
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
        let stream = response.bytes_stream().map(|r| r.map_err(HFError::from));
        Ok((content_length, Box::new(Box::pin(stream))))
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

    async fn download_file_to_local_dir(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &repo_path, revision, &params.filename);

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

        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: file_size,
        });

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

    async fn download_file_to_cache(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let cache_dir = self.hf_client.cache_dir();
        let repo_folder = cache::repo_folder_name(&self.repo_path(), Some(self.repo_type));
        let force_download = params.force_download.unwrap_or(false);

        if cache::is_commit_hash(revision) && !force_download {
            let snap = cache::snapshot_path(cache_dir, &repo_folder, revision, &params.filename);
            if snap.exists() {
                return Ok(snap);
            }
        }

        if params.local_files_only.unwrap_or(false) {
            return self.resolve_from_cache_only(&repo_folder, revision, &params.filename);
        }

        let result = self
            .download_file_to_cache_network(params, revision, cache_dir, &repo_folder, force_download)
            .await;

        match &result {
            Err(e) if e.is_transient() && !force_download => self
                .resolve_from_cache_only(&repo_folder, revision, &params.filename)
                .or(result),
            _ => result,
        }
    }

    async fn download_file_to_cache_network(
        &self,
        params: &DownloadFileParams,
        revision: &str,
        cache_dir: &Path,
        repo_folder: &str,
        force_download: bool,
    ) -> HFResult<PathBuf> {
        let repo_path = self.repo_path();
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &repo_path, revision, &params.filename);

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

        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .no_redirect_client()
                .head(&url)
                .headers(head_headers.clone())
                .send()
        })
        .await?;

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
            let etag =
                cached_etag.ok_or_else(|| HFError::Other("Received 304 but no cached etag available".to_string()))?;
            let commit_hash = if cache::is_commit_hash(revision) {
                revision.to_string()
            } else {
                cache::read_ref(cache_dir, repo_folder, revision)
                    .await?
                    .ok_or_else(|| HFError::Other("Received 304 but no cached commit hash".to_string()))?
            };
            return finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await;
        }

        let etag =
            extract_etag(&head_response).ok_or_else(|| HFError::Other("Missing ETag header in response".to_string()));
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
        let commit_hash = commit_hash.ok_or_else(|| HFError::Other("Missing X-Repo-Commit header".to_string()))?;

        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: file_size,
        });

        if has_xet_hash {
            let xet_hash = xet_hash.ok_or_else(|| HFError::Other("Missing X-Xet-Hash header".to_string()))?;
            let blob = cache::blob_path(cache_dir, repo_folder, &etag);
            if !blob.exists() || force_download {
                if let Some(parent) = blob.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let _lock = cache::acquire_lock(cache_dir, repo_folder, &etag).await?;

                self.xet_download_to_blob(revision, &params.filename, &xet_hash, file_size, &blob, &params.progress)
                    .await?;
            }

            return finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await;
        }

        let blob = cache::blob_path(cache_dir, repo_folder, &etag);

        if blob.exists() && !force_download {
            params.progress.emit(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: params.filename.clone(),
                    bytes_completed: file_size,
                    total_bytes: file_size,
                    status: FileStatus::Complete,
                }],
            });
            return finalize_cached_file(cache_dir, repo_folder, revision, &commit_hash, &params.filename, &etag).await;
        }

        let _lock = cache::acquire_lock(cache_dir, repo_folder, &etag).await?;
        let incomplete_path = PathBuf::from(format!("{}.incomplete", blob.display()));
        if let Some(parent) = incomplete_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let dl_headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(dl_headers.clone()).send()
        })
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

    async fn resolve_commit_hash(&self, revision: &str) -> HFResult<String> {
        if cache::is_commit_hash(revision) {
            return Ok(revision.to_string());
        }
        let repo_path = self.repo_path();
        let sha = match self.repo_type {
            RepoType::Dataset => self.dataset_info(Some(revision.to_string()), None).await?.sha,
            RepoType::Space => self.space_info(Some(revision.to_string()), None).await?.sha,
            _ => self.model_info(Some(revision.to_string()), None).await?.sha,
        };
        sha.ok_or_else(|| HFError::Other(format!("No commit hash returned for {}/{}", repo_path, revision)))
    }

    async fn list_filtered_files(
        &self,
        revision: &str,
        allow_patterns: Option<&Vec<String>>,
        ignore_patterns: Option<&Vec<String>>,
    ) -> HFResult<Vec<String>> {
        let stream = self.list_tree().revision(revision.to_string()).recursive(true).send()?;
        futures::pin_mut!(stream);

        let mut filenames: Vec<String> = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, .. } = entry {
                filenames.push(path);
            }
        }

        if let Some(allow) = allow_patterns {
            filenames.retain(|f| matches_any_glob(allow, f));
        }
        if let Some(ignore) = ignore_patterns {
            filenames.retain(|f| !matches_any_glob(ignore, f));
        }

        Ok(filenames)
    }

    async fn snapshot_download_impl(&self, params: SnapshotDownloadParams) -> HFResult<PathBuf> {
        if params.local_dir.is_none() && !self.hf_client.cache_enabled() {
            return Err(HFError::CacheNotEnabled);
        }
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let max_workers = params.max_workers.unwrap_or(8);
        let repo_folder = cache::repo_folder_name(&self.repo_path(), Some(self.repo_type));
        let cache_dir = self.hf_client.cache_dir();

        if params.local_files_only == Some(true) {
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

        let mut filenames = self
            .list_filtered_files(&commit_hash, params.allow_patterns.as_ref(), params.ignore_patterns.as_ref())
            .await?;

        let total_files = filenames.len();
        let force = params.force_download == Some(true);

        let mut cached_filenames = Vec::new();
        if !force && params.local_dir.is_none() {
            filenames.retain(|f| {
                if cache::snapshot_path(cache_dir, &repo_folder, &commit_hash, f).exists() {
                    cached_filenames.push(f.clone());
                    false
                } else {
                    true
                }
            });
        }

        let repo_path = self.repo_path();
        let commit_hash_ref = &commit_hash;
        let head_futs = filenames.iter().map(|filename| {
                let url = self
                    .hf_client.download_url(Some(self.repo_type), &repo_path, commit_hash_ref, filename);
                let auth = self.hf_client.auth_headers();
                let filename = filename.clone();
                let repo_folder_ref = &repo_folder;
                async move {
                    let resp = retry::retry(self.hf_client.retry_config(), || {
                        self.hf_client.no_redirect_client().head(&url).headers(auth.clone()).send()
                    })
                    .await?;
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
                    let etag =
                        extract_etag(&resp).ok_or_else(|| HFError::Other(format!("Missing ETag for {filename}")))?;
                    let commit = extract_commit_hash(&resp).unwrap_or_else(|| commit_hash_ref.clone());
                    let xet_hash = extract_xet_hash(&resp);
                    let file_size: u64 = extract_file_size(&resp).unwrap_or_else(|| {
                        tracing::warn!(file = %filename, "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0");
                        0
                    });
                    Ok::<_, HFError>(Some(FileMetadataInfo {
                        filename,
                        etag,
                        commit_hash: commit,
                        xet_hash,
                        file_size,
                    }))
                }
            });

        let file_metas: Vec<FileMetadataInfo> = futures::stream::iter(head_futs)
            .buffer_unordered(max_workers)
            .try_collect::<Vec<Option<FileMetadataInfo>>>()
            .await?
            .into_iter()
            .flatten()
            .collect();

        let total_bytes: u64 = file_metas.iter().map(|m| m.file_size).sum();
        params.progress.emit(DownloadEvent::Start {
            total_files,
            total_bytes,
        });
        if !cached_filenames.is_empty() {
            params.progress.emit(DownloadEvent::Progress {
                files: cached_filenames
                    .iter()
                    .map(|f| FileProgress {
                        filename: f.clone(),
                        bytes_completed: 0,
                        total_bytes: 0,
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
                    local_cached.push(meta.filename);
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
                        .map(|f| FileProgress {
                            filename: f.clone(),
                            bytes_completed: 0,
                            total_bytes: 0,
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
                Some(self.repo_type),
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
            if meta.xet_hash.is_some() {
                xet_metas.push(meta);
            } else {
                non_xet_filenames.push(meta.filename);
            }
        }
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
            Some(self.repo_type),
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

        if !cache::is_commit_hash(revision) {
            cache::write_ref(cache_dir, &repo_folder, revision, &commit_hash).await?;
        }

        params.progress.emit(DownloadEvent::Complete);
        Ok(cache_dir.join(&repo_folder).join("snapshots").join(&commit_hash))
    }
}

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

fn build_download_params(
    _repo_id: &str,
    filenames: &[String],
    _repo_type: Option<RepoType>,
    commit_hash: &str,
    force_download: Option<bool>,
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
            local_files_only: None,
            progress: progress.clone(),
        })
        .collect()
}

async fn download_concurrently(
    api: &HFRepository,
    params: &[DownloadFileParams],
    max_workers: usize,
) -> HFResult<Vec<PathBuf>> {
    futures::stream::iter(params.iter().map(|p| api.download_file_inner(p)))
        .buffer_unordered(max_workers)
        .try_collect()
        .await
}

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

#[bon]
impl HFRepository {
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
    /// Endpoint: `GET {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filename}`.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to download within the repository.
    /// - `local_dir`: local directory to download the file into. When set, the file is saved with its repo path
    ///   structure.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `force_download`: re-download the file even if a cached copy exists.
    /// - `local_files_only`: only return the file if cached locally; never make a network request.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send)]
    pub async fn download_file(
        &self,
        #[builder(into)] filename: String,
        local_dir: Option<PathBuf>,
        #[builder(into)] revision: Option<String>,
        force_download: Option<bool>,
        local_files_only: Option<bool>,
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.download_file_impl(DownloadFileParams {
            filename,
            local_dir,
            revision,
            force_download,
            local_files_only,
            progress,
        })
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
    /// - `range`: byte range to request (HTTP Range header).
    #[builder(finish_fn = send)]
    pub async fn download_file_stream(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
        range: Option<std::ops::Range<u64>>,
    ) -> HFResult<(Option<u64>, Box<dyn Stream<Item = std::result::Result<bytes::Bytes, HFError>> + Send + Unpin>)>
    {
        self.download_file_stream_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
        })
        .await
    }

    /// Download a file (or byte range) into memory and return the contents as [`bytes::Bytes`].
    ///
    /// This is a convenience wrapper around
    /// [`download_file_stream`](Self::download_file_stream) that collects the entire stream into
    /// a single buffer. When `range` is set, only the specified byte range is fetched.
    #[builder(finish_fn = send)]
    pub async fn download_file_to_bytes(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
        range: Option<std::ops::Range<u64>>,
    ) -> HFResult<bytes::Bytes> {
        self.download_file_to_bytes_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
        })
        .await
    }

    /// Download all selected files for a resolved revision.
    ///
    /// When `local_dir` is `None`, files are stored in the HF cache and the returned path is the
    /// cache snapshot directory for the resolved commit. When `local_dir` is `Some`, files are
    /// written directly under that directory.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `allow_patterns`: glob patterns of files to include.
    /// - `ignore_patterns`: glob patterns of files to exclude.
    /// - `local_dir`: local directory to download into.
    /// - `force_download`: re-download all files even if cached.
    /// - `local_files_only`: resolve only from the local cache.
    /// - `max_workers`: maximum concurrent file downloads (default 8).
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send)]
    pub async fn snapshot_download(
        &self,
        #[builder(into)] revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        local_dir: Option<PathBuf>,
        force_download: Option<bool>,
        local_files_only: Option<bool>,
        max_workers: Option<usize>,
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.snapshot_download_impl(SnapshotDownloadParams {
            revision,
            allow_patterns,
            ignore_patterns,
            local_dir,
            force_download,
            local_files_only,
            max_workers,
            progress,
        })
        .await
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync {
    /// Blocking counterpart of [`HFRepository::download_file`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send)]
    pub fn download_file(
        &self,
        #[builder(into)] filename: String,
        local_dir: Option<PathBuf>,
        #[builder(into)] revision: Option<String>,
        force_download: Option<bool>,
        local_files_only: Option<bool>,
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .download_file()
                .filename(filename)
                .maybe_local_dir(local_dir)
                .maybe_revision(revision)
                .maybe_force_download(force_download)
                .maybe_local_files_only(local_files_only)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::download_file_to_bytes`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send)]
    pub fn download_file_to_bytes(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
        range: Option<std::ops::Range<u64>>,
    ) -> HFResult<bytes::Bytes> {
        self.runtime.block_on(
            self.inner
                .download_file_to_bytes()
                .filename(filename)
                .maybe_revision(revision)
                .maybe_range(range)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::snapshot_download`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send)]
    pub fn snapshot_download(
        &self,
        #[builder(into)] revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        local_dir: Option<PathBuf>,
        force_download: Option<bool>,
        local_files_only: Option<bool>,
        max_workers: Option<usize>,
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .snapshot_download()
                .maybe_revision(revision)
                .maybe_allow_patterns(allow_patterns)
                .maybe_ignore_patterns(ignore_patterns)
                .maybe_local_dir(local_dir)
                .maybe_force_download(force_download)
                .maybe_local_files_only(local_files_only)
                .maybe_max_workers(max_workers)
                .maybe_progress(progress)
                .send(),
        )
    }
}
