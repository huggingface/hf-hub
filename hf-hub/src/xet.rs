//! Xet high-performance transfer support.
//!
//! This module is only compiled when the "xet" feature is enabled.
//! When xet headers are detected during download/upload but the feature
//! is not enabled, HFError::XetNotEnabled is returned at the call site.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;
use xet::xet_session::{Sha256Policy, XetFileDownload, XetFileInfo, XetFileMetadata, XetFileUpload};

use crate::client::HFClient;
use crate::constants;
use crate::error::{HFError, Result};
use crate::repository::HFRepository;
use crate::types::progress::{
    self, DownloadEvent, FileProgress, FileStatus, Progress, ProgressEvent, UploadEvent, UploadPhase,
};
use crate::types::{AddSource, GetXetTokenParams, RepoType};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XetTokenResponse {
    access_token: String,
    exp: u64,
    cas_url: String,
}

#[derive(Default)]
pub(crate) struct XetState {
    pub(crate) session: Option<xet::xet_session::XetSession>,
    pub(crate) generation: u64,
}

pub struct XetConnectionInfo {
    pub endpoint: String,
    pub access_token: String,
    pub expiration_unix_epoch: u64,
}

async fn fetch_xet_connection_info(
    api: &HFClient,
    token_url: &str,
    not_found_id: Option<&str>,
    not_found_ctx: crate::error::NotFoundContext,
) -> Result<XetConnectionInfo> {
    let response = api.http_client().get(token_url).headers(api.auth_headers()).send().await?;

    let response = api.check_response(response, not_found_id, not_found_ctx).await?;

    let token_resp: XetTokenResponse = response.json().await?;
    Ok(XetConnectionInfo {
        endpoint: token_resp.cas_url,
        access_token: token_resp.access_token,
        expiration_unix_epoch: token_resp.exp,
    })
}

fn repo_xet_token_url(
    api: &HFClient,
    token_type: &str,
    repo_id: &str,
    repo_type: Option<RepoType>,
    revision: &str,
) -> String {
    let segment = constants::repo_type_api_segment(repo_type);
    format!("{}/api/{}/{}/xet-{}-token/{}", api.endpoint(), segment, repo_id, token_type, revision)
}

pub(crate) fn bucket_xet_token_url(api: &HFClient, token_type: &str, bucket_id: &str) -> String {
    format!("{}/api/buckets/{}/xet-{}-token", api.endpoint(), bucket_id, token_type)
}

/// Returns `true` if the error indicates the XetSession is permanently
/// poisoned and must be replaced before retrying.
#[cfg(test)]
fn is_session_poisoned(err: &xet::error::XetError) -> bool {
    use xet::error::XetError;
    matches!(
        err,
        XetError::UserCancelled(_)
            | XetError::AlreadyCompleted
            | XetError::PreviousTaskError(_)
            | XetError::KeyboardInterrupt
    )
}

pub(crate) struct TrackedDownload {
    pub handle: XetFileDownload,
    pub filename: String,
    pub file_size: u64,
    pub complete_emitted: AtomicBool,
}

fn emit_remaining_completes(progress: &Progress, tracked: &[TrackedDownload]) {
    for t in tracked {
        if !t.complete_emitted.swap(true, Ordering::Relaxed) {
            progress::emit(
                progress,
                ProgressEvent::Download(DownloadEvent::Progress {
                    files: vec![FileProgress {
                        filename: t.filename.clone(),
                        bytes_completed: t.file_size,
                        total_bytes: t.file_size,
                        status: FileStatus::Complete,
                    }],
                }),
            );
        }
    }
}

fn spawn_download_progress_poller(
    progress: &Progress,
    group: &xet::xet_session::XetFileDownloadGroup,
    tracked: Arc<Vec<TrackedDownload>>,
) -> Option<tokio::task::JoinHandle<()>> {
    let handler = progress.as_ref()?.clone();
    let group = group.clone();
    Some(tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let report = group.progress();
            handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed: report.total_bytes_completed,
                total_bytes: report.total_bytes,
                bytes_per_sec: report.total_bytes_completion_rate,
            }));

            let mut completed = Vec::new();
            let mut in_progress = Vec::new();

            for t in tracked.iter() {
                if t.complete_emitted.load(Ordering::Relaxed) {
                    continue;
                }
                if t.handle.result().is_some() {
                    if !t.complete_emitted.swap(true, Ordering::Relaxed) {
                        completed.push(FileProgress {
                            filename: t.filename.clone(),
                            bytes_completed: t.file_size,
                            total_bytes: t.file_size,
                            status: FileStatus::Complete,
                        });
                    }
                    continue;
                }
                if let Some(item) = t.handle.progress() {
                    let total = item.total_bytes.max(t.file_size);
                    if item.bytes_completed >= total && total > 0 {
                        if !t.complete_emitted.swap(true, Ordering::Relaxed) {
                            completed.push(FileProgress {
                                filename: t.filename.clone(),
                                bytes_completed: total,
                                total_bytes: total,
                                status: FileStatus::Complete,
                            });
                        }
                    } else {
                        let status = if item.bytes_completed > 0 {
                            FileStatus::InProgress
                        } else {
                            FileStatus::Started
                        };
                        in_progress.push(FileProgress {
                            filename: t.filename.clone(),
                            bytes_completed: item.bytes_completed,
                            total_bytes: total,
                            status,
                        });
                    }
                }
            }

            if !completed.is_empty() {
                handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress { files: completed }));
            }
            if !in_progress.is_empty() {
                handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress { files: in_progress }));
            }
        }
    }))
}

pub(crate) struct XetBatchFile {
    pub hash: String,
    pub file_size: u64,
    pub path: PathBuf,
    pub filename: String,
}

impl HFRepository {
    pub(crate) async fn xet_download_to_local_dir(
        &self,
        revision: &str,
        filename: &str,
        local_dir: &std::path::Path,
        head_response: &reqwest::Response,
        progress: &Progress,
    ) -> Result<PathBuf> {
        let repo_path = self.repo_path();
        let repo_type = Some(self.repo_type);
        let file_hash = crate::api::files::extract_xet_hash(head_response)
            .ok_or_else(|| HFError::Other("Missing X-Xet-Hash header".to_string()))?;

        let file_size: u64 = crate::api::files::extract_file_size(head_response).unwrap_or(0);

        let token_url = repo_xet_token_url(&self.hf_client, "read", &repo_path, repo_type, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        tokio::fs::create_dir_all(local_dir).await?;
        let dest_path = local_dir.join(filename);
        if let Some(parent) = dest_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;

        let file_info = XetFileInfo::new(file_hash, file_size);

        let handle = group
            .download_file_to_path(file_info, dest_path.clone())
            .await
            .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;

        let tracked = Arc::new(vec![TrackedDownload {
            handle,
            filename: filename.to_string(),
            file_size,
            complete_emitted: AtomicBool::new(false),
        }]);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;
        emit_remaining_completes(progress, &tracked);

        Ok(dest_path)
    }

    pub(crate) async fn xet_download_to_blob(
        &self,
        revision: &str,
        filename: &str,
        file_hash: &str,
        file_size: u64,
        path: &std::path::Path,
        progress: &Progress,
    ) -> Result<()> {
        let repo_path = self.repo_path();
        let repo_type = Some(self.repo_type);
        let token_url = repo_xet_token_url(&self.hf_client, "read", &repo_path, repo_type, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let incomplete_path = PathBuf::from(format!("{}.incomplete", path.display()));

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;

        let file_info = XetFileInfo::new(file_hash.to_string(), file_size);

        let handle = group
            .download_file_to_path(file_info, incomplete_path.clone())
            .await
            .map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;

        let tracked = Arc::new(vec![TrackedDownload {
            handle,
            filename: filename.to_string(),
            file_size,
            complete_emitted: AtomicBool::new(false),
        }]);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::Other(format!("Xet download failed: {e}")))?;
        emit_remaining_completes(progress, &tracked);

        tokio::fs::rename(&incomplete_path, path).await?;
        Ok(())
    }

    pub(crate) async fn xet_download_batch(
        &self,
        revision: &str,
        files: &[XetBatchFile],
        progress: &Progress,
    ) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        let repo_path = self.repo_path();
        let repo_type = Some(self.repo_type);
        let token_url = repo_xet_token_url(&self.hf_client, "read", &repo_path, repo_type, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::Other(format!("Xet batch download failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet batch download failed: {e}")))?;

        let mut tracked_vec = Vec::with_capacity(files.len());
        let mut incomplete_paths = Vec::with_capacity(files.len());
        for file in files {
            if let Some(parent) = file.path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            let incomplete = PathBuf::from(format!("{}.incomplete", file.path.display()));

            let file_info = XetFileInfo::new(file.hash.clone(), file.file_size);

            let handle = group
                .download_file_to_path(file_info, incomplete.clone())
                .await
                .map_err(|e| HFError::Other(format!("Xet batch download failed: {e}")))?;

            tracked_vec.push(TrackedDownload {
                handle,
                filename: file.filename.clone(),
                file_size: file.file_size,
                complete_emitted: AtomicBool::new(false),
            });
            incomplete_paths.push((incomplete, file.path.clone()));
        }

        let tracked = Arc::new(tracked_vec);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::Other(format!("Xet batch download failed: {e}")))?;
        emit_remaining_completes(progress, &tracked);

        for (incomplete, final_path) in &incomplete_paths {
            tokio::fs::rename(incomplete, final_path).await?;
        }

        Ok(())
    }

    /// Download a file (or byte range) via xet and return a byte stream.
    ///
    /// Uses `XetDownloadStreamGroup` which supports `Option<Range<u64>>` for partial downloads.
    pub(crate) async fn xet_download_stream(
        &self,
        revision: &str,
        file_hash: &str,
        file_size: u64,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<impl futures::Stream<Item = Result<bytes::Bytes>> + use<>> {
        let repo_path = self.repo_path();
        let repo_type = Some(self.repo_type);
        let token_url = repo_xet_token_url(&self.hf_client, "read", &repo_path, repo_type, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_download_stream_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_download_stream_group()
                    .map_err(|e| HFError::Other(format!("Xet stream download failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet stream download failed: {e}")))?;

        let file_info = XetFileInfo::new(file_hash.to_string(), file_size);

        let mut stream = group
            .download_stream(file_info, range)
            .await
            .map_err(|e| HFError::Other(format!("Xet stream download failed: {e}")))?;

        stream.start();

        Ok(futures::stream::unfold(stream, |mut stream| async move {
            match stream.next().await {
                Ok(Some(bytes)) => Some((Ok(bytes), stream)),
                Ok(None) => None,
                Err(e) => Some((Err(HFError::Other(format!("Xet stream read failed: {e}"))), stream)),
            }
        }))
    }

    /// Upload files using the xet protocol.
    /// Fetches a write token and uses xet-session's UploadCommit.
    /// Returns the XetFileInfo (hash + size) for each uploaded file.
    pub(crate) async fn xet_upload(
        &self,
        files: &[(String, AddSource)],
        revision: &str,
        progress: &Progress,
    ) -> Result<Vec<XetFileInfo>> {
        let repo_path = self.repo_path();
        let repo_type = Some(self.repo_type);
        tracing::info!(repo = repo_path.as_str(), "fetching xet write token");
        let token_url = repo_xet_token_url(&self.hf_client, "write", &repo_path, repo_type, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;
        tracing::info!(endpoint = conn.endpoint.as_str(), "xet write token obtained, building session");

        tracing::info!("building xet upload commit");
        let (session, generation) = self.hf_client.xet_session()?;
        let commit = match session.new_upload_commit() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_upload_commit()
                    .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?;
        tracing::info!("xet upload commit built, queuing file uploads");

        let mut task_ids_in_order = Vec::with_capacity(files.len());
        let mut handles: Vec<XetFileUpload> = Vec::with_capacity(files.len());
        let mut item_name_to_repo_path: HashMap<String, String> = HashMap::with_capacity(files.len());

        for (path_in_repo, source) in files {
            tracing::info!(path = path_in_repo.as_str(), "queuing xet upload");
            let handle = match source {
                AddSource::File(path) => {
                    // Mimic xet-core's `std::path::absolute()` logic to derive the
                    // item_name that will appear in ItemProgressReport.
                    // See: xet-data upload_commit.rs XetUploadCommitInner::upload_from_path
                    if let Ok(abs) = std::path::absolute(path) {
                        if let Some(s) = abs.to_str() {
                            item_name_to_repo_path.insert(s.to_owned(), path_in_repo.clone());
                        } else {
                            tracing::warn!(path = ?abs, "non-UTF-8 path; per-file progress unavailable");
                        }
                    }
                    commit
                        .upload_from_path(path.clone(), Sha256Policy::Compute)
                        .await
                        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
                },
                AddSource::Bytes(bytes) => {
                    item_name_to_repo_path.insert(path_in_repo.clone(), path_in_repo.clone());
                    commit
                        .upload_bytes(bytes.clone(), Sha256Policy::Compute, Some(path_in_repo.clone()))
                        .await
                        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
                },
            };
            task_ids_in_order.push(handle.task_id());
            handles.push(handle);
        }

        tracing::info!(file_count = files.len(), "committing xet uploads");
        let shared_handles: Arc<Vec<XetFileUpload>> = Arc::new(handles);
        let shared_name_map: Arc<HashMap<String, String>> = Arc::new(item_name_to_repo_path);

        let poll_handle = progress.as_ref().map(|handler| {
            let handler = handler.clone();
            let commit = commit.clone();
            let poll_handles = Arc::clone(&shared_handles);
            let poll_name_map = Arc::clone(&shared_name_map);
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let report = commit.progress();
                    let file_progress: Vec<FileProgress> = poll_handles
                        .iter()
                        .filter_map(|h| {
                            let item = h.progress()?;
                            let repo_path = poll_name_map.get(&item.item_name)?;
                            let status = if item.bytes_completed >= item.total_bytes && item.total_bytes > 0 {
                                FileStatus::Complete
                            } else if item.bytes_completed > 0 {
                                FileStatus::InProgress
                            } else {
                                FileStatus::Started
                            };
                            Some(FileProgress {
                                filename: repo_path.clone(),
                                bytes_completed: item.bytes_completed,
                                total_bytes: item.total_bytes,
                                status,
                            })
                        })
                        .collect();
                    handler.on_progress(&ProgressEvent::Upload(UploadEvent::Progress {
                        phase: UploadPhase::Uploading,
                        bytes_completed: report.total_bytes_completed,
                        total_bytes: report.total_bytes,
                        bytes_per_sec: report.total_bytes_completion_rate,
                        transfer_bytes_completed: report.total_transfer_bytes_completed,
                        transfer_bytes: report.total_transfer_bytes,
                        transfer_bytes_per_sec: report.total_transfer_bytes_completion_rate,
                        files: file_progress,
                    }));
                }
            })
        });
        let results = commit
            .commit()
            .await
            .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?;
        if let Some(h) = poll_handle {
            h.abort();
        }
        tracing::info!("xet upload commit complete");

        let final_files: Vec<FileProgress> = files
            .iter()
            .map(|(path_in_repo, _)| FileProgress {
                filename: path_in_repo.clone(),
                bytes_completed: 0,
                total_bytes: 0,
                status: FileStatus::Complete,
            })
            .collect();

        progress::emit(
            progress,
            ProgressEvent::Upload(UploadEvent::Progress {
                phase: UploadPhase::Uploading,
                bytes_completed: results.progress.total_bytes_completed,
                total_bytes: results.progress.total_bytes,
                bytes_per_sec: results.progress.total_bytes_completion_rate,
                transfer_bytes_completed: results.progress.total_transfer_bytes_completed,
                transfer_bytes: results.progress.total_transfer_bytes,
                transfer_bytes_per_sec: results.progress.total_transfer_bytes_completion_rate,
                files: final_files,
            }),
        );

        let mut xet_file_infos = Vec::with_capacity(files.len());
        for task_id in &task_ids_in_order {
            let metadata: &XetFileMetadata = results
                .uploads
                .get(task_id)
                .ok_or_else(|| HFError::Other("Missing xet upload result for task".to_string()))?;
            xet_file_infos.push(metadata.xet_info.clone());
        }

        Ok(xet_file_infos)
    }
}

#[cfg(feature = "buckets")]
impl crate::bucket::HFBucket {
    pub(crate) async fn xet_upload(
        &self,
        files: &[(String, AddSource)],
        progress: &Progress,
    ) -> Result<Vec<XetFileInfo>> {
        let bucket_id = self.bucket_id();
        tracing::info!(bucket = bucket_id.as_str(), "fetching xet write token");
        let token_url = bucket_xet_token_url(&self.hf_client, "write", &bucket_id);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&bucket_id),
            crate::error::NotFoundContext::Bucket,
        )
        .await?;
        tracing::info!(endpoint = conn.endpoint.as_str(), "xet write token obtained, building session");

        tracing::info!("building xet upload commit");
        let (session, generation) = self.hf_client.xet_session()?;
        let commit = match session.new_upload_commit() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_upload_commit()
                    .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?;
        tracing::info!("xet upload commit built, queuing file uploads");

        let mut task_ids_in_order = Vec::with_capacity(files.len());
        let mut handles: Vec<XetFileUpload> = Vec::with_capacity(files.len());
        let mut item_name_to_bucket_path: HashMap<String, String> = HashMap::with_capacity(files.len());

        for (path_in_bucket, source) in files {
            tracing::info!(path = path_in_bucket.as_str(), "queuing xet upload");
            let handle = match source {
                AddSource::File(path) => {
                    if let Ok(abs) = std::path::absolute(path)
                        && let Some(s) = abs.to_str()
                    {
                        item_name_to_bucket_path.insert(s.to_owned(), path_in_bucket.clone());
                    }
                    commit
                        .upload_from_path(path.clone(), Sha256Policy::Compute)
                        .await
                        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
                },
                AddSource::Bytes(bytes) => {
                    item_name_to_bucket_path.insert(path_in_bucket.clone(), path_in_bucket.clone());
                    commit
                        .upload_bytes(bytes.clone(), Sha256Policy::Compute, Some(path_in_bucket.clone()))
                        .await
                        .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?
                },
            };
            task_ids_in_order.push(handle.task_id());
            handles.push(handle);
        }

        tracing::info!(file_count = files.len(), "committing xet uploads");
        let shared_handles: Arc<Vec<XetFileUpload>> = Arc::new(handles);
        let shared_name_map: Arc<HashMap<String, String>> = Arc::new(item_name_to_bucket_path);

        let poll_handle = progress.as_ref().map(|handler| {
            let handler = handler.clone();
            let commit = commit.clone();
            let poll_handles = Arc::clone(&shared_handles);
            let poll_name_map = Arc::clone(&shared_name_map);
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    let report = commit.progress();
                    let file_progress: Vec<FileProgress> = poll_handles
                        .iter()
                        .filter_map(|h| {
                            let item = h.progress()?;
                            let bucket_path = poll_name_map.get(&item.item_name)?;
                            let status = if item.bytes_completed >= item.total_bytes && item.total_bytes > 0 {
                                FileStatus::Complete
                            } else if item.bytes_completed > 0 {
                                FileStatus::InProgress
                            } else {
                                FileStatus::Started
                            };
                            Some(FileProgress {
                                filename: bucket_path.clone(),
                                bytes_completed: item.bytes_completed,
                                total_bytes: item.total_bytes,
                                status,
                            })
                        })
                        .collect();
                    handler.on_progress(&ProgressEvent::Upload(UploadEvent::Progress {
                        phase: UploadPhase::Uploading,
                        bytes_completed: report.total_bytes_completed,
                        total_bytes: report.total_bytes,
                        bytes_per_sec: report.total_bytes_completion_rate,
                        transfer_bytes_completed: report.total_transfer_bytes_completed,
                        transfer_bytes: report.total_transfer_bytes,
                        transfer_bytes_per_sec: report.total_transfer_bytes_completion_rate,
                        files: file_progress,
                    }));
                }
            })
        });
        let results = commit
            .commit()
            .await
            .map_err(|e| HFError::Other(format!("Xet upload failed: {e}")))?;
        if let Some(h) = poll_handle {
            h.abort();
        }
        tracing::info!("xet upload commit complete");

        let final_files: Vec<FileProgress> = files
            .iter()
            .map(|(path_in_bucket, _)| FileProgress {
                filename: path_in_bucket.clone(),
                bytes_completed: 0,
                total_bytes: 0,
                status: FileStatus::Complete,
            })
            .collect();

        progress::emit(
            progress,
            ProgressEvent::Upload(UploadEvent::Progress {
                phase: UploadPhase::Uploading,
                bytes_completed: results.progress.total_bytes_completed,
                total_bytes: results.progress.total_bytes,
                bytes_per_sec: results.progress.total_bytes_completion_rate,
                transfer_bytes_completed: results.progress.total_transfer_bytes_completed,
                transfer_bytes: results.progress.total_transfer_bytes,
                transfer_bytes_per_sec: results.progress.total_transfer_bytes_completion_rate,
                files: final_files,
            }),
        );

        let mut xet_file_infos = Vec::with_capacity(files.len());
        for task_id in &task_ids_in_order {
            let metadata: &XetFileMetadata = results
                .uploads
                .get(task_id)
                .ok_or_else(|| HFError::Other("Missing xet upload result for task".to_string()))?;
            xet_file_infos.push(metadata.xet_info.clone());
        }

        Ok(xet_file_infos)
    }

    pub(crate) async fn xet_download_batch(&self, files: &[XetBatchFile], progress: &Progress) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }

        let bucket_id = self.bucket_id();
        tracing::info!(bucket = bucket_id.as_str(), file_count = files.len(), "fetching xet read token");
        let token_url = bucket_xet_token_url(&self.hf_client, "read", &bucket_id);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&bucket_id),
            crate::error::NotFoundContext::Bucket,
        )
        .await?;
        tracing::info!(endpoint = conn.endpoint.as_str(), "xet download session ready, queuing files");

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::Other(format!("Xet bucket batch download failed: {e}")))?
            },
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::Other(format!("Xet bucket batch download failed: {e}")))?;

        let mut tracked_vec = Vec::with_capacity(files.len());
        let mut incomplete_paths = Vec::with_capacity(files.len());
        for file in files {
            if let Some(parent) = file.path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            let incomplete = PathBuf::from(format!("{}.incomplete", file.path.display()));

            let file_info = XetFileInfo::new(file.hash.clone(), file.file_size);

            let handle = group
                .download_file_to_path(file_info, incomplete.clone())
                .await
                .map_err(|e| HFError::Other(format!("Xet bucket batch download failed: {e}")))?;

            tracked_vec.push(TrackedDownload {
                handle,
                filename: file.filename.clone(),
                file_size: file.file_size,
                complete_emitted: AtomicBool::new(false),
            });
            incomplete_paths.push((incomplete, file.path.clone()));
        }

        let tracked = Arc::new(tracked_vec);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::Other(format!("Xet bucket batch download failed: {e}")))?;
        emit_remaining_completes(progress, &tracked);

        for (incomplete, final_path) in &incomplete_paths {
            tokio::fs::rename(incomplete, final_path).await?;
        }

        Ok(())
    }
}

impl HFClient {
    /// Fetch a Xet connection token (read or write) for a repository.
    /// Endpoint: GET /api/{repo_type}s/{repo_id}/xet-{read|write}-token/{revision}
    pub async fn get_xet_token(&self, params: &GetXetTokenParams) -> Result<XetConnectionInfo> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let token_url =
            repo_xet_token_url(self, params.token_type.as_str(), &params.repo_id, params.repo_type, revision);
        fetch_xet_connection_info(self, &token_url, Some(&params.repo_id), crate::error::NotFoundContext::Repo).await
    }
}

#[cfg(test)]
mod tests {
    use xet::error::XetError;

    use super::*;

    #[test]
    fn test_session_poisoned_positive() {
        assert!(is_session_poisoned(&XetError::UserCancelled("test".into())));
        assert!(is_session_poisoned(&XetError::AlreadyCompleted));
        assert!(is_session_poisoned(&XetError::PreviousTaskError("err".into())));
        assert!(is_session_poisoned(&XetError::KeyboardInterrupt));
    }

    #[test]
    fn test_session_poisoned_negative() {
        let non_poisoned = [
            XetError::Network("timeout".into()),
            XetError::Authentication("bad token".into()),
            XetError::Io("disk full".into()),
            XetError::Internal("bug".into()),
            XetError::Timeout("slow".into()),
            XetError::NotFound("missing".into()),
            XetError::DataIntegrity("corrupt".into()),
            XetError::Configuration("bad config".into()),
            XetError::Cancelled("cancelled".into()),
            XetError::WrongRuntimeMode("wrong mode".into()),
            XetError::TaskError("task failed".into()),
        ];
        for err in &non_poisoned {
            assert!(!is_session_poisoned(err), "{err:?} should NOT be classified as poisoned");
        }
    }

    #[test]
    fn test_xet_error_message_preserved_in_hferror() {
        let xet_err = XetError::Network("connection reset by peer".into());
        let hf_err = HFError::Other(format!("Xet download failed: {xet_err}"));
        let msg = hf_err.to_string();
        assert!(msg.contains("Xet download failed"), "missing prefix: {msg}");
        assert!(msg.contains("connection reset by peer"), "missing original message: {msg}");
    }
}
