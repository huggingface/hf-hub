//! Streaming xet upload session.
//!
//! Streams individual files into a repository's xet content-addressable
//! storage *as they are produced*, returning a per-file
//! `(sha256_oid, size)` descriptor that can be referenced from a later
//! [`HFRepository::create_commit`](super::HFRepository::create_commit) call
//! via [`AddSource::Lfs`](super::AddSource::Lfs). This lets callers produce a
//! single commit whose total payload exceeds local RAM and disk.
//!
//! The usual `upload_file` / `upload_folder` / `create_commit` paths buffer
//! every file's bytes in memory (`AddSource::Bytes`) or on disk
//! (`AddSource::File`) until the commit is built. For workloads that
//! generate many small or many large files one at a time — such as rendered
//! tile pyramids, sharded training outputs, or any kind of streaming export —
//! that buffering becomes the bottleneck. Use a streaming session instead:
//! upload each file, drop its bytes (or remove its temp file), and finalize
//! the commit at the end with only `(path, oid, size)` triples in memory.
//!
//! See [`HFRepository::open_xet_upload_session`](super::HFRepository::open_xet_upload_session)
//! for usage.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use xet::xet_session::{Sha256Policy, UniqueID, XetFileUpload};

use super::{HFRepository, RepoType};
use crate::constants;
use crate::error::{HFError, HFResult, NotFoundContext, XetOperation};
use crate::xet::{new_xet_upload_commit, repo_xet_token_url};

/// Per-file result of a completed streaming xet upload.
///
/// Pass `sha256_oid` and `size` into
/// [`AddSource::Lfs`](super::AddSource::Lfs) (or
/// [`CommitOperation::add_lfs`](super::CommitOperation::add_lfs)) when
/// building the final commit.
#[derive(Debug, Clone)]
pub struct UploadedXetFile {
    /// Lowercase hex SHA-256 of the file content, as known to xet CAS.
    pub sha256_oid: String,
    /// Total file size in bytes.
    pub size: u64,
    /// Xet content-addressable (Merkle) hash. Useful when the same blob will
    /// be referenced from a bucket (see
    /// [`HFBucket::open_xet_upload_session`](crate::buckets::HFBucket::open_xet_upload_session)),
    /// which addresses by xet hash rather than SHA-256.
    pub xet_hash: String,
}

/// Open streaming xet-upload session for a single repository revision.
///
/// Construct with
/// [`HFRepository::open_xet_upload_session`](super::HFRepository::open_xet_upload_session),
/// then call [`upload_bytes`](Self::upload_bytes) / [`upload_file`](Self::upload_file)
/// for each piece of content, and finish with [`finish`](Self::finish) to seal
/// the CAS uploads and recover per-path descriptors.
pub struct XetUploadSession {
    commit: xet::xet_session::XetUploadCommit,
    /// `(path_in_repo, task_id)` pairs, populated as the caller queues
    /// uploads; consumed at finish() time to map xet task results back to
    /// user-supplied paths.
    path_to_task: Mutex<Vec<(String, UniqueID)>>,
}

impl XetUploadSession {
    /// Queue an in-memory file for streaming upload.
    ///
    /// xet starts uploading the bytes to CAS in the background. The future
    /// resolves once xet has accepted the bytes — at that point the caller may
    /// drop their own copy. Use [`finish`](Self::finish) to wait for *all*
    /// queued uploads to settle.
    pub async fn upload_bytes(&self, path_in_repo: impl Into<String>, bytes: Vec<u8>) -> HFResult<()> {
        let path_in_repo = path_in_repo.into();
        tracing::debug!(path = path_in_repo.as_str(), len = bytes.len(), "queuing xet upload (bytes)");
        let handle = self
            .commit
            .upload_bytes(bytes, Sha256Policy::Compute, Some(path_in_repo.clone()))
            .await
            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        self.record_handle(path_in_repo, &handle);
        Ok(())
    }

    /// Queue a local file for streaming upload.
    ///
    /// xet reads the file lazily during the background transfer, so the file
    /// must exist (unchanged) until [`finish`](Self::finish) returns.
    pub async fn upload_file(&self, path_in_repo: impl Into<String>, source: impl Into<PathBuf>) -> HFResult<()> {
        let path_in_repo = path_in_repo.into();
        let source = source.into();
        tracing::debug!(path = path_in_repo.as_str(), source = ?source, "queuing xet upload (file)");
        let handle = self
            .commit
            .upload_from_path(source, Sha256Policy::Compute)
            .await
            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        self.record_handle(path_in_repo, &handle);
        Ok(())
    }

    fn record_handle(&self, path_in_repo: String, handle: &XetFileUpload) {
        self.path_to_task
            .lock()
            .expect("xet upload session mutex poisoned")
            .push((path_in_repo, handle.task_id()));
    }

    /// Finalize the session: wait for every queued upload to complete, seal
    /// the CAS state, and return a map of path → blob descriptor.
    ///
    /// On success, the returned descriptors can be passed to
    /// [`AddSource::Lfs`](super::AddSource::Lfs) in a subsequent
    /// [`create_commit`](super::HFRepository::create_commit) call.
    pub async fn finish(self) -> HFResult<HashMap<String, UploadedXetFile>> {
        let results = self
            .commit
            .commit()
            .await
            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        tracing::info!(file_count = results.uploads.len(), "xet streaming upload session committed");

        let path_to_task = self
            .path_to_task
            .into_inner()
            .expect("xet upload session mutex poisoned");

        let mut out: HashMap<String, UploadedXetFile> = HashMap::with_capacity(path_to_task.len());
        for (path, task_id) in path_to_task {
            let metadata = results
                .uploads
                .get(&task_id)
                .ok_or_else(|| HFError::Other(format!("xet upload result missing for {path:?}")))?;
            let sha256_oid = metadata
                .xet_info
                .sha256()
                .ok_or_else(|| {
                    HFError::Other(format!(
                        "xet upload for {path:?} did not produce a SHA-256; was Sha256Policy::Compute used?"
                    ))
                })?
                .to_string();
            let size = metadata
                .xet_info
                .file_size()
                .ok_or_else(|| HFError::Other(format!("xet upload for {path:?} did not report a file size")))?;
            out.insert(
                path,
                UploadedXetFile {
                    sha256_oid,
                    size,
                    xet_hash: metadata.xet_info.hash().to_string(),
                },
            );
        }

        Ok(out)
    }
}

/// Synchronous/blocking counterpart to [`XetUploadSession`].
///
/// Constructed via [`HFRepositorySync::open_xet_upload_session`](crate::HFRepositorySync::open_xet_upload_session).
#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub struct XetUploadSessionSync {
    inner: XetUploadSession,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "blocking")]
impl XetUploadSessionSync {
    /// Blocking counterpart of [`XetUploadSession::upload_bytes`].
    pub fn upload_bytes(&self, path_in_repo: impl Into<String>, bytes: Vec<u8>) -> HFResult<()> {
        self.runtime.block_on(self.inner.upload_bytes(path_in_repo, bytes))
    }

    /// Blocking counterpart of [`XetUploadSession::upload_file`].
    pub fn upload_file(&self, path_in_repo: impl Into<String>, source: impl Into<PathBuf>) -> HFResult<()> {
        self.runtime.block_on(self.inner.upload_file(path_in_repo, source))
    }

    /// Blocking counterpart of [`XetUploadSession::finish`].
    pub fn finish(self) -> HFResult<HashMap<String, UploadedXetFile>> {
        self.runtime.block_on(self.inner.finish())
    }
}

#[cfg(feature = "blocking")]
impl<T: RepoType> crate::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::open_xet_upload_session`].
    pub fn open_xet_upload_session(&self, revision: Option<String>) -> HFResult<XetUploadSessionSync> {
        let inner = self.runtime.block_on(self.inner.open_xet_upload_session(revision))?;
        Ok(XetUploadSessionSync {
            inner,
            runtime: self.runtime.clone(),
        })
    }
}

impl<T: RepoType> HFRepository<T> {
    /// Open a streaming xet-upload session for the given revision.
    ///
    /// Each [`upload_bytes`](XetUploadSession::upload_bytes) /
    /// [`upload_file`](XetUploadSession::upload_file) call uploads its content
    /// to xet content-addressable storage immediately. After all content is
    /// uploaded, call [`finish`](XetUploadSession::finish) to seal the CAS
    /// uploads and recover per-path `(sha256_oid, size, xet_hash)`
    /// descriptors. Those descriptors can then be fed into
    /// [`AddSource::Lfs`](super::AddSource::Lfs) for a final
    /// [`create_commit`](Self::create_commit) call.
    ///
    /// Use this when total commit payload exceeds local RAM or disk: the
    /// caller never has to hold all bytes (or all temp files) at once. Files
    /// smaller than the Hub's LFS threshold should still go through
    /// [`upload_file`](Self::upload_file) or
    /// [`create_commit`](Self::create_commit) — this session is xet-only and
    /// unconditionally produces `lfsFile` commit entries.
    pub async fn open_xet_upload_session(&self, revision: Option<String>) -> HFResult<XetUploadSession> {
        let revision = revision.unwrap_or_else(|| constants::DEFAULT_REVISION.to_string());
        let repo_path = self.repo_path();
        let api_segment = T::default().plural();
        let token_url = repo_xet_token_url(&self.hf_client, "write", &repo_path, api_segment, &revision);
        let commit = new_xet_upload_commit(&self.hf_client, token_url, &repo_path, NotFoundContext::Repo, "repo").await?;
        Ok(XetUploadSession {
            commit,
            path_to_task: Mutex::new(Vec::new()),
        })
    }
}
