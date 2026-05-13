//! Streaming xet upload session for buckets.
//!
//! Streams individual files into a bucket's xet content-addressable storage
//! *as they are produced*, returning per-file descriptors that can be used to
//! register the files in a subsequent
//! [`HFBucket::batch_operations`](super::HFBucket::batch_operations) call.
//! This lets callers populate a bucket whose total payload exceeds local RAM
//! and disk.
//!
//! The usual [`upload_files`](super::HFBucket::upload_files) / [`sync`](super::sync)
//! paths require every file to exist on local disk (`BucketUpload { local, ... }`).
//! For workloads that generate output incrementally — rendered tile pyramids,
//! sharded training outputs, anything streaming — staging the whole pyramid
//! locally becomes the bottleneck. Use a streaming session instead: upload
//! each tile, drop its bytes (or remove its temp file), and register the
//! batch at the end with only `(path, hash, size)` triples in memory.
//!
//! See [`HFBucket::open_xet_upload_session`](super::HFBucket::open_xet_upload_session)
//! for usage.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use xet::xet_session::{Sha256Policy, UniqueID, XetFileUpload};

use super::{BucketAddFile, HFBucket};
use crate::error::{HFError, HFResult, NotFoundContext, XetOperation};
use crate::repository::UploadedXetFile;
use crate::xet::{bucket_xet_token_url, new_xet_upload_commit};

/// Open streaming xet-upload session for a single bucket.
///
/// Construct with
/// [`HFBucket::open_xet_upload_session`](super::HFBucket::open_xet_upload_session),
/// then call [`upload_bytes`](Self::upload_bytes) / [`upload_file`](Self::upload_file)
/// for each piece of content, and finish with [`finish`](Self::finish) to
/// seal the CAS uploads and recover per-path descriptors. Use those to call
/// [`HFBucket::batch_operations`](super::HFBucket::batch_operations) with
/// [`BucketAddFile`] entries (or use
/// [`finish_and_register`](Self::finish_and_register) to do both in one step).
pub struct BucketXetUploadSession {
    bucket: HFBucket,
    commit: xet::xet_session::XetUploadCommit,
    path_to_task: Mutex<Vec<(String, UniqueID)>>,
}

impl BucketXetUploadSession {
    /// Queue an in-memory file for streaming upload to this bucket's CAS.
    ///
    /// xet starts uploading the bytes to CAS in the background. The future
    /// resolves once xet has accepted the bytes — at that point the caller
    /// may drop their own copy.
    pub async fn upload_bytes(&self, remote_path: impl Into<String>, bytes: Vec<u8>) -> HFResult<()> {
        let remote = remote_path.into();
        tracing::debug!(path = remote.as_str(), len = bytes.len(), "queuing bucket xet upload (bytes)");
        let handle = self
            .commit
            .upload_bytes(bytes, Sha256Policy::Compute, Some(remote.clone()))
            .await
            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        self.record_handle(remote, &handle);
        Ok(())
    }

    /// Queue a local file for streaming upload to this bucket's CAS.
    ///
    /// xet reads the file lazily during the background transfer, so the file
    /// must exist (unchanged) until [`finish`](Self::finish) or
    /// [`finish_and_register`](Self::finish_and_register) returns.
    pub async fn upload_file(&self, remote_path: impl Into<String>, source: impl Into<PathBuf>) -> HFResult<()> {
        let remote = remote_path.into();
        let source = source.into();
        tracing::debug!(path = remote.as_str(), source = ?source, "queuing bucket xet upload (file)");
        let handle = self
            .commit
            .upload_from_path(source, Sha256Policy::Compute)
            .await
            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        self.record_handle(remote, &handle);
        Ok(())
    }

    fn record_handle(&self, remote: String, handle: &XetFileUpload) {
        self.path_to_task
            .lock()
            .expect("bucket xet upload session mutex poisoned")
            .push((remote, handle.task_id()));
    }

    /// Finalize the CAS uploads. The returned descriptors must still be
    /// registered with the bucket (via
    /// [`batch_operations`](super::HFBucket::batch_operations)) before the
    /// files are visible — see [`finish_and_register`](Self::finish_and_register)
    /// for the common case.
    pub async fn finish(self) -> HFResult<HashMap<String, UploadedXetFile>> {
        let results = self.commit.commit().await.map_err(|e| HFError::xet(XetOperation::Upload, e))?;
        tracing::info!(file_count = results.uploads.len(), "bucket xet streaming upload session committed");

        let path_to_task = self
            .path_to_task
            .into_inner()
            .expect("bucket xet upload session mutex poisoned");

        let mut out: HashMap<String, UploadedXetFile> = HashMap::with_capacity(path_to_task.len());
        for (path, task_id) in path_to_task {
            let metadata = results
                .uploads
                .get(&task_id)
                .ok_or_else(|| HFError::Other(format!("xet upload result missing for {path:?}")))?;
            let sha256_oid = metadata.xet_info.sha256().map(|s| s.to_string()).unwrap_or_default(); // buckets use xet_hash, so sha256 is optional context.
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

    /// Convenience: seal CAS uploads and register every uploaded path in the
    /// bucket via [`batch_operations`](super::HFBucket::batch_operations) in
    /// one call. Files become visible to the bucket once this returns.
    pub async fn finish_and_register(self) -> HFResult<()> {
        let bucket = self.bucket.clone();
        let uploaded = self.finish().await?;
        let add_files: Vec<BucketAddFile> = uploaded
            .into_iter()
            .map(|(path, info)| BucketAddFile {
                path,
                xet_hash: info.xet_hash,
                size: info.size,
                mtime: None,
                content_type: None,
            })
            .collect();
        if add_files.is_empty() {
            return Ok(());
        }
        bucket.batch_operations().add_files(add_files).send().await?;
        Ok(())
    }
}

impl HFBucket {
    /// Open a streaming xet-upload session for this bucket.
    ///
    /// Each [`upload_bytes`](BucketXetUploadSession::upload_bytes) /
    /// [`upload_file`](BucketXetUploadSession::upload_file) call uploads its
    /// content to xet content-addressable storage immediately. After all
    /// content is uploaded, call
    /// [`finish_and_register`](BucketXetUploadSession::finish_and_register) to
    /// seal the CAS uploads and register the files with the bucket in one
    /// step; or call [`finish`](BucketXetUploadSession::finish) to recover
    /// descriptors and register them yourself (when combining with other
    /// `batch_operations` ops like deletes or copies).
    ///
    /// Use this when total upload payload exceeds local RAM or disk: the
    /// caller never has to hold all bytes (or all temp files) at once.
    pub async fn open_xet_upload_session(&self) -> HFResult<BucketXetUploadSession> {
        let bucket_id = self.bucket_id();
        let token_url = bucket_xet_token_url(&self.hf_client, "write", &bucket_id);
        let commit =
            new_xet_upload_commit(&self.hf_client, token_url, &bucket_id, NotFoundContext::Bucket, "bucket").await?;
        Ok(BucketXetUploadSession {
            bucket: self.clone(),
            commit,
            path_to_task: Mutex::new(Vec::new()),
        })
    }
}

/// Synchronous/blocking counterpart to [`BucketXetUploadSession`].
///
/// Constructed via
/// [`HFBucketSync::open_xet_upload_session`](crate::HFBucketSync::open_xet_upload_session).
#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub struct BucketXetUploadSessionSync {
    inner: BucketXetUploadSession,
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
}

#[cfg(feature = "blocking")]
impl BucketXetUploadSessionSync {
    /// Blocking counterpart of [`BucketXetUploadSession::upload_bytes`].
    pub fn upload_bytes(&self, remote_path: impl Into<String>, bytes: Vec<u8>) -> HFResult<()> {
        self.runtime.block_on(self.inner.upload_bytes(remote_path, bytes))
    }

    /// Blocking counterpart of [`BucketXetUploadSession::upload_file`].
    pub fn upload_file(&self, remote_path: impl Into<String>, source: impl Into<PathBuf>) -> HFResult<()> {
        self.runtime.block_on(self.inner.upload_file(remote_path, source))
    }

    /// Blocking counterpart of [`BucketXetUploadSession::finish`].
    pub fn finish(self) -> HFResult<HashMap<String, UploadedXetFile>> {
        self.runtime.block_on(self.inner.finish())
    }

    /// Blocking counterpart of [`BucketXetUploadSession::finish_and_register`].
    pub fn finish_and_register(self) -> HFResult<()> {
        self.runtime.block_on(self.inner.finish_and_register())
    }
}

#[cfg(feature = "blocking")]
impl crate::HFBucketSync {
    /// Blocking counterpart of [`HFBucket::open_xet_upload_session`].
    pub fn open_xet_upload_session(&self) -> HFResult<BucketXetUploadSessionSync> {
        let inner = self.runtime.block_on(self.inner.open_xet_upload_session())?;
        Ok(BucketXetUploadSessionSync {
            inner,
            runtime: self.runtime.clone(),
        })
    }
}
