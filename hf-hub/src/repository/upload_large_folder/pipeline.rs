//! Pipeline primitives for `upload_large_folder`: tuning config, the
//! oldest-item-anchored batch buffer, per-file commit-ready messages, shared
//! progress counters, and per-file stage seeding.

use std::collections::VecDeque;
use std::time::Duration;

use tokio::time::Instant;

use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::repository::upload::ResolvedAdd;
use crate::repository::upload_large_folder::local_folder::{LocalUploadFileMetadata, LocalUploadFilePaths};

/// Max files queued into a single xet `UploadCommit` before it is dispatched.
/// Larger batches improve client-side dedup and reduce finalize round-trips.
pub(crate) const XET_BATCH_SIZE: usize = 1024;
/// Max files in a single Hub commit (validated top of Python's old scale).
pub(crate) const MAX_COMMIT_FILES: usize = 1000;
/// How long the oldest queued lfs file may wait before a partial xet batch is
/// dispatched anyway.
pub(crate) const MAX_XET_BATCH_WAIT: Duration = Duration::from_secs(300);
/// How long the oldest queued commit-ready file may wait before a partial commit.
pub(crate) const MAX_COMMIT_WAIT: Duration = Duration::from_secs(300);

/// Tuning for one `upload_large_folder` run. Production builds it from
/// `num_workers`; tests construct it directly with tiny timeouts.
#[derive(Debug, Clone)]
pub(crate) struct PipelineConfig {
    pub num_workers: usize,
    pub xet_batch_size: usize,
    pub max_commit_files: usize,
    pub max_xet_batch_wait: Duration,
    pub max_commit_wait: Duration,
}

impl PipelineConfig {
    pub(crate) fn from_num_workers(num_workers: Option<usize>) -> Self {
        Self {
            num_workers: num_workers.map(|n| n.max(1)).unwrap_or_else(default_num_workers),
            xet_batch_size: XET_BATCH_SIZE,
            max_commit_files: MAX_COMMIT_FILES,
            max_xet_batch_wait: MAX_XET_BATCH_WAIT,
            max_commit_wait: MAX_COMMIT_WAIT,
        }
    }
}

/// Default concurrency budget: half the available parallelism, at least 1.
pub(crate) fn default_num_workers() -> usize {
    (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) / 2).max(1)
}

/// Await `deadline` if present, else never resolve. Lets a `select!` arm be a
/// no-op when no deadline is active (empty buffer).
pub(crate) async fn sleep_or_pending(deadline: Option<Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending::<()>().await,
    }
}

/// The next stage of work for a file, derived from its persisted metadata.
/// Tolerant of both Rust- and Python-written caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkStage {
    Classify,
    PreuploadLfs,
    Commit,
    Done,
}

pub(crate) fn seed_stage(meta: &LocalUploadFileMetadata) -> WorkStage {
    if meta.is_committed || meta.should_ignore == Some(true) {
        WorkStage::Done
    } else if meta.upload_mode.is_none() {
        WorkStage::Classify
    } else if meta.upload_mode.as_deref() == Some("lfs") && !meta.is_uploaded {
        WorkStage::PreuploadLfs
    } else {
        WorkStage::Commit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository::upload_large_folder::local_folder::LocalUploadFileMetadata;

    fn meta_with(
        mode: Option<&str>,
        sha: Option<&str>,
        uploaded: bool,
        committed: bool,
        ignore: Option<bool>,
    ) -> LocalUploadFileMetadata {
        let mut m = LocalUploadFileMetadata::new(10);
        m.upload_mode = mode.map(str::to_string);
        m.sha256 = sha.map(str::to_string);
        m.is_uploaded = uploaded;
        m.is_committed = committed;
        m.should_ignore = ignore;
        m
    }

    #[test]
    fn seed_stage_routing() {
        // Fresh: no mode -> classify.
        assert!(matches!(seed_stage(&meta_with(None, None, false, false, None)), WorkStage::Classify));
        // Python wrote sha256 but no mode yet -> still classify.
        assert!(matches!(seed_stage(&meta_with(None, Some("ab"), false, false, None)), WorkStage::Classify));
        // lfs, not uploaded -> preupload.
        assert!(matches!(
            seed_stage(&meta_with(Some("lfs"), Some("ab"), false, false, None)),
            WorkStage::PreuploadLfs
        ));
        // lfs, uploaded, not committed -> commit.
        assert!(matches!(seed_stage(&meta_with(Some("lfs"), Some("ab"), true, false, None)), WorkStage::Commit));
        // regular, not committed -> commit.
        assert!(matches!(seed_stage(&meta_with(Some("regular"), None, false, false, None)), WorkStage::Commit));
        // committed -> done.
        assert!(matches!(seed_stage(&meta_with(Some("regular"), None, false, true, None)), WorkStage::Done));
        // should_ignore -> done.
        assert!(matches!(seed_stage(&meta_with(Some("regular"), None, false, false, Some(true))), WorkStage::Done));
    }

    #[test]
    fn default_num_workers_is_at_least_one() {
        assert!(default_num_workers() >= 1);
    }

    #[test]
    fn pipeline_config_defaults() {
        let cfg = PipelineConfig::from_num_workers(Some(4));
        assert_eq!(cfg.num_workers, 4);
        assert_eq!(cfg.xet_batch_size, 1024);
        assert_eq!(cfg.max_commit_files, 1000);
        assert_eq!(cfg.max_xet_batch_wait, std::time::Duration::from_secs(300));
        assert_eq!(cfg.max_commit_wait, std::time::Duration::from_secs(300));

        let auto = PipelineConfig::from_num_workers(None);
        assert!(auto.num_workers >= 1);
    }

}
