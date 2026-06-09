//! Pipeline primitives for `upload_large_folder`: tuning config, the
//! oldest-item-anchored batch buffer, per-file commit-ready messages, shared
//! progress counters, and per-file stage seeding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::time::Instant;

use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::repository::upload::ResolvedAdd;
use crate::repository::upload_large_folder::local_folder::{LocalUploadFileMetadata, LocalUploadFilePaths};

/// Max files queued into a single xet `UploadCommit` before it is dispatched.
/// Larger batches improve client-side dedup and reduce finalize round-trips.
pub(crate) const XET_BATCH_SIZE: usize = 1024;
/// Files per `/preupload` classify request (Hub endpoint limit).
pub(crate) const PREUPLOAD_BATCH_SIZE: usize = 256;
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

/// FIFO buffer that timestamps each item at push and exposes a flush deadline
/// anchored to the OLDEST buffered item. The timer therefore does not run while
/// the buffer is empty, and advances to the next-oldest item after a partial
/// drain — so an item arriving after an idle gap gets its full wait window.
pub(crate) struct TimedBatchBuffer<T> {
    items: VecDeque<(Instant, T)>,
}

impl<T> TimedBatchBuffer<T> {
    pub(crate) fn new() -> Self {
        Self { items: VecDeque::new() }
    }

    pub(crate) fn push(&mut self, item: T) {
        self.items.push_back((Instant::now(), item));
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Flush deadline = oldest item's enqueue instant + `max_wait`; `None` when empty.
    pub(crate) fn deadline(&self, max_wait: Duration) -> Option<Instant> {
        self.items.front().map(|(t, _)| *t + max_wait)
    }

    /// Drain up to `max` oldest items (fewer if the buffer is smaller).
    pub(crate) fn take(&mut self, max: usize) -> Vec<T> {
        let n = max.min(self.items.len());
        self.items.drain(..n).map(|(_, item)| item).collect()
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

/// Live progress counters shared between the producer and committer. Display
/// only — the final report is computed from the post-join `items` scan plus the
/// per-invocation byte counters here.
#[derive(Default)]
pub(crate) struct StatusCounters {
    pub files_total: AtomicUsize,
    pub hashed: AtomicUsize,
    pub upload_mode_known: AtomicUsize,
    pub lfs_total: AtomicUsize,
    pub preuploaded: AtomicUsize,
    pub committed: AtomicUsize,
    pub ignored: AtomicUsize,
    pub bytes_uploaded: AtomicU64,
    pub dedup_bytes_saved: AtomicU64,
}

/// Emit a `LargeFolderStatus` snapshot from the current counter values.
pub(crate) fn emit_status(counters: &StatusCounters, progress: &Option<Progress>) {
    progress.emit(UploadEvent::LargeFolderStatus {
        files_total: counters.files_total.load(Ordering::Relaxed),
        hashed: counters.hashed.load(Ordering::Relaxed),
        upload_mode_known: counters.upload_mode_known.load(Ordering::Relaxed),
        lfs_total: counters.lfs_total.load(Ordering::Relaxed),
        preuploaded: counters.preuploaded.load(Ordering::Relaxed),
        committed: counters.committed.load(Ordering::Relaxed),
        ignored: counters.ignored.load(Ordering::Relaxed),
        bytes_uploaded: counters.bytes_uploaded.load(Ordering::Relaxed),
        dedup_bytes_saved: counters.dedup_bytes_saved.load(Ordering::Relaxed),
    });
}

/// One file, fully resolved and ready to be referenced in a Hub commit. Carries
/// owned cache handles so the committer can persist `is_committed` immediately
/// after a successful commit POST, without sharing the producer's `items`.
pub(crate) struct CommitReady {
    pub idx: usize,
    pub resolved: ResolvedAdd,
    pub paths: LocalUploadFilePaths,
    pub meta: LocalUploadFileMetadata,
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

    #[tokio::test(start_paused = true)]
    async fn timed_buffer_deadline_anchored_to_oldest() {
        let wait = std::time::Duration::from_secs(300);
        let mut buf: TimedBatchBuffer<u32> = TimedBatchBuffer::new();
        assert!(buf.is_empty());
        assert!(buf.deadline(wait).is_none());

        let t0 = tokio::time::Instant::now();
        buf.push(1);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.deadline(wait), Some(t0 + wait));

        tokio::time::advance(std::time::Duration::from_secs(100)).await;
        buf.push(2); // enqueued 100s later
        // Still anchored to item 1.
        assert_eq!(buf.deadline(wait), Some(t0 + wait));
    }

    #[tokio::test(start_paused = true)]
    async fn timed_buffer_take_advances_anchor() {
        let wait = std::time::Duration::from_secs(300);
        let mut buf: TimedBatchBuffer<u32> = TimedBatchBuffer::new();
        let t0 = tokio::time::Instant::now();
        buf.push(10);
        tokio::time::advance(std::time::Duration::from_secs(100)).await;
        buf.push(20);

        let drained = buf.take(1);
        assert_eq!(drained, vec![10]);
        assert_eq!(buf.len(), 1);
        // Anchor advanced to item 20 (enqueued at t0 + 100s).
        assert_eq!(buf.deadline(wait), Some(t0 + std::time::Duration::from_secs(100) + wait));

        let rest = buf.take(50);
        assert_eq!(rest, vec![20]);
        assert!(buf.is_empty());
        assert!(buf.deadline(wait).is_none());
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
