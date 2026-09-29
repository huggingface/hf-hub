//! Streamed multi-commit pipeline backing [`HFRepository::upload_operations`] and
//! [`HFRepository::upload_folder`].
//!
//! Mirrors `huggingface_hub`'s large-folder upload: add operations are pulled from a stream,
//! classified via the `preupload` endpoint in chunks, and grouped into adaptively-sized batches. A coordinator uploads
//! each batch's LFS content via xet while a committer commits the previous batch, so transfer and commit
//! round-trips overlap. A small folder lands as a single commit; a large one as several chained
//! commits. With `create_pr`, the first commit opens the pull request and every later commit
//! targets its `refs/pr/N` ref.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;
#[cfg(not(target_family = "wasm"))]
use std::time::Instant;

use futures::channel::mpsc;
use futures::{SinkExt, StreamExt};
#[cfg(target_family = "wasm")]
use web_time::Instant;

use super::{CommitRequest, UploadOperationsParams, prepare_source};
use crate::constants;
use crate::error::{HFError, HFResult};
use crate::progress::{EmitEvent, Progress, ProgressEvent, ProgressHandler, UploadEvent};
use crate::repository::files::matches_any_glob;
use crate::repository::{
    AddSource, CommitInfo, CommitOperation, CommitOperationStream, HFRepository, RepoTreeEntry, RepoType,
};

/// Files classified per `preupload` call.
const PREUPLOAD_BATCH_SIZE: usize = 256;
/// Files-per-commit ladder; grows after fast commits, shrinks after slow ones.
const COMMIT_SIZE_SCALE: [usize; 10] = [20, 50, 75, 100, 125, 200, 250, 400, 600, 1000];
const INITIAL_COMMIT_SIZE_INDEX: usize = 6;
/// Commits faster than this grow the next batch; slower ones shrink it.
const TARGET_COMMIT_DURATION: Duration = Duration::from_secs(40);
/// A batch older than this is flushed even if under the file cap.
const MAX_COMMIT_INTERVAL: Duration = Duration::from_secs(5 * 60);
/// Budget of base64-inlined (regular) content per commit.
const REGULAR_CONTENT_BYTES_BUDGET: u64 = 100 * 1024 * 1024;

struct AdaptiveCommitSize {
    index: AtomicUsize,
}

impl AdaptiveCommitSize {
    fn new() -> Self {
        Self {
            index: AtomicUsize::new(INITIAL_COMMIT_SIZE_INDEX),
        }
    }

    fn current(&self) -> usize {
        COMMIT_SIZE_SCALE[self.index.load(Ordering::Relaxed)]
    }

    fn record_commit(&self, duration: Duration) {
        let index = self.index.load(Ordering::Relaxed);
        let next = if duration < TARGET_COMMIT_DURATION {
            (index + 1).min(COMMIT_SIZE_SCALE.len() - 1)
        } else {
            index.saturating_sub(1)
        };
        self.index.store(next, Ordering::Relaxed);
    }
}

struct PreparedAdd {
    path_in_repo: String,
    source: AddSource,
    size: u64,
    sha256: String,
    lfs: bool,
}

struct BatchAccumulator {
    adds: Vec<PreparedAdd>,
    regular_bytes: u64,
    started_at: Instant,
}

impl BatchAccumulator {
    fn new() -> Self {
        Self {
            adds: Vec::new(),
            regular_bytes: 0,
            started_at: Instant::now(),
        }
    }

    fn push(&mut self, add: PreparedAdd) {
        if !add.lfs {
            self.regular_bytes += add.size;
        }
        self.adds.push(add);
    }

    fn should_flush(&self, max_files: usize, now: Instant) -> bool {
        !self.adds.is_empty()
            && (self.adds.len() >= max_files
                || self.regular_bytes >= REGULAR_CONTENT_BYTES_BUDGET
                || now.duration_since(self.started_at) >= MAX_COMMIT_INTERVAL)
    }
}

/// Tracks the target ref and parent chain across the sequential commits.
struct CommitState {
    revision: String,
    create_pr: bool,
    parent_commit: Option<String>,
    pr: Option<(u64, Option<String>)>,
    committed: usize,
}

impl CommitState {
    fn new(revision: String, create_pr: bool) -> Self {
        Self {
            revision,
            create_pr,
            parent_commit: None,
            pr: None,
            committed: 0,
        }
    }

    /// Only the first commit asks the Hub to open a PR; later ones push to its ref.
    fn open_pr_on_next_commit(&self) -> bool {
        self.create_pr && self.pr.is_none()
    }

    fn record(&mut self, info: &CommitInfo) -> HFResult<()> {
        if self.open_pr_on_next_commit() {
            let pr_num = info
                .pr_num
                .or_else(|| info.pr_url.as_deref().and_then(parse_pr_num))
                .ok_or_else(|| {
                    HFError::Other(
                        "commit with create_pr succeeded but the response did not identify the pull request"
                            .to_string(),
                    )
                })?;
            self.revision = format!("refs/pr/{pr_num}");
            self.pr = Some((pr_num, info.pr_url.clone()));
        }
        self.parent_commit = info.commit_oid.clone();
        self.committed += 1;
        Ok(())
    }

    fn contextualize_error(&self, err: HFError) -> HFError {
        match self.pr {
            Some((pr_num, _)) => HFError::Other(format!(
                "upload failed after {} commit(s) to pull request #{pr_num}; re-run with \
                 revision=\"refs/pr/{pr_num}\" and create_pr=false to resume: {err}",
                self.committed
            )),
            None => err,
        }
    }
}

fn parse_pr_num(pr_url: &str) -> Option<u64> {
    pr_url.trim_end_matches('/').rsplit('/').next()?.parse().ok()
}

/// Rebases per-batch xet `Progress` events onto the whole upload's totals. `total_bytes` grows as
/// operations are pulled from the stream. Lifecycle events from the inner per-batch calls are
/// dropped; the pipeline emits those once.
struct AggregatingProgress {
    inner: Progress,
    total_bytes: AtomicU64,
    completed_base: AtomicU64,
    transfer_base: AtomicU64,
    batch_transfer_completed: AtomicU64,
}

impl AggregatingProgress {
    fn new(inner: Progress) -> Self {
        Self {
            inner,
            total_bytes: AtomicU64::new(0),
            completed_base: AtomicU64::new(0),
            transfer_base: AtomicU64::new(0),
            batch_transfer_completed: AtomicU64::new(0),
        }
    }

    fn add_discovered_bytes(&self, bytes: u64) {
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn finish_batch(&self, batch_content_bytes: u64) {
        self.completed_base.fetch_add(batch_content_bytes, Ordering::Relaxed);
        let transferred = self.batch_transfer_completed.swap(0, Ordering::Relaxed);
        self.transfer_base.fetch_add(transferred, Ordering::Relaxed);
    }
}

impl ProgressHandler for AggregatingProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                files,
                ..
            }) => {
                self.batch_transfer_completed
                    .store(*transfer_bytes_completed, Ordering::Relaxed);
                let transfer_base = self.transfer_base.load(Ordering::Relaxed);
                self.inner.on_progress(&ProgressEvent::Upload(UploadEvent::Progress {
                    bytes_completed: self.completed_base.load(Ordering::Relaxed) + bytes_completed,
                    total_bytes: self.total_bytes.load(Ordering::Relaxed),
                    bytes_per_sec: *bytes_per_sec,
                    transfer_bytes_completed: transfer_base + transfer_bytes_completed,
                    transfer_bytes: transfer_base + transfer_bytes,
                    transfer_bytes_per_sec: *transfer_bytes_per_sec,
                    files: files.clone(),
                }));
            },
            ProgressEvent::Upload(_) => {},
            ProgressEvent::Download(_) => self.inner.on_progress(event),
        }
    }
}

/// A batch whose LFS content is uploaded and is ready to commit.
struct CommitJob {
    operations: Vec<CommitOperation>,
    lfs_uploaded: HashMap<String, (String, u64)>,
    is_last: bool,
}

/// Settings shared by every batch of one upload.
struct PipelineContext<'a> {
    revision: &'a str,
    create_pr: bool,
    commit_size: &'a AdaptiveCommitSize,
    aggregator: Option<Arc<AggregatingProgress>>,
}

impl<T: RepoType> HFRepository<T> {
    pub(super) async fn upload_operations_pipeline(
        &self,
        operations: CommitOperationStream,
        params: UploadOperationsParams,
    ) -> HFResult<CommitInfo> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let commit_message = params.commit_message.as_deref().unwrap_or("Upload files");

        let mut delete_operations = Vec::new();
        if let Some(ref delete_patterns) = params.delete_patterns {
            let stream = self.list_tree().revision(revision.to_string()).recursive(true).send()?;
            futures::pin_mut!(stream);
            while let Some(entry) = stream.next().await {
                if let RepoTreeEntry::File { path, .. } = entry?
                    && matches_any_glob(delete_patterns, &path)
                {
                    delete_operations.push(CommitOperation::delete(path));
                }
            }
        }

        params.progress.emit(UploadEvent::Start {
            total_files: 0,
            total_bytes: 0,
        });

        let commit_size = AdaptiveCommitSize::new();
        let context = PipelineContext {
            revision,
            create_pr: params.create_pr,
            commit_size: &commit_size,
            aggregator: params.progress.clone().map(|inner| Arc::new(AggregatingProgress::new(inner))),
        };

        // Zero buffer: one ready batch may wait while the next one uploads.
        let (job_sender, job_receiver) = mpsc::channel::<CommitJob>(0);
        let (_, info) = futures::try_join!(
            self.coordinate(operations, delete_operations, &context, job_sender),
            self.commit_batches(
                job_receiver,
                &context,
                commit_message,
                params.commit_description.as_deref(),
                &params.progress,
            ),
        )?;

        params.progress.emit(UploadEvent::Complete);
        Ok(info)
    }

    async fn coordinate(
        &self,
        operations: CommitOperationStream,
        delete_operations: Vec<CommitOperation>,
        context: &PipelineContext<'_>,
        mut job_sender: mpsc::Sender<CommitJob>,
    ) -> HFResult<()> {
        let mut operations = operations.peekable();
        let mut pending_deletes = Some(delete_operations);
        if std::pin::Pin::new(&mut operations).peek().await.is_none() {
            let job = self
                .upload_batch(BatchAccumulator::new(), pending_deletes.take(), context, true)
                .await?;
            return send_job(&mut job_sender, job).await;
        }

        let mut batch = BatchAccumulator::new();
        let mut chunk = Vec::with_capacity(PREUPLOAD_BATCH_SIZE);
        loop {
            while chunk.len() < PREUPLOAD_BATCH_SIZE {
                match operations.next().await {
                    Some(operation) => chunk.push(require_add(operation?)?),
                    None => break,
                }
            }
            self.classify_chunk(&chunk, context, &mut batch).await?;
            chunk.clear();
            let is_last = std::pin::Pin::new(&mut operations).peek().await.is_none();
            if is_last || batch.should_flush(context.commit_size.current(), Instant::now()) {
                let full_batch = std::mem::replace(&mut batch, BatchAccumulator::new());
                let job = self.upload_batch(full_batch, pending_deletes.take(), context, is_last).await?;
                send_job(&mut job_sender, job).await?;
            }
            if is_last {
                return Ok(());
            }
        }
    }

    async fn classify_chunk(
        &self,
        chunk: &[(String, AddSource)],
        context: &PipelineContext<'_>,
        batch: &mut BatchAccumulator,
    ) -> HFResult<()> {
        let mut prepared: Vec<(String, AddSource, u64, Vec<u8>, String)> = Vec::with_capacity(chunk.len());
        for (path_in_repo, source) in chunk {
            let (size, sample, sha256) = prepare_source(source).await?;
            prepared.push((path_in_repo.clone(), source.clone(), size, sample, sha256));
        }
        if let Some(aggregator) = &context.aggregator {
            aggregator.add_discovered_bytes(prepared.iter().map(|(_, _, size, ..)| size).sum());
        }
        let files: Vec<(&str, u64, &[u8])> = prepared
            .iter()
            .map(|(path, _, size, sample, _)| (path.as_str(), *size, sample.as_slice()))
            .collect();
        let upload_modes = self
            .fetch_upload_modes(&self.repo_path(), self.repo_type.plural(), context.revision, &files, context.create_pr)
            .await?;
        for (path_in_repo, source, size, _, sha256) in prepared {
            let lfs = size > 0 && upload_modes.get(&path_in_repo).is_some_and(|mode| mode == "lfs");
            batch.push(PreparedAdd {
                path_in_repo,
                source,
                size,
                sha256,
                lfs,
            });
        }
        Ok(())
    }

    /// Upload a batch's LFS files via xet and assemble its commit job. When the Hub does not
    /// offer xet transfer, LFS files fall back to inline upload, matching `create_commit`.
    async fn upload_batch(
        &self,
        batch: BatchAccumulator,
        delete_operations: Option<Vec<CommitOperation>>,
        context: &PipelineContext<'_>,
        is_last: bool,
    ) -> HFResult<CommitJob> {
        let mut operations = delete_operations.unwrap_or_default();
        operations.reserve(batch.adds.len());
        let mut xet_files: Vec<(String, AddSource)> = Vec::new();
        let mut lfs_uploaded: HashMap<String, (String, u64)> = HashMap::new();
        let mut batch_content_bytes = 0u64;
        for add in batch.adds {
            batch_content_bytes += add.size;
            if add.lfs {
                xet_files.push((add.path_in_repo.clone(), add.source.clone()));
                lfs_uploaded.insert(add.path_in_repo.clone(), (add.sha256, add.size));
            }
            operations.push(CommitOperation::Add {
                path_in_repo: add.path_in_repo,
                source: add.source,
            });
        }

        if !xet_files.is_empty() {
            let objects: Vec<(&str, u64)> =
                lfs_uploaded.values().map(|(sha256, size)| (sha256.as_str(), *size)).collect();
            // PR uploads omit the target branch ref because contributors may not have write access to it.
            let transfer_revision = if context.create_pr {
                None
            } else {
                Some(context.revision)
            };
            let chosen_transfer = self
                .post_lfs_batch_info(&self.repo_path(), self.repo_type.url_prefix(), transfer_revision, &objects)
                .await?;
            if chosen_transfer.as_deref() == Some("xet") {
                let batch_progress = context
                    .aggregator
                    .clone()
                    .map(|aggregator| Progress::from(aggregator as Arc<dyn ProgressHandler>));
                self.xet_upload(&xet_files, context.revision, context.create_pr, &batch_progress)
                    .await?;
            } else {
                tracing::warn!(
                    ?chosen_transfer,
                    "LFS batch did not choose xet transfer; LFS files will fall through to inline upload"
                );
                lfs_uploaded.clear();
            }
        }
        if let Some(aggregator) = &context.aggregator {
            aggregator.finish_batch(batch_content_bytes);
        }

        Ok(CommitJob {
            operations,
            lfs_uploaded,
            is_last,
        })
    }

    async fn commit_batches(
        &self,
        mut job_receiver: mpsc::Receiver<CommitJob>,
        context: &PipelineContext<'_>,
        commit_message: &str,
        commit_description: Option<&str>,
        progress: &Option<Progress>,
    ) -> HFResult<CommitInfo> {
        let mut state = CommitState::new(context.revision.to_string(), context.create_pr);
        let mut last_info: Option<CommitInfo> = None;
        while let Some(job) = job_receiver.next().await {
            if job.is_last {
                progress.emit(UploadEvent::Committing);
            }
            let started_at = Instant::now();
            let info = self
                .post_commit(CommitRequest {
                    operations: &job.operations,
                    lfs_uploaded: &job.lfs_uploaded,
                    commit_message,
                    commit_description,
                    parent_commit: state.parent_commit.as_deref(),
                    revision: &state.revision,
                    create_pr: state.open_pr_on_next_commit(),
                })
                .await
                .map_err(|err| state.contextualize_error(err))?;
            context.commit_size.record_commit(started_at.elapsed());
            state.record(&info)?;
            tracing::info!(
                commit_index = state.committed - 1,
                commit_oid = info.commit_oid.as_deref(),
                files = job.operations.len(),
                "upload batch committed"
            );
            progress.emit(UploadEvent::CommitCompleted {
                commit_index: state.committed - 1,
                commit_oid: info.commit_oid.clone(),
            });
            last_info = Some(info);
        }

        let mut info = last_info.ok_or_else(|| HFError::Other("upload produced no commits".to_string()))?;
        if let Some((pr_num, pr_url)) = state.pr {
            info.pr_num = Some(pr_num);
            info.pr_url = pr_url;
        }
        Ok(info)
    }
}

fn require_add(operation: CommitOperation) -> HFResult<(String, AddSource)> {
    match operation {
        CommitOperation::Add { path_in_repo, source } => Ok((path_in_repo, source)),
        CommitOperation::Delete { path_in_repo } => Err(HFError::InvalidParameter(format!(
            "upload_operations only accepts add operations, got a delete for {path_in_repo:?}; use delete_patterns \
             to delete remote files"
        ))),
    }
}

async fn send_job(job_sender: &mut mpsc::Sender<CommitJob>, job: CommitJob) -> HFResult<()> {
    job_sender
        .send(job)
        .await
        .map_err(|_| HFError::Other("upload committer stopped before all batches were sent".to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn commit_size_starts_at_250() {
        assert_eq!(AdaptiveCommitSize::new().current(), 250);
    }

    #[test]
    fn commit_size_grows_after_fast_commit_and_shrinks_after_slow_commit() {
        let size = AdaptiveCommitSize::new();
        size.record_commit(Duration::from_secs(5));
        assert_eq!(size.current(), 400);
        size.record_commit(Duration::from_secs(120));
        size.record_commit(Duration::from_secs(120));
        assert_eq!(size.current(), 200);
    }

    #[test]
    fn commit_size_is_clamped_to_scale() {
        let size = AdaptiveCommitSize::new();
        for _ in 0..20 {
            size.record_commit(Duration::from_secs(1));
        }
        assert_eq!(size.current(), 1000);
        for _ in 0..20 {
            size.record_commit(Duration::from_secs(120));
        }
        assert_eq!(size.current(), 20);
    }

    fn prepared(size: u64, lfs: bool) -> PreparedAdd {
        PreparedAdd {
            path_in_repo: "f".to_string(),
            source: AddSource::bytes(Vec::new()),
            size,
            sha256: String::new(),
            lfs,
        }
    }

    #[test]
    fn batch_flushes_on_file_count() {
        let mut batch = BatchAccumulator::new();
        for _ in 0..5 {
            batch.push(prepared(1, false));
        }
        assert!(batch.should_flush(5, Instant::now()));
        assert!(!batch.should_flush(6, Instant::now()));
    }

    #[test]
    fn batch_flushes_on_regular_byte_budget_only() {
        let mut batch = BatchAccumulator::new();
        batch.push(prepared(REGULAR_CONTENT_BYTES_BUDGET, true));
        assert!(!batch.should_flush(10_000, Instant::now()));
        batch.push(prepared(REGULAR_CONTENT_BYTES_BUDGET, false));
        assert!(batch.should_flush(10_000, Instant::now()));
    }

    #[test]
    fn batch_flushes_on_age() {
        let mut batch = BatchAccumulator::new();
        batch.push(prepared(1, false));
        let later = Instant::now() + MAX_COMMIT_INTERVAL + Duration::from_secs(1);
        assert!(batch.should_flush(10_000, later));
    }

    #[test]
    fn empty_batch_never_flushes() {
        let batch = BatchAccumulator::new();
        assert!(!batch.should_flush(1, Instant::now() + Duration::from_secs(3600)));
    }

    fn commit_info(oid: &str, pr_url: Option<&str>) -> CommitInfo {
        CommitInfo {
            commit_url: None,
            commit_message: None,
            commit_description: None,
            commit_oid: Some(oid.to_string()),
            pr_url: pr_url.map(str::to_string),
            pr_num: None,
        }
    }

    #[test]
    fn commit_state_chains_parents_on_branch() {
        let mut state = CommitState::new("main".to_string(), false);
        assert!(!state.open_pr_on_next_commit());
        assert_eq!(state.parent_commit, None);
        state.record(&commit_info("sha1", None)).unwrap();
        assert_eq!(state.parent_commit.as_deref(), Some("sha1"));
        state.record(&commit_info("sha2", None)).unwrap();
        assert_eq!(state.parent_commit.as_deref(), Some("sha2"));
        assert_eq!(state.revision, "main");
    }

    #[test]
    fn commit_state_switches_to_pr_ref_after_first_commit() {
        let mut state = CommitState::new("main".to_string(), true);
        assert!(state.open_pr_on_next_commit());
        state
            .record(&commit_info("sha1", Some("https://huggingface.co/owner/repo/discussions/7")))
            .unwrap();
        assert!(!state.open_pr_on_next_commit());
        assert_eq!(state.revision, "refs/pr/7");
        assert_eq!(state.parent_commit.as_deref(), Some("sha1"));
        state.record(&commit_info("sha2", None)).unwrap();
        assert_eq!(state.revision, "refs/pr/7");
        assert!(matches!(state.pr, Some((7, Some(_)))));
    }

    #[test]
    fn commit_state_errors_when_pr_is_unidentified() {
        let mut state = CommitState::new("main".to_string(), true);
        assert!(state.record(&commit_info("sha1", None)).is_err());
    }

    #[test]
    fn parse_pr_num_from_url() {
        assert_eq!(parse_pr_num("https://huggingface.co/datasets/o/r/discussions/12"), Some(12));
        assert_eq!(parse_pr_num("https://huggingface.co/o/r/discussions/3/"), Some(3));
        assert_eq!(parse_pr_num("https://huggingface.co/o/r"), None);
    }

    #[derive(Default)]
    struct CaptureProgress(Mutex<Vec<(u64, u64, u64, u64)>>, AtomicUsize);

    impl ProgressHandler for CaptureProgress {
        fn on_progress(&self, event: &ProgressEvent) {
            self.1.fetch_add(1, Ordering::Relaxed);
            if let ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                transfer_bytes_completed,
                transfer_bytes,
                ..
            }) = event
            {
                self.0.lock().unwrap().push((
                    *bytes_completed,
                    *total_bytes,
                    *transfer_bytes_completed,
                    *transfer_bytes,
                ));
            }
        }
    }

    fn progress_event(done: u64, transfer_done: u64) -> ProgressEvent {
        ProgressEvent::Upload(UploadEvent::Progress {
            bytes_completed: done,
            total_bytes: 50,
            bytes_per_sec: None,
            transfer_bytes_completed: transfer_done,
            transfer_bytes: 30,
            transfer_bytes_per_sec: None,
            files: vec![],
        })
    }

    #[test]
    fn aggregating_progress_rebases_onto_growing_totals() {
        let capture = Arc::new(CaptureProgress::default());
        let aggregator = AggregatingProgress::new(Progress::from(capture.clone()));
        aggregator.add_discovered_bytes(50);
        aggregator.on_progress(&progress_event(10, 5));
        aggregator.on_progress(&progress_event(50, 30));
        aggregator.finish_batch(50);
        aggregator.add_discovered_bytes(50);
        aggregator.on_progress(&progress_event(20, 10));
        let seen = capture.0.lock().unwrap().clone();
        assert_eq!(seen, vec![(10, 50, 5, 30), (50, 50, 30, 30), (70, 100, 40, 60)]);
    }

    #[test]
    fn require_add_rejects_deletes() {
        assert!(require_add(CommitOperation::add_bytes("a", b"x".to_vec())).is_ok());
        assert!(matches!(require_add(CommitOperation::delete("a")), Err(HFError::InvalidParameter(_))));
    }

    #[test]
    fn aggregating_progress_drops_inner_lifecycle_events() {
        let capture = Arc::new(CaptureProgress::default());
        let aggregator = AggregatingProgress::new(Progress::from(capture.clone()));
        aggregator.on_progress(&ProgressEvent::Upload(UploadEvent::Start {
            total_files: 3,
            total_bytes: 100,
        }));
        aggregator.on_progress(&ProgressEvent::Upload(UploadEvent::Committing));
        aggregator.on_progress(&ProgressEvent::Upload(UploadEvent::Complete));
        assert_eq!(capture.1.load(Ordering::Relaxed), 0);
    }
}
