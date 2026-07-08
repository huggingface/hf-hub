//! Streamed multi-commit upload pipeline backing [`HFRepository::upload_folder`].
//! A coordinator classifies and batches files while a committer commits each batch;
//! see the method docs for the coordinator/committer split and PR handling.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::stream::StreamExt;
use tokio::sync::mpsc;

use super::{UploadFolderParams, collect_files_recursive, prepare_source};
use crate::error::{HFError, HFResult};
use crate::progress::{EmitEvent, Progress, ProgressEvent, ProgressHandler, UploadEvent};
use crate::repository::files::matches_any_glob;
use crate::repository::{AddSource, CommitInfo, CommitOperation, HFRepository, RepoTreeEntry, RepoType};
use crate::{constants, retry};

/// Number of files classified per `preupload` call (server-side limit).
const PREUPLOAD_BATCH_SIZE: usize = 256;
/// Files-per-commit ladder; grows after fast commits, shrinks after slow ones.
const COMMIT_SIZE_SCALE: [usize; 10] = [20, 50, 75, 100, 125, 200, 250, 400, 600, 1000];
/// Start at 250 files per commit.
const INITIAL_COMMIT_SIZE_INDEX: usize = 6;
/// Grow the batch size if a commit finished faster than this.
const TARGET_COMMIT_DURATION: Duration = Duration::from_secs(40);
/// Force a flush once the open batch is older than this, even if under the file cap.
const MAX_COMMIT_INTERVAL: Duration = Duration::from_secs(5 * 60);
/// Budget of regular-file content per commit (regular files are base64-inlined).
const REGULAR_CONTENT_BYTES_BUDGET: u64 = 100 * 1024 * 1024;

/// Adaptive files-per-commit ladder. Grows one step after a fast commit, shrinks
/// one step after a slow one or a retryable failure. Clamped to the scale bounds.
struct AdaptiveCommitSize {
    index: usize,
}

impl AdaptiveCommitSize {
    fn new() -> Self {
        Self {
            index: INITIAL_COMMIT_SIZE_INDEX,
        }
    }

    fn current(&self) -> usize {
        COMMIT_SIZE_SCALE[self.index]
    }

    fn record_commit(&mut self, duration: Duration) {
        if duration < TARGET_COMMIT_DURATION {
            self.index = (self.index + 1).min(COMMIT_SIZE_SCALE.len() - 1);
        } else if self.index > 0 {
            self.index -= 1;
        }
    }
}

/// Running state threaded across the sequential batch commits. All commits target a
/// single fixed revision (the branch, or `refs/pr/N` when uploading to a PR); this
/// only tracks the parent-oid chain so each commit builds on the previous one.
struct CommitState {
    prev_oid: Option<String>,
    committed: usize,
}

impl CommitState {
    fn new() -> Self {
        Self {
            prev_oid: None,
            committed: 0,
        }
    }

    /// Parent oid for the next commit: `None` for the first (commit to the ref head),
    /// then the previous commit's oid so commits chain.
    fn next_parent(&self) -> Option<String> {
        if self.committed == 0 {
            None
        } else {
            self.prev_oid.clone()
        }
    }

    /// Record the result of the commit just sent.
    fn record(&mut self, info: &CommitInfo) {
        self.prev_oid = info.commit_oid.clone();
        self.committed += 1;
    }
}

/// A single classified add destined for a batch.
#[derive(Debug, Clone)]
enum PreparedAdd {
    /// Inlined as base64 in the commit body; carries its content size.
    Regular { op: CommitOperation, size: u64 },
    /// Uploaded via xet; committed as an `lfsFile` entry. `oid` is the SHA-256
    /// computed during classification (single-pass with size + sample).
    Xet {
        path_in_repo: String,
        source: AddSource,
        size: u64,
        oid: String,
    },
}

/// Accumulates classified adds until a flush trigger fires.
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
        if let PreparedAdd::Regular { size, .. } = &add {
            self.regular_bytes += *size;
        }
        self.adds.push(add);
    }

    fn is_empty(&self) -> bool {
        self.adds.is_empty()
    }

    /// Should this batch be flushed now, given the current adaptive file cap?
    fn should_flush(&self, max_files: usize, now: Instant) -> bool {
        !self.adds.is_empty()
            && (self.adds.len() >= max_files
                || self.regular_bytes >= REGULAR_CONTENT_BYTES_BUDGET
                || now.duration_since(self.started_at) >= MAX_COMMIT_INTERVAL)
    }
}

/// Wraps a user `Progress` handler so per-batch xet `Progress` events are rebased
/// onto grand totals. `Start`/`Committing`/`Complete` events emitted by the inner
/// per-batch xet calls are swallowed — the pipeline owns those.
struct AggregatingProgress {
    inner: Progress,
    grand_total_bytes: u64,
    completed_base: AtomicU64,
    transfer_base: AtomicU64,
}

impl AggregatingProgress {
    fn new(inner: Progress, grand_total_bytes: u64) -> Self {
        Self {
            inner,
            grand_total_bytes,
            completed_base: AtomicU64::new(0),
            transfer_base: AtomicU64::new(0),
        }
    }

    /// Call after a batch's xet upload finishes to advance the running bases.
    fn finish_batch(&self, batch_content_bytes: u64, batch_transfer_bytes: u64) {
        self.completed_base.fetch_add(batch_content_bytes, Ordering::Relaxed);
        self.transfer_base.fetch_add(batch_transfer_bytes, Ordering::Relaxed);
    }
}

impl ProgressHandler for AggregatingProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes_per_sec,
                files,
                ..
            }) => {
                let cbase = self.completed_base.load(Ordering::Relaxed);
                let tbase = self.transfer_base.load(Ordering::Relaxed);
                self.inner.on_progress(&ProgressEvent::Upload(UploadEvent::Progress {
                    bytes_completed: cbase + bytes_completed,
                    total_bytes: self.grand_total_bytes,
                    bytes_per_sec: *bytes_per_sec,
                    transfer_bytes_completed: tbase + transfer_bytes_completed,
                    // Post-dedup network total isn't known upfront; report only running sent.
                    transfer_bytes: 0,
                    transfer_bytes_per_sec: *transfer_bytes_per_sec,
                    files: files.clone(),
                }));
            },
            // Swallow inner Start/Committing/Complete/CommitCompleted; the pipeline emits those once.
            ProgressEvent::Upload(_) => {},
            ProgressEvent::Download(_) => self.inner.on_progress(event),
        }
    }
}

/// One unit of work handed from the coordinator to the committer: the commit's
/// operations and a handle to the batch's in-flight xet upload.
struct CommitJob {
    operations: Vec<CommitOperation>,
    upload: tokio::task::JoinHandle<HFResult<HashMap<String, (String, u64)>>>,
}

fn source_size(source: &AddSource) -> u64 {
    match source {
        AddSource::Bytes(b) => b.len() as u64,
        AddSource::Stream(s) => s.size(),
        AddSource::File(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
    }
}

impl<T: RepoType> HFRepository<T> {
    /// Streamed multi-commit upload backing [`HFRepository::upload_folder`].
    ///
    /// A coordinator (this task) classifies files via the `preupload` endpoint and
    /// forms adaptively-sized batches; each flush spawns the batch's xet upload and
    /// hands the batch to a committer task over a capacity-1 channel, so the next
    /// batch uploads while the current one commits. Commits chain their parent oid.
    /// When `create_pr` is set, a draft PR is opened up front and every commit targets
    /// its `refs/pr/N` ref.
    pub(super) async fn run_multi_commit_pipeline(&self, params: UploadFolderParams) -> HFResult<CommitInfo> {
        let revision = params
            .revision
            .clone()
            .unwrap_or_else(|| constants::DEFAULT_REVISION.to_string());
        let commit_message = params.commit_message.clone().unwrap_or_else(|| "Upload folder".to_string());

        // 1. Walk the folder into add operations.
        let mut add_operations = Vec::new();
        let base_repo_path = params.path_in_repo.as_deref().unwrap_or("");
        collect_files_recursive(
            &params.folder_path,
            &params.folder_path,
            base_repo_path,
            &params.allow_patterns,
            &params.ignore_patterns,
            &mut add_operations,
        )?;

        // 2. Resolve delete_patterns against the base tree (goes into the first commit).
        let mut delete_operations = Vec::new();
        if let Some(ref delete_patterns) = params.delete_patterns {
            let stream = self.list_tree().revision(revision.clone()).recursive(true).send()?;
            futures::pin_mut!(stream);
            while let Some(entry) = stream.next().await {
                if let RepoTreeEntry::File { path, .. } = entry?
                    && matches_any_glob(delete_patterns, &path)
                {
                    delete_operations.push(CommitOperation::delete(path));
                }
            }
        }

        // 3. When uploading to a PR, open the draft PR up front so every commit targets its ref. The commit response
        //    does not carry PR info, so the PR must be created explicitly (mirroring huggingface_hub); `refs/pr/N` is
        //    derived from its number.
        let (commit_revision, pr_num, pr_url) = if params.create_pr {
            let (num, url) = self
                .create_draft_pull_request(&commit_message, params.commit_description.as_deref())
                .await?;
            (format!("refs/pr/{num}"), Some(num), Some(url))
        } else {
            (revision.clone(), None, None)
        };

        // 4. Grand totals for progress (the full file list is known upfront).
        let total_files = add_operations.len();
        let total_bytes: u64 = add_operations
            .iter()
            .filter_map(|op| match op {
                CommitOperation::Add { source, .. } => Some(source_size(source)),
                _ => None,
            })
            .sum();
        params.progress.emit(UploadEvent::Start {
            total_files,
            total_bytes,
        });

        let aggregator = params
            .progress
            .clone()
            .map(|inner| Arc::new(AggregatingProgress::new(inner, total_bytes)));

        // 5. Coordinator/committer channel: capacity 1 => one batch in flight.
        let (tx, rx) = mpsc::channel::<CommitJob>(1);
        let commit_size = Arc::new(Mutex::new(AdaptiveCommitSize::new()));

        let committer = tokio::spawn(committer_loop(
            self.clone(),
            rx,
            commit_revision,
            commit_message,
            params.commit_description.clone(),
            params.progress.clone(),
            Arc::clone(&commit_size),
        ));

        // Coordinator runs on this task; dropping tx afterwards ends the committer loop.
        // xet uploads and the base-tree listing use the base `revision`; only the commit
        // target differs when uploading to a PR.
        let coord_result = self
            .coordinate(&add_operations, &delete_operations, &revision, &aggregator, &commit_size, &tx)
            .await;
        drop(tx);

        let committer_result = committer
            .await
            .map_err(|e| HFError::Other(format!("multi-commit committer task panicked: {e}")))?;
        // A committer error is the root cause; the coordinator's `send` failure is only a
        // consequence of the committer dropping its receiver, so surface the committer
        // error first.
        let mut info = match committer_result {
            Err(e) => {
                if let Some(n) = pr_num {
                    return Err(HFError::Other(format!(
                        "multi-commit upload to pull request #{n} failed; resume with \
                         revision=\"refs/pr/{n}\": {e}"
                    )));
                }
                return Err(e);
            },
            Ok(maybe_info) => {
                coord_result?;
                maybe_info.ok_or_else(|| HFError::Other("multi-commit upload produced no commits".to_string()))?
            },
        };
        // Commit responses don't carry PR info; attach what we learned when opening the PR.
        if pr_url.is_some() {
            info.pr_url = pr_url;
            info.pr_num = pr_num;
        }
        params.progress.emit(UploadEvent::Complete);
        Ok(info)
    }

    /// Open a draft pull request and return `(num, web_url)`. Each multi-commit batch is
    /// then committed to `refs/pr/{num}`. Mirrors `huggingface_hub.create_pull_request`:
    /// `POST /api/{type}s/{repo}/discussions` with `pullRequest: true`.
    async fn create_draft_pull_request(&self, title: &str, description: Option<&str>) -> HFResult<(u64, String)> {
        let repo_path = self.repo_path();
        let url = format!("{}/discussions", self.hf_client.api_url(self.repo_type.plural(), &repo_path));
        let body = serde_json::json!({
            "title": title,
            "description": description.unwrap_or("Pull request opened with the hf-hub Rust library"),
            "pullRequest": true,
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
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;

        #[derive(serde::Deserialize)]
        struct CreateDiscussionResponse {
            num: u64,
        }
        let parsed: CreateDiscussionResponse = response.json().await?;
        let pr_url = format!(
            "{}/{}{}/discussions/{}",
            self.hf_client.endpoint(),
            self.repo_type.url_prefix(),
            repo_path,
            parsed.num
        );
        Ok((parsed.num, pr_url))
    }

    async fn coordinate(
        &self,
        add_operations: &[CommitOperation],
        delete_operations: &[CommitOperation],
        revision: &str,
        aggregator: &Option<Arc<AggregatingProgress>>,
        commit_size: &Arc<Mutex<AdaptiveCommitSize>>,
        tx: &mpsc::Sender<CommitJob>,
    ) -> HFResult<()> {
        let mut batch = BatchAccumulator::new();
        let mut first_batch = true;
        let mut idx = 0usize;
        while idx < add_operations.len() {
            let end = (idx + PREUPLOAD_BATCH_SIZE).min(add_operations.len());
            self.classify_chunk_into(&add_operations[idx..end], revision, &mut batch)
                .await?;
            idx = end;

            let max_files = commit_size.lock().unwrap().current();
            if batch.should_flush(max_files, Instant::now()) {
                self.flush_batch(&mut batch, &mut first_batch, delete_operations, revision, aggregator, tx)
                    .await?;
            }
        }
        // Final flush: remaining adds, and/or pending deletes when nothing else forced a commit.
        if !batch.is_empty() || first_batch {
            self.flush_batch(&mut batch, &mut first_batch, delete_operations, revision, aggregator, tx)
                .await?;
        }
        Ok(())
    }

    /// Classify a chunk of adds via the preupload endpoint and push them to `batch`.
    async fn classify_chunk_into(
        &self,
        chunk: &[CommitOperation],
        revision: &str,
        batch: &mut BatchAccumulator,
    ) -> HFResult<()> {
        // Compute size + preupload sample + sha256 in one pass per source (prepare_source),
        // reusing the same single-read helper the single-commit path uses.
        let mut infos: Vec<(String, u64, Vec<u8>, String, AddSource)> = Vec::with_capacity(chunk.len());
        for op in chunk {
            if let CommitOperation::Add { path_in_repo, source } = op {
                let (size, sample, oid) = prepare_source(source).await?;
                infos.push((path_in_repo.clone(), size, sample, oid, source.clone()));
            }
        }
        let files: Vec<(&str, u64, &[u8])> = infos
            .iter()
            .map(|(p, s, sample, _, _)| (p.as_str(), *s, sample.as_slice()))
            .collect();
        let modes = self
            .fetch_upload_modes(&self.repo_path(), self.repo_type.plural(), revision, &files)
            .await?;
        for (path, size, _, oid, source) in infos {
            let is_lfs = size > 0 && modes.get(&path).map(|m| m == "lfs").unwrap_or(false);
            if is_lfs {
                batch.push(PreparedAdd::Xet {
                    path_in_repo: path,
                    source,
                    size,
                    oid,
                });
            } else {
                batch.push(PreparedAdd::Regular {
                    op: CommitOperation::Add {
                        path_in_repo: path,
                        source,
                    },
                    size,
                });
            }
        }
        Ok(())
    }

    /// Spawn the batch's xet upload and hand the job to the committer.
    async fn flush_batch(
        &self,
        batch: &mut BatchAccumulator,
        first_batch: &mut bool,
        delete_operations: &[CommitOperation],
        revision: &str,
        aggregator: &Option<Arc<AggregatingProgress>>,
        tx: &mpsc::Sender<CommitJob>,
    ) -> HFResult<()> {
        let drained = std::mem::replace(batch, BatchAccumulator::new());

        let mut operations: Vec<CommitOperation> = Vec::with_capacity(drained.adds.len() + delete_operations.len());
        if *first_batch {
            operations.extend(delete_operations.iter().cloned());
        }
        *first_batch = false;

        let mut xet_files: Vec<(String, AddSource)> = Vec::new();
        // (path, oid, size) for the committed lfsFile entries; oid was computed during
        // classification, so the upload task never re-reads the source to hash it.
        let mut xet_meta: Vec<(String, String, u64)> = Vec::new();
        let mut batch_content_bytes: u64 = 0;
        for add in drained.adds {
            match add {
                PreparedAdd::Regular { op, size } => {
                    batch_content_bytes += size;
                    operations.push(op);
                },
                PreparedAdd::Xet {
                    path_in_repo,
                    source,
                    size,
                    oid,
                } => {
                    batch_content_bytes += size;
                    // Keep the real source: post_commit routes it to `lfsFile` via the
                    // uploaded map, but if it were missing it inlines real content, not empty.
                    operations.push(CommitOperation::Add {
                        path_in_repo: path_in_repo.clone(),
                        source: source.clone(),
                    });
                    xet_files.push((path_in_repo.clone(), source));
                    xet_meta.push((path_in_repo, oid, size));
                },
            }
        }

        let repo = self.clone();
        let revision = revision.to_string();
        let agg = aggregator.clone();
        let upload = tokio::spawn(async move {
            if xet_files.is_empty() {
                return Ok::<HashMap<String, (String, u64)>, HFError>(HashMap::new());
            }
            let batch_progress: Option<Progress> =
                agg.as_ref().map(|a| Progress::from(Arc::clone(a) as Arc<dyn ProgressHandler>));
            repo.xet_upload(&xet_files, &revision, &batch_progress).await?;
            if let Some(a) = agg.as_ref() {
                a.finish_batch(batch_content_bytes, 0);
            }
            let lfs_uploaded: HashMap<String, (String, u64)> =
                xet_meta.into_iter().map(|(path, oid, size)| (path, (oid, size))).collect();
            Ok(lfs_uploaded)
        });

        tx.send(CommitJob { operations, upload })
            .await
            .map_err(|_| HFError::Other("multi-commit committer stopped early".to_string()))?;
        Ok(())
    }
}

/// Sequentially commits batches received from the coordinator, chaining commits and
/// feeding commit durations back into the shared adaptive sizer.
async fn committer_loop<T: RepoType>(
    repo: HFRepository<T>,
    mut rx: mpsc::Receiver<CommitJob>,
    commit_revision: String,
    commit_message: String,
    commit_description: Option<String>,
    progress: Option<Progress>,
    commit_size: Arc<Mutex<AdaptiveCommitSize>>,
) -> HFResult<Option<CommitInfo>> {
    let mut state = CommitState::new();
    let mut last_info: Option<CommitInfo> = None;
    let mut commit_index = 0usize;
    while let Some(job) = rx.recv().await {
        let lfs_uploaded = job
            .upload
            .await
            .map_err(|e| HFError::Other(format!("xet upload task panicked: {e}")))??;
        let parent = state.next_parent();
        let started = Instant::now();
        let info = repo
            .post_commit(
                &job.operations,
                &lfs_uploaded,
                &commit_message,
                commit_description.as_deref(),
                parent.as_deref(),
                &commit_revision,
                false,
            )
            .await?;
        commit_size.lock().unwrap().record_commit(started.elapsed());
        state.record(&info);
        progress.emit(UploadEvent::CommitCompleted {
            commit_index,
            commit_oid: info.commit_oid.clone(),
        });
        commit_index += 1;
        last_info = Some(info);
    }
    // `None` means the coordinator ended the stream without any commit (e.g. it failed
    // early); that is not itself an error here — the caller decides based on coord_result.
    Ok(last_info)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    // --- AdaptiveCommitSize ---

    #[test]
    fn sizer_starts_at_250() {
        assert_eq!(AdaptiveCommitSize::new().current(), 250);
    }

    #[test]
    fn sizer_grows_after_fast_commit() {
        let mut s = AdaptiveCommitSize::new();
        s.record_commit(Duration::from_secs(5));
        assert_eq!(s.current(), 400);
    }

    #[test]
    fn sizer_shrinks_after_slow_commit() {
        let mut s = AdaptiveCommitSize::new();
        s.record_commit(Duration::from_secs(120));
        assert_eq!(s.current(), 200);
    }

    #[test]
    fn sizer_clamps_at_top() {
        let mut s = AdaptiveCommitSize::new();
        for _ in 0..20 {
            s.record_commit(Duration::from_secs(1));
        }
        assert_eq!(s.current(), 1000);
    }

    #[test]
    fn sizer_clamps_at_bottom_after_slow_commits() {
        let mut s = AdaptiveCommitSize::new();
        for _ in 0..20 {
            s.record_commit(Duration::from_secs(120));
        }
        assert_eq!(s.current(), 20);
    }

    // --- CommitState ---

    fn commit_info(oid: &str) -> CommitInfo {
        CommitInfo {
            commit_url: None,
            commit_message: None,
            commit_description: None,
            commit_oid: Some(oid.to_string()),
            pr_url: None,
            pr_num: None,
        }
    }

    #[test]
    fn commit_state_first_has_no_parent_then_chains_oids() {
        let mut s = CommitState::new();
        assert_eq!(s.next_parent(), None);
        s.record(&commit_info("sha1"));
        assert_eq!(s.next_parent().as_deref(), Some("sha1"));
        s.record(&commit_info("sha2"));
        assert_eq!(s.next_parent().as_deref(), Some("sha2"));
    }

    // --- BatchAccumulator ---

    fn regular(size: u64) -> PreparedAdd {
        PreparedAdd::Regular {
            op: CommitOperation::add_bytes("f", vec![0u8; size as usize]),
            size,
        }
    }

    #[test]
    fn batch_flushes_on_file_count() {
        let mut b = BatchAccumulator::new();
        for _ in 0..5 {
            b.push(regular(1));
        }
        assert!(b.should_flush(5, Instant::now()));
        assert!(!b.should_flush(6, Instant::now()));
    }

    #[test]
    fn batch_flushes_on_regular_byte_budget() {
        let mut b = BatchAccumulator::new();
        b.push(regular(REGULAR_CONTENT_BYTES_BUDGET));
        assert!(b.should_flush(10_000, Instant::now()));
    }

    #[test]
    fn batch_flushes_on_age() {
        let mut b = BatchAccumulator::new();
        b.push(regular(1));
        let future = Instant::now() + MAX_COMMIT_INTERVAL + Duration::from_secs(1);
        assert!(b.should_flush(10_000, future));
    }

    #[test]
    fn batch_empty_never_flushes() {
        let b = BatchAccumulator::new();
        assert!(!b.should_flush(1, Instant::now() + Duration::from_secs(3600)));
    }

    // --- AggregatingProgress ---

    #[derive(Default)]
    struct Capture(Mutex<Vec<(u64, u64)>>); // (bytes_completed, total_bytes)
    impl ProgressHandler for Capture {
        fn on_progress(&self, event: &ProgressEvent) {
            if let ProgressEvent::Upload(UploadEvent::Progress {
                bytes_completed,
                total_bytes,
                ..
            }) = event
            {
                self.0.lock().unwrap().push((*bytes_completed, *total_bytes));
            }
        }
    }

    fn progress_event(done: u64) -> ProgressEvent {
        ProgressEvent::Upload(UploadEvent::Progress {
            bytes_completed: done,
            total_bytes: 50,
            bytes_per_sec: None,
            transfer_bytes_completed: done,
            transfer_bytes: 50,
            transfer_bytes_per_sec: None,
            files: vec![],
        })
    }

    #[test]
    fn aggregating_rebases_onto_grand_total() {
        let cap = Arc::new(Capture::default());
        let agg = AggregatingProgress::new(Progress::from(cap.clone() as Arc<dyn ProgressHandler>), 100);
        agg.on_progress(&progress_event(10)); // batch 1: 10/50 -> 10/100
        agg.finish_batch(50, 50);
        agg.on_progress(&progress_event(20)); // batch 2: 20/50 -> 70/100
        let seen = cap.0.lock().unwrap().clone();
        assert_eq!(seen, vec![(10, 100), (70, 100)]);
    }

    #[test]
    fn aggregating_swallows_start_and_complete() {
        let cap = Arc::new(Capture::default());
        let agg = AggregatingProgress::new(Progress::from(cap.clone() as Arc<dyn ProgressHandler>), 100);
        agg.on_progress(&ProgressEvent::Upload(UploadEvent::Start {
            total_files: 3,
            total_bytes: 100,
        }));
        agg.on_progress(&ProgressEvent::Upload(UploadEvent::Complete));
        assert!(cap.0.lock().unwrap().is_empty());
    }
}
