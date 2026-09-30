//! Streamed multi-commit pipeline backing [`HFRepository::upload_operations`] and
//! [`HFRepository::upload_folder`].
//!
//! Mirrors `huggingface_hub`'s upload pipeline: add operations are pulled from a stream,
//! classified via the `preupload` endpoint in chunks, and grouped into adaptively-sized batches. A coordinator uploads
//! each batch's LFS content via xet while a committer commits the previous batch, so transfer and commit
//! round-trips overlap. A small folder lands as a single commit; a large one as several commits. With
//! `create_pr`, the pull request is opened right before the first commit and every commit targets its
//! `refs/pr/N` ref.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use futures::channel::mpsc;
use futures::{FutureExt, SinkExt, StreamExt};
use serde::Deserialize;

use super::{CreateCommitParams, UploadOperationsParams, prepare_source};
use crate::constants;
use crate::error::{HFError, HFResult, NotFoundContext};
use crate::progress::{EmitEvent, Progress, ProgressEvent, ProgressHandler, UploadEvent};
use crate::repository::files::matches_any_glob;
use crate::repository::{
    AddSource, CommitInfo, CommitOperation, CommitOperationStream, HFRepository, RepoTreeEntry, RepoType,
};

/// Files classified per `preupload` call.
const PREUPLOAD_BATCH_SIZE: usize = 256;
/// Files-per-commit ladder; grows after fast full commits, shrinks after slow or failed ones.
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
    fn new(initial_index: usize) -> Self {
        Self {
            index: AtomicUsize::new(initial_index.min(COMMIT_SIZE_SCALE.len() - 1)),
        }
    }

    fn current(&self) -> usize {
        COMMIT_SIZE_SCALE[self.index.load(Ordering::Relaxed)]
    }

    /// Only a fast commit that filled the current target grows it, so small tail batches do not
    /// inflate the target.
    fn record_success(&self, duration: Duration, files: usize) {
        let index = self.index.load(Ordering::Relaxed);
        let next = if duration < TARGET_COMMIT_DURATION && files >= COMMIT_SIZE_SCALE[index] {
            (index + 1).min(COMMIT_SIZE_SCALE.len() - 1)
        } else if duration > TARGET_COMMIT_DURATION {
            index.saturating_sub(1)
        } else {
            index
        };
        self.index.store(next, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        let index = self.index.load(Ordering::Relaxed);
        self.index.store(index.saturating_sub(1), Ordering::Relaxed);
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
        if self.adds.is_empty() {
            self.started_at = Instant::now();
        }
        if !add.lfs {
            self.regular_bytes += add.size;
        }
        self.adds.push(add);
    }

    fn should_flush(&self, max_files: usize) -> bool {
        !self.adds.is_empty() && (self.adds.len() >= max_files || self.regular_bytes >= REGULAR_CONTENT_BYTES_BUDGET)
    }

    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::new())
    }
}

/// Adds `add` to `batch` and hands back the batch when its size limits are reached. Anything pushed
/// afterwards starts a fresh batch.
fn push_and_take_ready(batch: &mut BatchAccumulator, add: PreparedAdd, max_files: usize) -> Option<BatchAccumulator> {
    batch.push(add);
    batch.should_flush(max_files).then(|| batch.take())
}

/// Next stream item, or `None` if `deadline` passes first.
async fn next_before<S: futures::Stream + Unpin>(stream: &mut S, deadline: Option<Instant>) -> Option<Option<S::Item>> {
    let Some(deadline) = deadline else {
        return Some(stream.next().await);
    };
    let timer = std::pin::pin!(tokio::time::sleep(deadline.saturating_duration_since(Instant::now())));
    match futures::future::select(stream.next(), timer).await {
        futures::future::Either::Left((item, _)) => Some(item),
        futures::future::Either::Right(_) => None,
    }
}

fn duplicate_paths<'a>(paths: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    for path in paths {
        if !seen.insert(path) && !duplicates.contains(&path) {
            duplicates.push(path);
        }
    }
    duplicates
}

/// Tracks the target ref and commit count across the sequential commits.
struct CommitState {
    revision: String,
    parent_commit: Option<String>,
    pr_num: Option<u64>,
    committed: usize,
}

impl CommitState {
    fn new(revision: String, parent_commit: Option<String>) -> Self {
        Self {
            revision,
            parent_commit,
            pr_num: None,
            committed: 0,
        }
    }

    fn record_pr(&mut self, pr_num: u64) {
        self.revision = format!("refs/pr/{pr_num}");
        self.pr_num = Some(pr_num);
    }

    /// The expected parent only guards the first commit; later ones build on the upload's own commits.
    fn parent_for_next_commit(&self) -> Option<&str> {
        if self.committed == 0 {
            self.parent_commit.as_deref()
        } else {
            None
        }
    }

    fn message_for_next_commit<'a>(&self, commit_message: &'a str) -> Cow<'a, str> {
        if self.committed == 0 {
            Cow::Borrowed(commit_message)
        } else {
            Cow::Owned(format!("{commit_message} (part {})", self.committed + 1))
        }
    }
}

#[derive(Deserialize)]
struct CreatedDiscussion {
    num: u64,
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
    adds: Vec<CommitOperation>,
    lfs_uploaded: HashMap<String, (String, u64)>,
    is_last: bool,
}

/// Settings shared by every batch of one upload.
struct PipelineContext<'a> {
    revision: &'a str,
    create_pr: bool,
    commit_size: &'a AdaptiveCommitSize,
    aggregator: Option<Arc<AggregatingProgress>>,
    opened_pr: OnceLock<u64>,
    max_commit_interval: Duration,
    committing_emitted: AtomicBool,
}

impl<T: RepoType> HFRepository<T> {
    pub(super) async fn upload_operations_pipeline(
        &self,
        operations: CommitOperationStream,
        params: UploadOperationsParams,
    ) -> HFResult<CommitInfo> {
        self.upload_operations_pipeline_tuned(operations, params, MAX_COMMIT_INTERVAL, INITIAL_COMMIT_SIZE_INDEX)
            .await
    }

    async fn upload_operations_pipeline_tuned(
        &self,
        operations: CommitOperationStream,
        params: UploadOperationsParams,
        max_commit_interval: Duration,
        initial_commit_size_index: usize,
    ) -> HFResult<CommitInfo> {
        if params.create_pr
            && let Some(revision) = params.revision.as_deref()
            && revision != constants::DEFAULT_REVISION
        {
            return Err(HFError::InvalidParameter(format!(
                "cannot use create_pr=true with revision={revision:?}: pull requests created by upload_operations and \
                 upload_folder are always opened against the default branch; don't set revision when create_pr=true"
            )));
        }
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

        let commit_size = AdaptiveCommitSize::new(initial_commit_size_index);
        let context = PipelineContext {
            revision,
            create_pr: params.create_pr,
            commit_size: &commit_size,
            aggregator: params.progress.clone().map(|inner| Arc::new(AggregatingProgress::new(inner))),
            opened_pr: OnceLock::new(),
            max_commit_interval,
            committing_emitted: AtomicBool::new(false),
        };

        // Zero buffer: one ready batch may wait while the next one uploads.
        let (job_sender, job_receiver) = mpsc::channel::<CommitJob>(0);
        let coordinator = std::pin::pin!(self.coordinate(operations, &context, job_sender));
        let committer = std::pin::pin!(self.commit_batches(
            job_receiver,
            &context,
            delete_operations,
            CommitState::new(revision.to_string(), params.parent_commit),
            commit_message,
            params.commit_description.as_deref(),
            &params.progress,
        ));
        // A committer failure stops the coordinator right away. A coordinator failure drops its job
        // sender, so the committer still lands the batches already handed to it before exiting.
        let result = match futures::future::select(coordinator, committer).await {
            futures::future::Either::Left((Ok(()), committer)) => committer.await,
            futures::future::Either::Left((Err(err), committer)) => {
                if let Err(commit_err) = committer.await {
                    tracing::warn!(error = %commit_err, "committing already-uploaded batches failed after upload error");
                }
                Err(err)
            },
            futures::future::Either::Right((Ok(info), coordinator)) => coordinator.await.map(|()| info),
            futures::future::Either::Right((Err(err), _)) => Err(err),
        };
        let info = match result {
            Ok(info) => info,
            Err(err) => {
                if let Some(&pr_num) = context.opened_pr.get() {
                    tracing::warn!(
                        pr_num,
                        pr_url = %self.pr_url(pr_num),
                        resume_revision = %format!("refs/pr/{pr_num}"),
                        error = %err,
                        "upload to pull request did not complete; to resume into the same pull request, re-run with \
                         revision=\"refs/pr/N\" and without create_pr (create_pr=true would open a new pull request)"
                    );
                }
                return Err(err);
            },
        };

        if !context.committing_emitted.load(Ordering::Relaxed) {
            params.progress.emit(UploadEvent::Committing);
        }
        params.progress.emit(UploadEvent::Complete);
        Ok(info)
    }

    async fn coordinate(
        &self,
        mut operations: CommitOperationStream,
        context: &PipelineContext<'_>,
        mut job_sender: mpsc::Sender<CommitJob>,
    ) -> HFResult<()> {
        let mut batch = BatchAccumulator::new();
        let mut chunk = Vec::with_capacity(PREUPLOAD_BATCH_SIZE);
        let mut chunk_started_at = None;
        let mut sent_any = false;
        loop {
            let mut deadline_passed = false;
            let mut stream_done = false;
            let mut target_reached = false;
            while chunk.len() < PREUPLOAD_BATCH_SIZE {
                if batch.adds.len() + chunk.len() >= context.commit_size.current() {
                    target_reached = true;
                    break;
                }
                let oldest_pending = match (batch.adds.is_empty(), chunk_started_at) {
                    (false, Some(chunk_started)) => Some(batch.started_at.min(chunk_started)),
                    (false, None) => Some(batch.started_at),
                    (true, chunk_started) => chunk_started,
                };
                let deadline = oldest_pending.map(|started| started + context.max_commit_interval);
                match next_before(&mut operations, deadline).await {
                    Some(Some(operation)) => {
                        chunk.push(require_add(operation?)?);
                        chunk_started_at.get_or_insert_with(Instant::now);
                    },
                    Some(None) => {
                        stream_done = true;
                        break;
                    },
                    None => {
                        deadline_passed = true;
                        break;
                    },
                }
            }
            // Learning about EOF now lets the batch about to flush be marked as the last one.
            if target_reached && chunk.len() < PREUPLOAD_BATCH_SIZE {
                match operations.next().now_or_never() {
                    Some(Some(operation)) => chunk.push(require_add(operation?)?),
                    Some(None) => stream_done = true,
                    None => {},
                }
            }
            let prepared = self.classify_chunk(&chunk, context).await?;
            chunk.clear();
            chunk_started_at = None;
            let prepared_count = prepared.len();
            for (index, add) in prepared.into_iter().enumerate() {
                if let Some(ready) = push_and_take_ready(&mut batch, add, context.commit_size.current()) {
                    let is_last = stream_done && index + 1 == prepared_count;
                    let job = self.upload_batch(ready, context, is_last).await?;
                    send_job(&mut job_sender, job).await?;
                    sent_any = true;
                }
            }
            if stream_done {
                let job = if !batch.adds.is_empty() {
                    self.upload_batch(batch, context, true).await?
                } else if !sent_any {
                    // Empty stream: the committer still owes a deletes-only commit, if any.
                    CommitJob {
                        adds: Vec::new(),
                        lfs_uploaded: HashMap::new(),
                        is_last: true,
                    }
                } else {
                    return Ok(());
                };
                return send_job(&mut job_sender, job).await;
            }
            let batch_expired = batch.started_at.elapsed() >= context.max_commit_interval;
            if !batch.adds.is_empty()
                && (deadline_passed || batch_expired || batch.should_flush(context.commit_size.current()))
            {
                let job = self.upload_batch(batch.take(), context, false).await?;
                send_job(&mut job_sender, job).await?;
                sent_any = true;
            }
        }
    }

    async fn classify_chunk(
        &self,
        chunk: &[(String, AddSource)],
        context: &PipelineContext<'_>,
    ) -> HFResult<Vec<PreparedAdd>> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }
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
        Ok(prepared
            .into_iter()
            .map(|(path_in_repo, source, size, _, sha256)| {
                let lfs = size > 0 && upload_modes.get(&path_in_repo).is_some_and(|mode| mode == "lfs");
                PreparedAdd {
                    path_in_repo,
                    source,
                    size,
                    sha256,
                    lfs,
                }
            })
            .collect())
    }

    /// Upload a batch's LFS files via xet and assemble its commit job. When the Hub does not
    /// offer xet transfer, LFS files fall back to inline upload, matching `create_commit`.
    async fn upload_batch(
        &self,
        batch: BatchAccumulator,
        context: &PipelineContext<'_>,
        is_last: bool,
    ) -> HFResult<CommitJob> {
        let duplicates = duplicate_paths(batch.adds.iter().map(|add| add.path_in_repo.as_str()));
        if !duplicates.is_empty() {
            tracing::warn!(
                ?duplicates,
                "about to commit several add operations for the same path_in_repo; only the last one is kept"
            );
        }

        let mut adds = Vec::with_capacity(batch.adds.len());
        let mut xet_files: Vec<(String, AddSource)> = Vec::new();
        let mut lfs_uploaded: HashMap<String, (String, u64)> = HashMap::new();
        let mut batch_content_bytes = 0u64;
        for add in batch.adds {
            batch_content_bytes += add.size;
            if add.lfs {
                xet_files.push((add.path_in_repo.clone(), add.source.clone()));
                lfs_uploaded.insert(add.path_in_repo.clone(), (add.sha256, add.size));
            }
            adds.push(CommitOperation::Add {
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
            adds,
            lfs_uploaded,
            is_last,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn commit_batches(
        &self,
        mut job_receiver: mpsc::Receiver<CommitJob>,
        context: &PipelineContext<'_>,
        mut pending_deletes: Vec<CommitOperation>,
        mut state: CommitState,
        commit_message: &str,
        commit_description: Option<&str>,
        progress: &Option<Progress>,
    ) -> HFResult<CommitInfo> {
        let mut last_info: Option<CommitInfo> = None;
        while let Some(job) = job_receiver.next().await {
            if job.is_last && !context.committing_emitted.swap(true, Ordering::Relaxed) {
                progress.emit(UploadEvent::Committing);
            }
            let mut pending_pieces = VecDeque::from([job.adds]);
            while let Some(adds) = pending_pieces.pop_front() {
                let delete_count = if state.committed == 0 { pending_deletes.len() } else { 0 };
                if adds.is_empty() && delete_count == 0 {
                    continue;
                }
                if context.create_pr && state.pr_num.is_none() {
                    // Opened explicitly (and lazily, so an empty upload opens none) rather than via
                    // `?create_pr=1`, so a retried commit can never open a second pull request.
                    let pr_num = self.create_pull_request(commit_message, commit_description).await?;
                    let _ = context.opened_pr.set(pr_num);
                    state.record_pr(pr_num);
                }

                let mut operations = pending_deletes[..delete_count].to_vec();
                operations.extend(adds);
                let add_count = operations.len() - delete_count;
                let started_at = Instant::now();
                let result = self
                    .create_commit_impl(CreateCommitParams {
                        operations: operations.clone(),
                        commit_message: state.message_for_next_commit(commit_message).into_owned(),
                        commit_description: commit_description.map(str::to_owned),
                        revision: Some(state.revision.clone()),
                        create_pr: false,
                        parent_commit: state.parent_for_next_commit().map(str::to_owned),
                        progress: None,
                        pre_uploaded_lfs: Some(job.lfs_uploaded.clone()),
                    })
                    .await;
                let info = match result {
                    Ok(info) => info,
                    Err(err) => {
                        context.commit_size.record_failure();
                        let adds = operations.split_off(delete_count);
                        if adds.len() <= COMMIT_SIZE_SCALE[0] {
                            return Err(err);
                        }
                        tracing::warn!(
                            files = adds.len(),
                            retry_commit_size = context.commit_size.current(),
                            error = %err,
                            "commit failed; retrying in smaller chunks"
                        );
                        for piece in split_into_pieces(adds, context.commit_size.current()).into_iter().rev() {
                            pending_pieces.push_front(piece);
                        }
                        continue;
                    },
                };
                context.commit_size.record_success(started_at.elapsed(), add_count);
                if delete_count > 0 {
                    pending_deletes.clear();
                }
                state.committed += 1;
                tracing::info!(
                    commit_index = state.committed - 1,
                    commit_oid = info.commit_oid.as_deref(),
                    files = operations.len(),
                    "upload batch committed"
                );
                progress.emit(UploadEvent::CommitCompleted {
                    commit_index: state.committed - 1,
                    commit_oid: info.commit_oid.clone(),
                });
                last_info = Some(info);
            }
        }

        let Some(mut info) = last_info else {
            return self.head_commit_info(&state.revision, commit_message).await;
        };
        if let Some(pr_num) = state.pr_num {
            info.pr_num = Some(pr_num);
            info.pr_url = Some(self.pr_url(pr_num));
        }
        Ok(info)
    }

    /// Nothing was committed: mirror `create_commit` and describe the revision's current head.
    async fn head_commit_info(&self, revision: &str, commit_message: &str) -> HFResult<CommitInfo> {
        tracing::warn!("no files to upload; skipping to prevent an empty commit");
        let stream = self.list_commits().revision(revision.to_string()).limit(1).send()?;
        futures::pin_mut!(stream);
        let head = stream.next().await.transpose()?.map(|commit| commit.id);
        Ok(CommitInfo {
            commit_url: head.as_ref().map(|sha| {
                format!(
                    "{}/{}{}/commit/{sha}",
                    self.hf_client.endpoint(),
                    self.repo_type.url_prefix(),
                    self.repo_path()
                )
            }),
            commit_message: Some(commit_message.to_string()),
            commit_description: None,
            commit_oid: head,
            pr_url: None,
            pr_num: None,
        })
    }

    async fn create_pull_request(&self, title: &str, description: Option<&str>) -> HFResult<u64> {
        let url = format!("{}/discussions", self.hf_client.api_url(self.repo_type.plural(), &self.repo_path()));
        let body = serde_json::json!({
            "title": title.trim(),
            "description": description.unwrap_or(""),
            "pullRequest": true,
        });
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), NotFoundContext::Repo)
            .await?;
        let created: CreatedDiscussion = response.json().await?;
        tracing::info!(pr_num = created.num, "opened pull request for upload");
        Ok(created.num)
    }

    fn pr_url(&self, pr_num: u64) -> String {
        format!(
            "{}/{}{}/discussions/{pr_num}",
            self.hf_client.endpoint(),
            self.repo_type.url_prefix(),
            self.repo_path()
        )
    }
}

fn split_into_pieces(mut operations: Vec<CommitOperation>, piece_size: usize) -> Vec<Vec<CommitOperation>> {
    let mut pieces = Vec::with_capacity(operations.len().div_ceil(piece_size));
    while !operations.is_empty() {
        let rest = operations.split_off(piece_size.min(operations.len()));
        pieces.push(std::mem::replace(&mut operations, rest));
    }
    pieces
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
        assert_eq!(AdaptiveCommitSize::new(INITIAL_COMMIT_SIZE_INDEX).current(), 250);
    }

    #[test]
    fn commit_size_grows_after_fast_full_commit_and_shrinks_after_slow_commit() {
        let size = AdaptiveCommitSize::new(INITIAL_COMMIT_SIZE_INDEX);
        size.record_success(Duration::from_secs(5), 250);
        assert_eq!(size.current(), 400);
        size.record_success(Duration::from_secs(120), 400);
        size.record_success(Duration::from_secs(120), 250);
        assert_eq!(size.current(), 200);
    }

    #[test]
    fn commit_size_does_not_grow_after_fast_partial_commit() {
        let size = AdaptiveCommitSize::new(INITIAL_COMMIT_SIZE_INDEX);
        size.record_success(Duration::from_secs(5), 249);
        assert_eq!(size.current(), 250);
    }

    #[test]
    fn commit_size_shrinks_on_failure() {
        let size = AdaptiveCommitSize::new(INITIAL_COMMIT_SIZE_INDEX);
        size.record_failure();
        assert_eq!(size.current(), 200);
    }

    #[test]
    fn commit_size_is_clamped_to_scale() {
        let size = AdaptiveCommitSize::new(INITIAL_COMMIT_SIZE_INDEX);
        for _ in 0..20 {
            size.record_success(Duration::from_secs(1), 1000);
        }
        assert_eq!(size.current(), 1000);
        for _ in 0..20 {
            size.record_failure();
        }
        assert_eq!(size.current(), 20);
    }

    fn prepared_at(path: &str, size: u64, lfs: bool) -> PreparedAdd {
        PreparedAdd {
            path_in_repo: path.to_string(),
            source: AddSource::bytes(Vec::new()),
            size,
            sha256: String::new(),
            lfs,
        }
    }

    fn prepared(size: u64, lfs: bool) -> PreparedAdd {
        prepared_at("f", size, lfs)
    }

    #[test]
    fn batch_flushes_on_file_count() {
        let mut batch = BatchAccumulator::new();
        for _ in 0..5 {
            batch.push(prepared(1, false));
        }
        assert!(batch.should_flush(5));
        assert!(!batch.should_flush(6));
    }

    #[test]
    fn batch_flushes_on_regular_byte_budget_only() {
        let mut batch = BatchAccumulator::new();
        batch.push(prepared(REGULAR_CONTENT_BYTES_BUDGET, true));
        assert!(!batch.should_flush(10_000));
        batch.push(prepared(REGULAR_CONTENT_BYTES_BUDGET, false));
        assert!(batch.should_flush(10_000));
    }

    #[test]
    fn empty_batch_never_flushes() {
        assert!(!BatchAccumulator::new().should_flush(1));
    }

    /// Feeds one preupload chunk through the per-file flush and returns the committed batch sizes.
    fn flush_sizes(adds: Vec<PreparedAdd>, max_files: usize) -> (Vec<usize>, usize) {
        let mut batch = BatchAccumulator::new();
        let sizes = adds
            .into_iter()
            .filter_map(|add| push_and_take_ready(&mut batch, add, max_files))
            .map(|flushed| flushed.adds.len())
            .collect();
        (sizes, batch.adds.len())
    }

    #[test]
    fn flushes_mid_chunk_at_small_target_and_carries_leftovers() {
        let adds = (0..PREUPLOAD_BATCH_SIZE).map(|_| prepared(1, false)).collect();
        let (sizes, leftover) = flush_sizes(adds, 20);
        assert_eq!(sizes, vec![20; PREUPLOAD_BATCH_SIZE / 20]);
        assert_eq!(leftover, PREUPLOAD_BATCH_SIZE % 20);
    }

    #[test]
    fn regular_byte_budget_is_checked_per_file() {
        let half = REGULAR_CONTENT_BYTES_BUDGET / 2;
        let adds = vec![
            prepared(half, false),
            prepared(half, false),
            prepared(half, false),
            prepared(REGULAR_CONTENT_BYTES_BUDGET, true),
            prepared(1, false),
        ];
        let (sizes, leftover) = flush_sizes(adds, 1000);
        assert_eq!(sizes, vec![2]);
        assert_eq!(leftover, 3);
    }

    #[test]
    fn duplicate_paths_are_reported_once() {
        assert_eq!(duplicate_paths(["a", "b", "a", "c", "a", "b"]), vec!["a", "b"]);
        assert!(duplicate_paths(["a", "b"]).is_empty());
    }

    #[test]
    fn split_into_pieces_respects_size() {
        let operations: Vec<CommitOperation> = (0..45).map(|i| CommitOperation::delete(i.to_string())).collect();
        let sizes: Vec<usize> = split_into_pieces(operations, 20).iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![20, 20, 5]);
    }

    #[test]
    fn commit_state_passes_parent_only_on_first_commit() {
        let mut state = CommitState::new("main".to_string(), Some("sha0".to_string()));
        assert_eq!(state.parent_for_next_commit(), Some("sha0"));
        state.committed += 1;
        assert_eq!(state.parent_for_next_commit(), None);
    }

    #[test]
    fn commit_state_titles_later_commits_as_parts() {
        let mut state = CommitState::new("main".to_string(), None);
        assert_eq!(state.message_for_next_commit("Upload folder"), "Upload folder");
        state.committed += 1;
        assert_eq!(state.message_for_next_commit("Upload folder"), "Upload folder (part 2)");
        state.committed += 1;
        assert_eq!(state.message_for_next_commit("Upload folder"), "Upload folder (part 3)");
    }

    #[test]
    fn commit_state_targets_pr_ref_once_opened() {
        let mut state = CommitState::new("main".to_string(), None);
        state.record_pr(7);
        assert_eq!(state.revision, "refs/pr/7");
        assert_eq!(state.pr_num, Some(7));
    }

    /// Minimal Hub stand-in: every commit is rejected with 403, everything else succeeds. Records
    /// each request line.
    async fn spawn_rejecting_hub() -> (String, Arc<Mutex<Vec<String>>>) {
        let (endpoint, requests, _) = spawn_hub(usize::MAX).await;
        (endpoint, requests)
    }

    type Recorded = Arc<Mutex<Vec<String>>>;

    /// Minimal Hub stand-in; the first `rejected_commits` commits get a 403, later ones succeed.
    /// Records each request line, with the number of committed files and deletes appended to
    /// successful commit requests, and every commit request body. The tree lists one `old.txt`.
    async fn spawn_hub(rejected_commits: usize) -> (String, Recorded, Recorded) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let commit_bodies = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let recorded_bodies = commit_bodies.clone();
        let mut remaining_rejections = rejected_commits;
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                let header_end = loop {
                    let read = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..read]);
                    if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let head = String::from_utf8_lossy(&raw[..header_end]).to_string();
                let content_length: usize = head
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().to_string())
                    })
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                while raw.len() < header_end + content_length {
                    let read = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..read]);
                }
                let mut request_line = head.lines().next().unwrap_or_default().to_string();
                let is_commit = request_line.contains("/commit/");
                if is_commit {
                    recorded_bodies
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&raw[header_end..]).to_string());
                }
                let (status, body) = if is_commit && remaining_rejections == 0 {
                    let commit_body = String::from_utf8_lossy(&raw[header_end..]);
                    let files = commit_body.matches(r#""key":"file""#).count();
                    let deletes = commit_body.matches(r#""key":"deletedFile""#).count();
                    request_line.push_str(&format!(" deletes={deletes} files={files}"));
                    ("200 OK", r#"{"commitOid":"abc"}"#)
                } else if is_commit {
                    remaining_rejections = remaining_rejections.saturating_sub(1);
                    ("403 Forbidden", r#"{"error":"no"}"#)
                } else if request_line.contains("/tree/") {
                    ("200 OK", r#"[{"type":"file","oid":"o","size":1,"path":"old.txt"}]"#)
                } else if request_line.contains("/discussions") {
                    ("200 OK", r#"{"num":3}"#)
                } else {
                    ("200 OK", r#"{"files":[]}"#)
                };
                recorded.lock().unwrap().push(request_line);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                tokio::io::AsyncWriteExt::write_all(&mut sock, response.as_bytes())
                    .await
                    .unwrap();
            }
        });
        (format!("http://{addr}"), requests, commit_bodies)
    }

    fn two_file_stream() -> CommitOperationStream {
        Box::pin(futures::stream::iter(
            ["a.txt", "b.txt"].map(|path| Ok(CommitOperation::add_bytes(path, b"x".to_vec()))),
        ))
    }

    #[tokio::test]
    async fn pr_is_opened_before_first_commit_and_commit_error_variant_is_preserved() {
        let (endpoint, requests) = spawn_rejecting_hub().await;
        let client = crate::HFClient::builder().endpoint(endpoint).token("hf_test").build().unwrap();
        let result = client
            .model("owner", "repo")
            .upload_operations()
            .operations(two_file_stream())
            .create_pr(true)
            .send()
            .await;
        assert!(matches!(result, Err(HFError::Forbidden { .. })), "got {result:?}");
        let requests = requests.lock().unwrap().clone();
        let discussion = requests
            .iter()
            .position(|r| r.starts_with("POST /api/models/owner/repo/discussions "));
        let commit = requests.iter().position(|r| r.contains("/commit/"));
        assert!(discussion.is_some() && discussion < commit, "requests: {requests:?}");
        let commit_line = &requests[commit.unwrap()];
        assert!(commit_line.contains("/commit/refs%2Fpr%2F3"), "{commit_line}");
        assert!(!commit_line.contains("create_pr"), "{commit_line}");
    }

    #[tokio::test]
    async fn create_pr_rejects_non_default_revision() {
        let client = crate::HFClient::builder().endpoint("http://127.0.0.1:9").build().unwrap();
        let result = client
            .model("owner", "repo")
            .upload_operations()
            .operations(two_file_stream())
            .revision("dev")
            .create_pr(true)
            .send()
            .await;
        assert!(matches!(result, Err(HFError::InvalidParameter(_))));
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

    fn commit_files(requests: &Mutex<Vec<String>>) -> Vec<usize> {
        requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.contains("/commit/"))
            .filter_map(|r| r.rsplit_once(" files=")?.1.parse().ok())
            .collect()
    }

    async fn wait_for_commits(requests: &Mutex<Vec<String>>, count: usize) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while commit_files(requests).len() < count {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("commit did not happen while the stream was still open");
    }

    fn upload_params() -> UploadOperationsParams {
        UploadOperationsParams {
            revision: None,
            commit_message: None,
            commit_description: None,
            create_pr: false,
            parent_commit: None,
            delete_patterns: None,
            progress: None,
        }
    }

    type OperationSender = mpsc::UnboundedSender<HFResult<CommitOperation>>;

    /// Starts an upload fed by a channel; the returned sender keeps the stream open until dropped.
    async fn start_channel_upload(
        max_commit_interval: Duration,
    ) -> (OperationSender, tokio::task::JoinHandle<HFResult<CommitInfo>>, Arc<Mutex<Vec<String>>>) {
        let (sender, upload, requests, _) =
            start_tuned_upload(max_commit_interval, INITIAL_COMMIT_SIZE_INDEX, 0, upload_params()).await;
        (sender, upload, requests)
    }

    async fn start_tuned_upload(
        max_commit_interval: Duration,
        initial_commit_size_index: usize,
        rejected_commits: usize,
        params: UploadOperationsParams,
    ) -> (OperationSender, tokio::task::JoinHandle<HFResult<CommitInfo>>, Recorded, Recorded) {
        let (endpoint, requests, commit_bodies) = spawn_hub(rejected_commits).await;
        let client = crate::HFClient::builder().endpoint(endpoint).token("hf_test").build().unwrap();
        let (sender, receiver) = mpsc::unbounded();
        let upload = tokio::spawn(async move {
            client
                .model("owner", "repo")
                .upload_operations_pipeline_tuned(
                    Box::pin(receiver),
                    params,
                    max_commit_interval,
                    initial_commit_size_index,
                )
                .await
        });
        (sender, upload, requests, commit_bodies)
    }

    fn add_op(index: usize) -> HFResult<CommitOperation> {
        Ok(CommitOperation::add_bytes(format!("f{index}.txt"), b"x".to_vec()))
    }

    #[tokio::test]
    async fn partial_batch_commits_after_interval_while_stream_stays_open() {
        let (sender, upload, requests) = start_channel_upload(Duration::from_millis(100)).await;
        sender.unbounded_send(add_op(0)).unwrap();
        wait_for_commits(&requests, 1).await;
        assert_eq!(commit_files(&requests), vec![1]);
        drop(sender);
        upload.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn full_batch_commits_without_waiting_for_more_items() {
        let (sender, upload, requests) = start_channel_upload(Duration::from_secs(3600)).await;
        for index in 0..PREUPLOAD_BATCH_SIZE {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        let commit_size = COMMIT_SIZE_SCALE[INITIAL_COMMIT_SIZE_INDEX];
        wait_for_commits(&requests, 1).await;
        assert_eq!(commit_files(&requests), vec![commit_size]);
        drop(sender);
        upload.await.unwrap().unwrap();
        assert_eq!(commit_files(&requests), vec![commit_size, PREUPLOAD_BATCH_SIZE - commit_size]);
    }

    #[tokio::test]
    async fn eof_after_flushed_batch_makes_no_empty_commit() {
        let (sender, upload, requests) = start_channel_upload(Duration::from_millis(100)).await;
        sender.unbounded_send(add_op(0)).unwrap();
        wait_for_commits(&requests, 1).await;
        drop(sender);
        let info = upload.await.unwrap().unwrap();
        assert_eq!(info.commit_oid.as_deref(), Some("abc"));
        assert_eq!(commit_files(&requests), vec![1]);
        assert!(!requests.lock().unwrap().iter().any(|r| r.contains("/commits/")), "{requests:?}");
    }
    fn commit_deletes(requests: &Mutex<Vec<String>>) -> Vec<usize> {
        requests
            .lock()
            .unwrap()
            .iter()
            .filter_map(|r| r.split_once(" deletes=")?.1.split_once(' ')?.0.parse().ok())
            .collect()
    }

    fn commit_header(body: &str) -> serde_json::Value {
        let header: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        header["value"].clone()
    }

    const SMALLEST_COMMIT_SIZE_INDEX: usize = 0;

    #[derive(Default)]
    struct LifecycleProgress(Mutex<Vec<String>>);

    impl ProgressHandler for LifecycleProgress {
        fn on_progress(&self, event: &ProgressEvent) {
            let name = match event {
                ProgressEvent::Upload(UploadEvent::Committing) => "committing",
                ProgressEvent::Upload(UploadEvent::CommitCompleted { .. }) => "commit_completed",
                ProgressEvent::Upload(UploadEvent::Complete) => "complete",
                _ => return,
            };
            self.0.lock().unwrap().push(name.to_string());
        }
    }

    #[tokio::test]
    async fn coordinator_error_still_lands_queued_commit_and_keeps_error_variant() {
        let (sender, upload, requests, _) =
            start_tuned_upload(Duration::from_secs(3600), SMALLEST_COMMIT_SIZE_INDEX, 0, upload_params()).await;
        // One file past the target so the full batch is queued before the stream error is read.
        for index in 0..=COMMIT_SIZE_SCALE[0] {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        sender.unbounded_send(Err(HFError::Other("stream broke".to_string()))).unwrap();
        let result = upload.await.unwrap();
        assert!(matches!(&result, Err(HFError::Other(message)) if message == "stream broke"), "got {result:?}");
        assert_eq!(commit_files(&requests), vec![COMMIT_SIZE_SCALE[0]]);
    }

    #[tokio::test]
    async fn exactly_target_files_emit_committing_once_before_complete() {
        let capture = Arc::new(LifecycleProgress::default());
        let params = UploadOperationsParams {
            progress: Some(Progress::from(capture.clone() as Arc<dyn ProgressHandler>)),
            ..upload_params()
        };
        let (sender, upload, requests, _) =
            start_tuned_upload(Duration::from_secs(3600), INITIAL_COMMIT_SIZE_INDEX, 0, params).await;
        let target = COMMIT_SIZE_SCALE[INITIAL_COMMIT_SIZE_INDEX];
        for index in 0..target {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        drop(sender);
        upload.await.unwrap().unwrap();
        assert_eq!(commit_files(&requests), vec![target]);
        let events = capture.0.lock().unwrap().clone();
        assert_eq!(events, vec!["committing", "commit_completed", "complete"]);
    }

    #[tokio::test]
    async fn batch_at_small_target_commits_without_waiting_for_chunk_or_deadline() {
        let (sender, upload, requests, _) =
            start_tuned_upload(Duration::from_secs(3600), SMALLEST_COMMIT_SIZE_INDEX, 0, upload_params()).await;
        for index in 0..COMMIT_SIZE_SCALE[0] {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        wait_for_commits(&requests, 1).await;
        assert_eq!(commit_files(&requests), vec![COMMIT_SIZE_SCALE[0]]);
        drop(sender);
        upload.await.unwrap().unwrap();
        assert_eq!(commit_files(&requests), vec![COMMIT_SIZE_SCALE[0]]);
    }

    #[tokio::test]
    async fn failed_commit_is_resplit_at_lowered_target_with_deletes_in_first_piece() {
        let params = UploadOperationsParams {
            delete_patterns: Some(vec!["old.txt".to_string()]),
            ..upload_params()
        };
        let initial_index = 2;
        let (sender, upload, requests, _) =
            start_tuned_upload(Duration::from_secs(3600), initial_index, 1, params).await;
        let files = COMMIT_SIZE_SCALE[initial_index];
        for index in 0..files {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        drop(sender);
        upload.await.unwrap().unwrap();
        let lowered = COMMIT_SIZE_SCALE[initial_index - 1];
        assert_eq!(commit_files(&requests), vec![lowered, files - lowered]);
        assert_eq!(commit_deletes(&requests), vec![1, 0]);
    }

    #[tokio::test]
    async fn later_commits_are_numbered_parts_and_only_first_sends_parent() {
        let params = UploadOperationsParams {
            parent_commit: Some("sha0".to_string()),
            commit_message: Some("Add data".to_string()),
            ..upload_params()
        };
        let (sender, upload, requests, commit_bodies) =
            start_tuned_upload(Duration::from_secs(3600), SMALLEST_COMMIT_SIZE_INDEX, 0, params).await;
        for index in 0..COMMIT_SIZE_SCALE[0] {
            sender.unbounded_send(add_op(index)).unwrap();
        }
        wait_for_commits(&requests, 1).await;
        sender.unbounded_send(add_op(COMMIT_SIZE_SCALE[0])).unwrap();
        drop(sender);
        upload.await.unwrap().unwrap();
        let headers: Vec<serde_json::Value> = commit_bodies.lock().unwrap().iter().map(|b| commit_header(b)).collect();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0]["summary"], "Add data");
        assert_eq!(headers[0]["parentCommit"], "sha0");
        assert_eq!(headers[1]["summary"], "Add data (part 2)");
        assert!(headers[1].get("parentCommit").is_none(), "{}", headers[1]);
    }

    #[tokio::test]
    async fn delete_only_upload_makes_one_deletes_only_commit() {
        let params = UploadOperationsParams {
            delete_patterns: Some(vec!["*.txt".to_string()]),
            ..upload_params()
        };
        let (sender, upload, requests, _) =
            start_tuned_upload(Duration::from_secs(3600), INITIAL_COMMIT_SIZE_INDEX, 0, params).await;
        drop(sender);
        upload.await.unwrap().unwrap();
        assert_eq!(commit_files(&requests), vec![0]);
        assert_eq!(commit_deletes(&requests), vec![1]);
    }
}
