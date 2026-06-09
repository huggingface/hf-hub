//! `HFRepository::upload_large_folder`: resumable, xet-optimized upload of a
//! large local folder as a sequence of adaptively-batched commits.

pub mod local_folder;
pub mod pipeline;

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use bon::bon;
use futures::future::FutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::constants;
use crate::error::{HFError, HFResult};
use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::repository::files::{AddSource, matches_any_glob};
use crate::repository::upload::ResolvedAdd;
use crate::repository::upload_large_folder::local_folder::{
    LocalUploadFileMetadata, LocalUploadFilePaths, get_local_upload_paths, read_upload_metadata,
};
use crate::repository::upload_large_folder::pipeline::{
    CommitReady, PREUPLOAD_BATCH_SIZE, PipelineConfig, StatusCounters, TimedBatchBuffer, WorkStage, seed_stage,
    sleep_or_pending,
};
use crate::repository::{CommitInfo, HFRepository, RepoType};

/// Summary returned by [`HFRepository::upload_large_folder`].
#[derive(Debug, Clone, Default)]
pub struct UploadLargeFolderReport {
    /// The commits created, in the order they were committed (one per batch).
    pub commits: Vec<CommitInfo>,
    /// Total files in the upload set after filtering.
    pub total_files: usize,
    /// Files classified as lfs (uploaded via xet). Counts lfs-mode files in the set, including ones skipped on a
    /// resumed run because they were already uploaded.
    pub files_uploaded_lfs: usize,
    /// Files committed inline (regular).
    pub files_committed_inline: usize,
    /// Files the Hub told us to ignore.
    pub files_ignored: usize,
    /// Bytes actually transferred to CAS (post-dedup) during THIS invocation. A resumed run skips already-uploaded
    /// files, so this counts only the current run's transfers.
    pub bytes_uploaded: u64,
    /// Bytes saved by xet deduplication during this invocation.
    pub dedup_bytes_saved: u64,
}

const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    ".git",
    ".git/**",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/**",
    "**/.cache/huggingface/**",
];

/// Recursively collects `(repo_path, absolute_file_path)` for every file under
/// `folder` that survives the default ignores, the user allow/ignore globs, and
/// is then prefixed by `path_in_repo`. Repo paths use `/` separators.
fn discover_files(
    folder: &Path,
    path_in_repo: Option<&str>,
    allow_patterns: &Option<Vec<String>>,
    ignore_patterns: &Option<Vec<String>>,
) -> HFResult<Vec<(String, PathBuf)>> {
    let mut default_and_user_ignores: Vec<String> = DEFAULT_IGNORE_PATTERNS.iter().map(|s| s.to_string()).collect();
    if let Some(user) = ignore_patterns {
        default_and_user_ignores.extend(user.iter().cloned());
    }

    let prefix = path_in_repo.map(|p| p.trim_matches('/').to_string()).filter(|p| !p.is_empty());

    let mut out = Vec::new();
    collect(folder, folder, allow_patterns, &default_and_user_ignores, prefix.as_deref(), &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn collect(
    root: &Path,
    current: &Path,
    allow: &Option<Vec<String>>,
    ignore: &[String],
    prefix: Option<&str>,
    out: &mut Vec<(String, PathBuf)>,
) -> HFResult<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect(root, &path, allow, ignore, prefix, out)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|e| {
                HFError::InvalidParameter(format!("path {} not under {}: {e}", path.display(), root.display()))
            })?;
            let rel: String = relative
                .components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");

            if let Some(allow) = allow
                && !matches_any_glob(allow, &rel)
            {
                continue;
            }
            if matches_any_glob(ignore, &rel) {
                continue;
            }

            let repo_path = match prefix {
                Some(p) => format!("{p}/{rel}"),
                None => rel,
            };
            out.push((repo_path, path));
        }
    }
    Ok(())
}

struct Item {
    paths: LocalUploadFilePaths,
    file_path: PathBuf,
    meta: LocalUploadFileMetadata,
    sample: Vec<u8>,
    size: u64,
}

/// An lfs file awaiting upload, buffered in the producer's staging buffer.
struct StagedLfs {
    idx: usize,
    path_in_repo: String,
    file_path: PathBuf,
    size: u64,
}

/// Result of classifying one file via `/preupload`.
struct ClassifyResult {
    idx: usize,
    upload_mode: String,
    should_ignore: bool,
}

/// Output of one producer work future.
enum WorkOutput {
    Classified(Vec<ClassifyResult>),
    Uploaded {
        entries: Vec<StagedLfs>,
        result: crate::xet::XetBatchResult,
    },
}

impl<T: RepoType> HFRepository<T> {
    /// Classify a chunk of files via `/preupload`. Owns its input; borrows only
    /// `&self` + `revision`. Empty files are forced to "regular".
    async fn classify_chunk(&self, input: Vec<(usize, String, u64, Vec<u8>)>, revision: &str) -> HFResult<WorkOutput> {
        let repo_path = self.repo_path();
        let api_segment = self.repo_type.plural();
        let payload: Vec<(&str, u64, &[u8])> = input
            .iter()
            .map(|(_, path, size, sample)| (path.as_str(), *size, sample.as_slice()))
            .collect();
        let infos = self.fetch_upload_modes(&repo_path, api_segment, revision, &payload).await?;
        let results = input
            .iter()
            .map(|(idx, path, size, _)| {
                let info = infos.get(path);
                let should_ignore = info.map(|i| i.should_ignore).unwrap_or(false);
                let upload_mode = if *size == 0 {
                    "regular".to_string()
                } else {
                    info.map(|i| i.upload_mode.clone()).unwrap_or_else(|| "regular".to_string())
                };
                ClassifyResult {
                    idx: *idx,
                    upload_mode,
                    should_ignore,
                }
            })
            .collect();
        Ok(WorkOutput::Classified(results))
    }

    /// Upload one batch of lfs files as a single xet `UploadCommit`. Suppresses
    /// per-batch progress (passes `&None`); the orchestrator emits aggregate
    /// `LargeFolderStatus` instead.
    async fn upload_chunk(&self, entries: Vec<StagedLfs>, revision: &str) -> HFResult<WorkOutput> {
        let files_in: Vec<(String, AddSource)> = entries
            .iter()
            .map(|e| (e.path_in_repo.clone(), AddSource::File(e.file_path.clone())))
            .collect();
        let result = self.xet_upload_batch(&files_in, revision, &None).await?;
        Ok(WorkOutput::Uploaded { entries, result })
    }
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Upload a large local folder to this repository as a sequence of resumable,
    /// adaptively-batched commits. Mirrors Python `huggingface_hub.upload_large_folder`
    /// with a byte-compatible on-disk cache (`<folder>/.cache/huggingface/upload/`),
    /// so a partial upload by either tool can be resumed by the other.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn upload_large_folder(
        &self,
        /// Local folder to upload.
        folder_path: PathBuf,
        /// Destination prefix within the repo. Defaults to the repo root.
        #[builder(into)]
        path_in_repo: Option<String>,
        /// Branch to commit to. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message used for every batch. Defaults to
        /// "Add files using upload-large-folder tool".
        #[builder(into)]
        commit_message: Option<String>,
        /// Commit description used for every batch.
        #[builder(into)]
        commit_description: Option<String>,
        /// Commit onto a PR instead of the branch.
        #[builder(default)]
        create_pr: bool,
        /// When the repo is created (it is created if missing), whether it is private.
        private: Option<bool>,
        /// Include-only globs (repo-relative).
        allow_patterns: Option<Vec<String>>,
        /// Exclude globs (repo-relative), appended to the built-in defaults.
        ignore_patterns: Option<Vec<String>>,
        /// Max concurrent producer tasks — `/preupload` classify requests and
        /// in-flight xet `UploadCommit` batches share this single budget. The Hub
        /// committer is always single-flight and is not counted here. Defaults to
        /// `available_parallelism() / 2` (min 1).
        num_workers: Option<usize>,
        /// Progress handler (aggregate + LargeFolderStatus events; no per-file).
        #[builder(into)]
        progress: Option<Progress>,
    ) -> crate::error::HFResult<UploadLargeFolderReport> {
        self.upload_large_folder_impl(UploadLargeFolderArgs {
            folder_path,
            path_in_repo,
            revision,
            commit_message,
            commit_description,
            create_pr,
            private,
            allow_patterns,
            ignore_patterns,
            num_workers,
            progress,
        })
        .await
    }
}

struct UploadLargeFolderArgs {
    folder_path: PathBuf,
    path_in_repo: Option<String>,
    revision: Option<String>,
    commit_message: Option<String>,
    commit_description: Option<String>,
    create_pr: bool,
    private: Option<bool>,
    allow_patterns: Option<Vec<String>>,
    ignore_patterns: Option<Vec<String>>,
    num_workers: Option<usize>,
    progress: Option<Progress>,
}

impl<T: RepoType> HFRepository<T> {
    async fn upload_large_folder_impl(
        &self,
        args: UploadLargeFolderArgs,
    ) -> crate::error::HFResult<UploadLargeFolderReport> {
        let revision = args.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION).to_string();
        let commit_message = args
            .commit_message
            .unwrap_or_else(|| "Add files using upload-large-folder tool".to_string());

        let folder = args.folder_path.canonicalize()?;
        if !folder.is_dir() {
            return Err(crate::error::HFError::InvalidParameter(format!(
                "folder_path {} is not a directory",
                folder.display()
            )));
        }

        // Ensure the repo exists (create-if-missing), matching Python.
        self.hf_client
            .create_repository()
            .repo_id(self.repo_path())
            .repo_type(*self.repo_type())
            .maybe_private(args.private)
            .exist_ok(true)
            .send()
            .await?;

        let discovered =
            discover_files(&folder, args.path_in_repo.as_deref(), &args.allow_patterns, &args.ignore_patterns)?;
        let total_files = discovered.len();

        let mut items: Vec<Item> = Vec::with_capacity(discovered.len());
        for (repo_path, file_path) in discovered {
            let paths = get_local_upload_paths(&folder, &repo_path)?;
            let meta = read_upload_metadata(&paths)?;
            let (size, sample) = crate::repository::upload::read_size_and_sample(&AddSource::File(file_path.clone()))?;
            items.push(Item {
                paths,
                file_path,
                meta,
                sample,
                size,
            });
        }

        let total_bytes: u64 = items.iter().map(|i| i.size).sum();
        args.progress.emit(UploadEvent::Start {
            total_files,
            total_bytes,
        });

        let config = PipelineConfig::from_num_workers(args.num_workers);
        let counters = StatusCounters::default();
        let (tx, rx) = unbounded_channel::<CommitReady>();

        let producer = self.run_producer(&mut items, tx, &revision, &config, &counters, &args.progress);
        let committer = self.run_committer(
            rx,
            &commit_message,
            args.commit_description.as_deref(),
            &revision,
            args.create_pr,
            &config,
            &counters,
            &args.progress,
        );

        let ((), (commits, committed_indices)) = tokio::try_join!(producer, committer)?;

        for idx in committed_indices {
            items[idx].meta.is_committed = true;
        }

        let files_uploaded_lfs = items.iter().filter(|i| i.meta.upload_mode.as_deref() == Some("lfs")).count();
        let files_ignored = items.iter().filter(|i| i.meta.should_ignore == Some(true)).count();
        let files_committed_inline = items
            .iter()
            .filter(|i| i.meta.is_committed && i.meta.upload_mode.as_deref() == Some("regular"))
            .count();

        args.progress.emit(UploadEvent::Complete);

        Ok(UploadLargeFolderReport {
            commits,
            total_files,
            files_uploaded_lfs,
            files_committed_inline,
            files_ignored,
            bytes_uploaded: counters.bytes_uploaded.load(Ordering::Relaxed),
            dedup_bytes_saved: counters.dedup_bytes_saved.load(Ordering::Relaxed),
        })
    }

    /// Producer half of the pipeline. Owns `items`, classifies + uploads with a
    /// shared `num_workers`-bounded concurrency budget, and streams per-file
    /// `CommitReady` messages to the committer. Closes the channel when done.
    async fn run_producer(
        &self,
        items: &mut [Item],
        tx: UnboundedSender<CommitReady>,
        revision: &str,
        config: &PipelineConfig,
        counters: &StatusCounters,
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        counters.files_total.store(items.len(), Ordering::Relaxed);

        let mut to_classify: VecDeque<usize> = VecDeque::new();
        let mut staging: TimedBatchBuffer<StagedLfs> = TimedBatchBuffer::new();

        // Initial routing from persisted metadata (resume + already-classified).
        for (idx, item) in items.iter().enumerate() {
            match seed_stage(&item.meta) {
                WorkStage::Done => {
                    if item.meta.should_ignore == Some(true) {
                        counters.ignored.fetch_add(1, Ordering::Relaxed);
                    }
                    if item.meta.is_committed {
                        counters.committed.fetch_add(1, Ordering::Relaxed);
                        counters.skipped.fetch_add(1, Ordering::Relaxed);
                        counters.skipped_bytes.fetch_add(item.size, Ordering::Relaxed);
                    }
                },
                WorkStage::Classify => to_classify.push_back(idx),
                WorkStage::PreuploadLfs => {
                    counters.upload_mode_known.fetch_add(1, Ordering::Relaxed);
                    counters.lfs_total.fetch_add(1, Ordering::Relaxed);
                    staging.push(StagedLfs {
                        idx,
                        path_in_repo: item.paths.path_in_repo.clone(),
                        file_path: item.file_path.clone(),
                        size: item.size,
                    });
                },
                WorkStage::Commit => {
                    counters.upload_mode_known.fetch_add(1, Ordering::Relaxed);
                    if item.meta.upload_mode.as_deref() == Some("lfs") {
                        counters.lfs_total.fetch_add(1, Ordering::Relaxed);
                        counters.preuploaded.fetch_add(1, Ordering::Relaxed);
                        // Already uploaded to CAS in a previous run — its upload was reused.
                        counters.skipped.fetch_add(1, Ordering::Relaxed);
                        counters.skipped_bytes.fetch_add(item.size, Ordering::Relaxed);
                        let oid = item.meta.sha256.clone().ok_or_else(|| {
                            HFError::Other(format!("missing oid for uploaded lfs file {}", item.paths.path_in_repo))
                        })?;
                        let _ = tx.send(CommitReady {
                            idx,
                            resolved: ResolvedAdd::Lfs {
                                path_in_repo: item.paths.path_in_repo.clone(),
                                oid,
                                size: item.size,
                            },
                            paths: item.paths.clone(),
                            meta: item.meta.clone(),
                        });
                    } else {
                        let _ = tx.send(CommitReady {
                            idx,
                            resolved: ResolvedAdd::Inline {
                                path_in_repo: item.paths.path_in_repo.clone(),
                                source: AddSource::File(item.file_path.clone()),
                            },
                            paths: item.paths.clone(),
                            meta: item.meta.clone(),
                        });
                    }
                },
            }
        }
        pipeline::emit_status(counters, progress);

        // Element type (a boxed Send future) is inferred from the `.boxed()` pushes below.
        let mut inflight = FuturesUnordered::new();
        let mut classify_inflight: usize = 0usize;
        let mut stale_flush_pending = false;

        loop {
            // Dispatch classify work while under the shared concurrency cap.
            while inflight.len() < config.num_workers && !to_classify.is_empty() {
                let mut chunk_input: Vec<(usize, String, u64, Vec<u8>)> = Vec::with_capacity(PREUPLOAD_BATCH_SIZE);
                for _ in 0..PREUPLOAD_BATCH_SIZE {
                    match to_classify.pop_front() {
                        Some(idx) => chunk_input.push((
                            idx,
                            items[idx].paths.path_in_repo.clone(),
                            items[idx].size,
                            items[idx].sample.clone(),
                        )),
                        None => break,
                    }
                }
                classify_inflight += 1;
                inflight.push(self.classify_chunk(chunk_input, revision).boxed());
            }

            let classify_done = to_classify.is_empty() && classify_inflight == 0;

            // Dispatch upload batches: full, drained, or a pending stale flush.
            while inflight.len() < config.num_workers
                && !staging.is_empty()
                && (staging.len() >= config.xet_batch_size || classify_done || stale_flush_pending)
            {
                let batch = staging.take(config.xet_batch_size);
                inflight.push(self.upload_chunk(batch, revision).boxed());
                stale_flush_pending = false;
            }

            if inflight.is_empty() && to_classify.is_empty() && staging.is_empty() {
                break;
            }

            let deadline = staging.deadline(config.max_xet_batch_wait);
            tokio::select! {
                maybe = inflight.next(), if !inflight.is_empty() => {
                    let out = maybe.expect("guarded by !inflight.is_empty()")?;
                    match out {
                        WorkOutput::Classified(results) => {
                            classify_inflight -= 1;
                            for r in results {
                                items[r.idx].meta.upload_mode = Some(r.upload_mode.clone());
                                items[r.idx].meta.should_ignore = Some(r.should_ignore);
                                items[r.idx].meta.save(&items[r.idx].paths)?;
                                counters.upload_mode_known.fetch_add(1, Ordering::Relaxed);
                                if r.should_ignore {
                                    counters.ignored.fetch_add(1, Ordering::Relaxed);
                                } else if r.upload_mode == "lfs" {
                                    counters.lfs_total.fetch_add(1, Ordering::Relaxed);
                                    staging.push(StagedLfs {
                                        idx: r.idx,
                                        path_in_repo: items[r.idx].paths.path_in_repo.clone(),
                                        file_path: items[r.idx].file_path.clone(),
                                        size: items[r.idx].size,
                                    });
                                } else {
                                    let _ = tx.send(CommitReady {
                                        idx: r.idx,
                                        resolved: ResolvedAdd::Inline {
                                            path_in_repo: items[r.idx].paths.path_in_repo.clone(),
                                            source: AddSource::File(items[r.idx].file_path.clone()),
                                        },
                                        paths: items[r.idx].paths.clone(),
                                        meta: items[r.idx].meta.clone(),
                                    });
                                }
                            }
                            pipeline::emit_status(counters, progress);
                        },
                        WorkOutput::Uploaded { entries, result } => {
                            let by_path: HashMap<String, String> =
                                result.files.into_iter().map(|f| (f.path_in_repo, f.oid)).collect();
                            counters.bytes_uploaded.fetch_add(result.transfer_bytes, Ordering::Relaxed);
                            counters.dedup_bytes_saved.fetch_add(result.dedup_bytes_saved, Ordering::Relaxed);
                            for e in entries {
                                let oid = by_path.get(&e.path_in_repo).cloned().ok_or_else(|| {
                                    HFError::Other(format!(
                                        "xet_upload_batch returned no result for {}; aborting to avoid an upload loop",
                                        e.path_in_repo
                                    ))
                                })?;
                                items[e.idx].meta.sha256 = Some(oid.clone());
                                items[e.idx].meta.is_uploaded = true;
                                items[e.idx].meta.save(&items[e.idx].paths)?;
                                counters.hashed.fetch_add(1, Ordering::Relaxed);
                                counters.preuploaded.fetch_add(1, Ordering::Relaxed);
                                let _ = tx.send(CommitReady {
                                    idx: e.idx,
                                    resolved: ResolvedAdd::Lfs { path_in_repo: e.path_in_repo, oid, size: e.size },
                                    paths: items[e.idx].paths.clone(),
                                    meta: items[e.idx].meta.clone(),
                                });
                            }
                            pipeline::emit_status(counters, progress);
                        },
                    }
                }
                _ = sleep_or_pending(deadline), if !staging.is_empty() => {
                    stale_flush_pending = true;
                }
            }
        }

        drop(tx);
        Ok(())
    }

    /// Commit up to `max` buffered files in one Hub commit, persist `is_committed`
    /// for each on success, and record committed indices. Single call site of
    /// `commit_resolved_operations` in the pipeline.
    #[allow(clippy::too_many_arguments)]
    async fn commit_batch(
        &self,
        buffer: &mut TimedBatchBuffer<CommitReady>,
        max: usize,
        commit_message: &str,
        commit_description: Option<&str>,
        revision: &str,
        create_pr: bool,
        counters: &StatusCounters,
        progress: &Option<Progress>,
        commits: &mut Vec<CommitInfo>,
        committed_indices: &mut Vec<usize>,
    ) -> HFResult<()> {
        let batch = buffer.take(max);
        if batch.is_empty() {
            return Ok(());
        }
        let mut adds: Vec<ResolvedAdd> = Vec::with_capacity(batch.len());
        let mut metas: Vec<(usize, LocalUploadFilePaths, LocalUploadFileMetadata)> = Vec::with_capacity(batch.len());
        for cr in batch {
            adds.push(cr.resolved);
            metas.push((cr.idx, cr.paths, cr.meta));
        }
        let info = self
            .commit_resolved_operations(
                &adds,
                &[],
                commit_message,
                commit_description,
                revision,
                create_pr,
                None,
                progress,
            )
            .await?;
        for (idx, paths, mut meta) in metas {
            meta.is_committed = true;
            meta.save(&paths)?;
            committed_indices.push(idx);
            counters.committed.fetch_add(1, Ordering::Relaxed);
        }
        commits.push(info);
        pipeline::emit_status(counters, progress);
        Ok(())
    }

    /// Committer half of the pipeline. Single task, single-flight commits. Drains
    /// `rx` into a `TimedBatchBuffer`; commits on full / stale (oldest-anchored) /
    /// channel-close.
    #[allow(clippy::too_many_arguments)]
    async fn run_committer(
        &self,
        mut rx: UnboundedReceiver<CommitReady>,
        commit_message: &str,
        commit_description: Option<&str>,
        revision: &str,
        create_pr: bool,
        config: &PipelineConfig,
        counters: &StatusCounters,
        progress: &Option<Progress>,
    ) -> HFResult<(Vec<CommitInfo>, Vec<usize>)> {
        let mut buffer: TimedBatchBuffer<CommitReady> = TimedBatchBuffer::new();
        let mut commits: Vec<CommitInfo> = Vec::new();
        let mut committed_indices: Vec<usize> = Vec::new();
        let mut closed = false;

        loop {
            while buffer.len() >= config.max_commit_files {
                self.commit_batch(
                    &mut buffer,
                    config.max_commit_files,
                    commit_message,
                    commit_description,
                    revision,
                    create_pr,
                    counters,
                    progress,
                    &mut commits,
                    &mut committed_indices,
                )
                .await?;
            }

            if closed {
                if buffer.is_empty() {
                    break;
                }
                self.commit_batch(
                    &mut buffer,
                    config.max_commit_files,
                    commit_message,
                    commit_description,
                    revision,
                    create_pr,
                    counters,
                    progress,
                    &mut commits,
                    &mut committed_indices,
                )
                .await?;
                continue;
            }

            let deadline = buffer.deadline(config.max_commit_wait);
            tokio::select! {
                recv = rx.recv() => {
                    match recv {
                        Some(cr) => buffer.push(cr),
                        None => closed = true,
                    }
                }
                _ = sleep_or_pending(deadline), if !buffer.is_empty() => {
                    self.commit_batch(
                        &mut buffer, config.max_commit_files, commit_message, commit_description, revision,
                        create_pr, counters, progress, &mut commits, &mut committed_indices,
                    )
                    .await?;
                }
            }
        }

        Ok((commits, committed_indices))
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::upload_large_folder`]. See the async
    /// method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn upload_large_folder(
        &self,
        folder_path: PathBuf,
        #[builder(into)] path_in_repo: Option<String>,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(into)] commit_description: Option<String>,
        #[builder(default)] create_pr: bool,
        private: Option<bool>,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        /// Max concurrent producer tasks (classify + xet upload batches); the Hub
        /// committer is always single-flight. Defaults to `available_parallelism() / 2`.
        num_workers: Option<usize>,
        #[builder(into)] progress: Option<crate::progress::Progress>,
    ) -> crate::error::HFResult<UploadLargeFolderReport> {
        self.runtime.block_on(
            self.inner
                .upload_large_folder()
                .folder_path(folder_path)
                .maybe_path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .maybe_commit_description(commit_description)
                .create_pr(create_pr)
                .maybe_private(private)
                .maybe_allow_patterns(allow_patterns)
                .maybe_ignore_patterns(ignore_patterns)
                .maybe_num_workers(num_workers)
                .maybe_progress(progress)
                .send(),
        )
    }
}

#[cfg(test)]
mod discovery_tests {
    use super::*;

    fn write(root: &std::path::Path, rel: &str, bytes: &[u8]) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn discovers_files_applies_default_and_user_ignores_and_prefix() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.txt", b"1");
        write(dir.path(), "sub/b.bin", b"22");
        write(dir.path(), ".git/config", b"x");
        write(dir.path(), ".cache/huggingface/upload/a.txt.metadata", b"meta");
        write(dir.path(), "skip.me", b"z");

        let found = discover_files(dir.path(), Some("models"), &None, &Some(vec!["*.me".to_string()])).unwrap();

        let repo_paths: std::collections::BTreeSet<String> =
            found.iter().map(|(repo_path, _)| repo_path.clone()).collect();
        assert_eq!(
            repo_paths,
            ["models/a.txt".to_string(), "models/sub/b.bin".to_string()]
                .into_iter()
                .collect()
        );
    }
}
