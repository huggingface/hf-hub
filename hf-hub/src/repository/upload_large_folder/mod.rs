//! `HFRepository::upload_large_folder`: resumable, xet-optimized upload of a
//! large local folder as a sequence of adaptively-batched commits.

pub mod local_folder;
pub mod pipeline;

use std::path::{Path, PathBuf};

use bon::bon;

use crate::constants;
use crate::error::{HFError, HFResult};
use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::repository::files::{AddSource, matches_any_glob};
use crate::repository::upload::ResolvedAdd;
use crate::repository::upload_large_folder::local_folder::{
    LocalUploadFileMetadata, LocalUploadFilePaths, get_local_upload_paths, read_upload_metadata,
};
use crate::repository::upload_large_folder::pipeline::{CommitChunkSizer, WorkStage, seed_stage};
use crate::repository::{CommitInfo, HFRepository, RepoType};

/// Summary returned by [`HFRepository::upload_large_folder`].
#[derive(Debug, Clone, Default)]
pub struct UploadLargeFolderReport {
    /// The commits created, in the order they were committed (one per batch).
    pub commits: Vec<CommitInfo>,
    /// Total files in the upload set after filtering.
    pub total_files: usize,
    /// Files uploaded via xet/lfs.
    pub files_uploaded_lfs: usize,
    /// Files committed inline (regular).
    pub files_committed_inline: usize,
    /// Files the Hub told us to ignore.
    pub files_ignored: usize,
    /// Bytes actually transferred to CAS (post-dedup).
    pub bytes_uploaded: u64,
    /// Bytes saved by xet deduplication.
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
        /// Bound on concurrent classify/read tasks (reserved in v1; xet manages
        /// upload parallelism internally). Defaults to available_parallelism()/2.
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
        // num_workers is reserved for future concurrency tuning (xet manages its
        // own upload parallelism in v1).
        let _ = args.num_workers;

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

        // Discover and seed.
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

        self.classify_items(&mut items, &revision).await?;

        let mut sizer = CommitChunkSizer::new();
        let mut bytes_uploaded = 0u64;
        let mut dedup_bytes_saved = 0u64;
        self.upload_lfs_stage(
            &mut items,
            &revision,
            &args.progress,
            &mut sizer,
            &mut bytes_uploaded,
            &mut dedup_bytes_saved,
        )
        .await?;

        let commits = self
            .commit_stage(
                &mut items,
                &commit_message,
                args.commit_description.as_deref(),
                &revision,
                args.create_pr,
                &args.progress,
                &mut sizer,
            )
            .await?;

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
            bytes_uploaded,
            dedup_bytes_saved,
        })
    }

    /// Classify every item whose mode is unknown via `/preupload` (batched 256),
    /// persisting `upload_mode` to the cache. Empty files are always regular.
    async fn classify_items(&self, items: &mut [Item], revision: &str) -> crate::error::HFResult<()> {
        let to_classify: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|(_, i)| seed_stage(&i.meta) == WorkStage::Classify)
            .map(|(idx, _)| idx)
            .collect();

        for chunk in to_classify.chunks(256) {
            let payload: Vec<(&str, u64, &[u8])> = chunk
                .iter()
                .map(|&idx| (items[idx].paths.path_in_repo.as_str(), items[idx].size, items[idx].sample.as_slice()))
                .collect();
            let modes = self
                .fetch_upload_modes(&self.repo_path(), self.repo_type.plural(), revision, &payload)
                .await?;
            for &idx in chunk {
                let path = items[idx].paths.path_in_repo.clone();
                let mode = if items[idx].size == 0 {
                    "regular".to_string()
                } else {
                    modes.get(&path).cloned().unwrap_or_else(|| "regular".to_string())
                };
                items[idx].meta.upload_mode = Some(mode);
                items[idx].meta.save(&items[idx].paths)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn upload_lfs_stage(
        &self,
        items: &mut [Item],
        revision: &str,
        progress: &Option<Progress>,
        sizer: &mut CommitChunkSizer,
        bytes_uploaded: &mut u64,
        dedup_bytes_saved: &mut u64,
    ) -> crate::error::HFResult<()> {
        loop {
            let batch: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, i)| seed_stage(&i.meta) == WorkStage::PreuploadLfs)
                .map(|(idx, _)| idx)
                .take(sizer.target().min(256))
                .collect();
            if batch.is_empty() {
                break;
            }

            let files_in: Vec<(String, AddSource)> = batch
                .iter()
                .map(|&idx| (items[idx].paths.path_in_repo.clone(), AddSource::File(items[idx].file_path.clone())))
                .collect();

            let result = self.xet_upload_batch(&files_in, revision, progress).await?;
            *bytes_uploaded += result.transfer_bytes;
            *dedup_bytes_saved += result.dedup_bytes_saved;

            let by_path: std::collections::HashMap<String, (String, u64)> =
                result.files.into_iter().map(|f| (f.path_in_repo, (f.oid, f.size))).collect();

            for &idx in &batch {
                let path = items[idx].paths.path_in_repo.clone();
                match by_path.get(&path) {
                    Some((oid, _size)) => {
                        items[idx].meta.sha256 = Some(oid.clone());
                        items[idx].meta.is_uploaded = true;
                        items[idx].meta.save(&items[idx].paths)?;
                    },
                    None => {
                        return Err(crate::error::HFError::Other(format!(
                            "xet_upload_batch returned no result for {path}; aborting to avoid an upload loop"
                        )));
                    },
                }
            }
            self.emit_status(items, *bytes_uploaded, *dedup_bytes_saved, progress);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn commit_stage(
        &self,
        items: &mut [Item],
        commit_message: &str,
        commit_description: Option<&str>,
        revision: &str,
        create_pr: bool,
        progress: &Option<Progress>,
        sizer: &mut CommitChunkSizer,
    ) -> crate::error::HFResult<Vec<crate::repository::CommitInfo>> {
        let mut commits = Vec::new();
        loop {
            let batch: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, i)| seed_stage(&i.meta) == WorkStage::Commit)
                .map(|(idx, _)| idx)
                .take(sizer.target())
                .collect();
            if batch.is_empty() {
                break;
            }

            let mut adds: Vec<ResolvedAdd> = Vec::with_capacity(batch.len());
            for &idx in &batch {
                let path = items[idx].paths.path_in_repo.clone();
                if items[idx].meta.upload_mode.as_deref() == Some("lfs") {
                    let oid = items[idx].meta.sha256.clone().ok_or_else(|| {
                        crate::error::HFError::Other(format!("missing oid for uploaded lfs file {path}"))
                    })?;
                    adds.push(ResolvedAdd::Lfs {
                        path_in_repo: path,
                        oid,
                        size: items[idx].size,
                    });
                } else {
                    adds.push(ResolvedAdd::Inline {
                        path_in_repo: path,
                        source: AddSource::File(items[idx].file_path.clone()),
                    });
                }
            }

            let start = std::time::Instant::now();
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
                .await;
            let duration = start.elapsed().as_secs_f64();

            match info {
                Ok(info) => {
                    for &idx in &batch {
                        items[idx].meta.is_committed = true;
                        items[idx].meta.save(&items[idx].paths)?;
                    }
                    sizer.update(true, batch.len(), duration);
                    commits.push(info);
                },
                Err(e) => {
                    sizer.update(false, batch.len(), duration);
                    return Err(e);
                },
            }
        }
        Ok(commits)
    }

    fn emit_status(&self, items: &[Item], bytes_uploaded: u64, dedup_bytes_saved: u64, progress: &Option<Progress>) {
        let files_total = items.len();
        let upload_mode_known = items.iter().filter(|i| i.meta.upload_mode.is_some()).count();
        let lfs_total = items.iter().filter(|i| i.meta.upload_mode.as_deref() == Some("lfs")).count();
        let preuploaded = items.iter().filter(|i| i.meta.is_uploaded).count();
        let committed = items.iter().filter(|i| i.meta.is_committed).count();
        let ignored = items.iter().filter(|i| i.meta.should_ignore == Some(true)).count();
        let hashed = items.iter().filter(|i| i.meta.sha256.is_some()).count();
        progress.emit(UploadEvent::LargeFolderStatus {
            files_total,
            hashed,
            upload_mode_known,
            lfs_total,
            preuploaded,
            committed,
            ignored,
            bytes_uploaded,
            dedup_bytes_saved,
        });
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
