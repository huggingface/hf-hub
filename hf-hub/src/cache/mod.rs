//! Inspect the local Hugging Face cache.
//!
//! [`HFClient::scan_cache`] walks the cache directory and produces an
//! [`HFCacheInfo`] tree: cached repositories, their revisions, and the
//! individual files in each revision.
//!
//! On disk, downloads are content-addressed — files in
//! `<cache>/<repo_folder>/snapshots/<commit>/<filename>` are pointers (symlinks
//! on Unix, copies on Windows) to immutable blobs in
//! `<cache>/<repo_folder>/blobs/<etag>`. Multiple revisions of the same repo
//! often share the same blob, which is why repo-level sizes are deduplicated
//! while revision-level sizes are not — see [`CachedRepoInfo::size_on_disk`]
//! and [`CachedRevisionInfo::size_on_disk`].

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bon::bon;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::repository::{HFRepository, RepoType};

pub(crate) mod delete;
pub(crate) mod storage;

/// A single file in a cached revision.
///
/// `file_path` is the pointer in the `snapshots/` tree (a symlink on Unix);
/// `blob_path` is the canonical location of the underlying content under
/// `blobs/`. Both paths refer to the same bytes.
#[derive(Debug, Clone)]
pub struct CachedFileInfo {
    /// Path of the file relative to its revision's snapshot root, including
    /// any subdirectories (e.g., `subdir/model.bin`).
    pub file_name: String,
    /// Absolute path of the pointer file inside the `snapshots/` tree.
    /// Symlink on Unix, regular file (a copy of the blob) on Windows.
    pub file_path: PathBuf,
    /// Absolute path of the actual blob under `blobs/`, after resolving
    /// `file_path`. Multiple revisions can point at the same blob.
    pub blob_path: PathBuf,
    /// Size of the blob in bytes.
    pub size_on_disk: u64,
    /// Last access time of the blob, from the filesystem.
    pub blob_last_accessed: SystemTime,
    /// Last modification time of the blob, from the filesystem.
    pub blob_last_modified: SystemTime,
}

/// A cached revision (commit) of a repository.
#[derive(Debug, Clone)]
pub struct CachedRevisionInfo {
    /// Full 40-character commit SHA.
    pub commit_hash: String,
    /// Directory of pointer files for this revision
    /// (`<cache>/<repo_folder>/snapshots/<commit_hash>/`).
    pub snapshot_path: PathBuf,
    /// Files belonging to this revision.
    pub files: Vec<CachedFileInfo>,
    /// Sum of [`CachedFileInfo::size_on_disk`] for every file in this
    /// revision. Blobs shared with other revisions of the same repo are
    /// counted here but not in [`CachedRepoInfo::size_on_disk`].
    pub size_on_disk: u64,
    /// Refs (branches, tags, `refs/pr/<n>`, …) that point at this commit.
    /// May be empty for revisions only reachable by SHA.
    pub refs: Vec<String>,
    /// Latest blob modification time across the files in this revision.
    pub last_modified: SystemTime,
}

/// A cached repository, with its revisions aggregated.
#[derive(Debug, Clone)]
pub struct CachedRepoInfo {
    /// Hub repo identifier (e.g., `gpt2`, `google/bert-base-uncased`).
    pub repo_id: String,
    /// Lowercase singular name of the repo kind (`"model"`, `"dataset"`, `"space"`, or `"kernel"`)
    /// — see [`crate::repository::RepoType::singular`].
    pub repo_type: &'static str,
    /// Absolute path of the repo's cache subfolder
    /// (`<cache>/<type>s--<owner>--<name>/`).
    pub repo_path: PathBuf,
    /// Cached revisions of this repo.
    pub revisions: Vec<CachedRevisionInfo>,
    /// Number of unique blobs stored for this repo. Two revisions sharing the
    /// same blob count once.
    pub nb_files: usize,
    /// Bytes used on disk for unique blobs, with shared blobs counted once.
    pub size_on_disk: u64,
    /// Latest blob access time across all revisions.
    pub last_accessed: SystemTime,
    /// Latest blob modification time across all revisions.
    pub last_modified: SystemTime,
}

/// Snapshot of the local Hugging Face cache directory.
///
/// Returned by [`HFClient::scan_cache`]; aggregates every cached repository
/// found at `cache_dir` along with total disk usage and any warnings emitted
/// during scanning.
#[derive(Debug, Clone)]
pub struct HFCacheInfo {
    /// Cache directory that was scanned.
    pub cache_dir: PathBuf,
    /// Cached repositories discovered under `cache_dir`.
    pub repos: Vec<CachedRepoInfo>,
    /// Sum of [`CachedRepoInfo::size_on_disk`] across all repos.
    pub size_on_disk: u64,
    /// Human-readable warnings for entries that could not be fully scanned —
    /// for example, snapshot pointers whose blobs are missing or unreadable.
    pub warnings: Vec<String>,
}

/// Outcome of deleting one cached revision, returned by [`HFRepository::delete_cached_revision`].
#[derive(Debug, Clone)]
pub struct DeletedRevision {
    /// Bytes reclaimed on disk: blobs that became unreferenced by removing this revision,
    /// plus the revision's `.no_exist` entries and, if this was the last cached revision, the
    /// repo's cache folder and blob-lock directory.
    pub freed_size: u64,
    /// Whether the repo's cache folder was removed because no revisions remained.
    pub repo_removed: bool,
}

#[bon]
impl HFClient {
    /// Scan the configured cache directory and return a summary of all cached repositories,
    /// revisions, and files.
    ///
    /// If the cache directory does not exist, returns an [`HFCacheInfo`] with no repos and zero
    /// size — not an error. Unreadable blobs and dangling snapshot pointers are reported via
    /// [`HFCacheInfo::warnings`] rather than failing the scan.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn scan_cache(&self) -> HFResult<HFCacheInfo> {
        storage::scan_cache_dir(self.cache_dir()).await
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::scan_cache`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn scan_cache(&self) -> HFResult<HFCacheInfo> {
        self.runtime.block_on(self.inner.scan_cache().send())
    }
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Delete one cached revision of this repository from the local cache.
    ///
    /// Removes the revision's snapshot directory and its `.no_exist` entries, then frees any
    /// blob under `blobs/` that no other cached revision of this repo still references. If this
    /// was the last cached revision, the repo's whole cache folder (including its blob-lock
    /// directory) is removed too, and [`DeletedRevision::repo_removed`] is `true`.
    ///
    /// Returns [`HFError::CacheNotEnabled`] if the client has no local cache, or
    /// [`HFError::LocalEntryNotFound`] if `revision` is not cached.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_cached_revision(&self, #[builder(into)] revision: String) -> HFResult<DeletedRevision> {
        if !self.hf_client.cache_enabled() {
            return Err(HFError::CacheNotEnabled);
        }
        let cache_dir = self.hf_client.cache_dir().to_path_buf();
        let repo_folder = storage::repo_folder_name(&self.repo_path(), self.repo_type().plural());

        let plan = {
            let cache_dir = cache_dir.clone();
            let repo_folder = repo_folder.clone();
            tokio::task::spawn_blocking(move || delete::plan(&cache_dir, &repo_folder, &revision))
                .await
                .map_err(|e| HFError::Other(format!("delete_cached_revision plan task failed: {e}")))??
        };

        // The plan carries its own copies of the repo folder and locks directory, already
        // derived from `cache_dir`/`repo_folder` — reuse them instead of rederiving.
        let repo_folder = plan.folder.clone();
        let locks_dir = plan.locks_dir.clone();

        // Sort so every caller acquires locks in the same global order, regardless of which
        // blobs a given deletion happens to touch — this is what makes two overlapping
        // deletions unable to deadlock against each other.
        let mut candidate_etags = plan.candidate_etags.clone();
        candidate_etags.sort();
        let mut guards = Vec::with_capacity(candidate_etags.len());
        for etag in &candidate_etags {
            guards.push(storage::acquire_lock(&cache_dir, &repo_folder, etag).await?);
        }

        let outcome = tokio::task::spawn_blocking(move || delete::apply(plan))
            .await
            .map_err(|e| HFError::Other(format!("delete_cached_revision apply task failed: {e}")))??;

        // Only drop the locks once `apply` has finished with them. The locks directory is
        // removed after that, never before: each lock is an open file handle, and Windows
        // cannot delete a file that is still open.
        drop(guards);

        let mut freed_size = outcome.freed;
        if outcome.repo_removed {
            let freed_locks = tokio::task::spawn_blocking(move || -> HFResult<u64> {
                match std::fs::metadata(&locks_dir) {
                    Ok(_) => {
                        let size = dir_size(&locks_dir)?;
                        std::fs::remove_dir_all(&locks_dir)?;
                        Ok(size)
                    },
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
                    Err(e) => Err(e.into()),
                }
            })
            .await
            .map_err(|e| HFError::Other(format!("delete_cached_revision locks cleanup task failed: {e}")))??;
            freed_size += freed_locks;
        }

        Ok(DeletedRevision {
            freed_size,
            repo_removed: outcome.repo_removed,
        })
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::delete_cached_revision`]. See the async method
    /// for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_cached_revision(&self, #[builder(into)] revision: String) -> HFResult<DeletedRevision> {
        self.runtime
            .block_on(self.inner.delete_cached_revision().revision(revision).send())
    }
}

/// Total size in bytes of all files under `path`, recursing into subdirectories.
fn dir_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Lays out a single cached revision: one blob under `blobs/`, symlinked from a snapshot
    /// pointer named `file.txt`.
    #[cfg(not(windows))]
    fn write_single_revision(cache_dir: &Path, repo_folder: &str, commit: &str, etag: &str, content: &[u8]) {
        let blob = storage::blob_path(cache_dir, repo_folder, etag);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, content).unwrap();

        let snap = storage::snapshot_path(cache_dir, repo_folder, commit, "file.txt");
        std::fs::create_dir_all(snap.parent().unwrap()).unwrap();
        let relative = pathdiff::diff_paths(&blob, snap.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(relative, &snap).unwrap();
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_delete_cached_revision_frees_expected_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let client = HFClient::builder().cache_dir(dir.path()).build().unwrap();
        let repo_folder = storage::repo_folder_name("test/repo", "models");
        write_single_revision(dir.path(), &repo_folder, "commit1", "etag1", b"hello world");
        write_single_revision(dir.path(), &repo_folder, "commit2", "etag2", b"other revision content");

        let repo = client.model("test", "repo");
        let outcome = repo.delete_cached_revision().revision("commit1").send().await.unwrap();

        assert_eq!(outcome.freed_size, "hello world".len() as u64);
        assert!(!outcome.repo_removed);
        assert!(!storage::blob_path(dir.path(), &repo_folder, "etag1").exists());
        assert!(storage::blob_path(dir.path(), &repo_folder, "etag2").exists());
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_delete_cached_revision_blocks_on_held_lock() {
        let dir = tempfile::tempdir().unwrap();
        let client = HFClient::builder().cache_dir(dir.path()).build().unwrap();
        let repo_folder = storage::repo_folder_name("test/repo", "models");
        write_single_revision(dir.path(), &repo_folder, "commit1", "etag1", b"hello world");

        let held = storage::acquire_lock(dir.path(), &repo_folder, "etag1").await.unwrap();

        let repo = client.model("test", "repo");
        let handle = tokio::spawn(async move { repo.delete_cached_revision().revision("commit1").send().await });

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!handle.is_finished(), "delete should still be blocked on the held lock");

        drop(held);

        let outcome = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("delete did not complete after lock release")
            .expect("delete task panicked")
            .expect("delete_cached_revision failed");

        assert_eq!(outcome.freed_size, "hello world".len() as u64);
        assert!(outcome.repo_removed);
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_delete_last_revision_removes_repo_and_locks_dir() {
        let dir = tempfile::tempdir().unwrap();
        let client = HFClient::builder().cache_dir(dir.path()).build().unwrap();
        let repo_folder = storage::repo_folder_name("test/repo", "models");
        write_single_revision(dir.path(), &repo_folder, "commit1", "etag1", b"hello world");

        let repo = client.model("test", "repo");
        let outcome = repo.delete_cached_revision().revision("commit1").send().await.unwrap();

        assert!(outcome.repo_removed);
        assert!(!dir.path().join(&repo_folder).exists());
        assert!(!dir.path().join(".locks").join(&repo_folder).exists());
    }

    #[tokio::test]
    async fn test_delete_cached_revision_uncached_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let client = HFClient::builder().cache_dir(dir.path()).build().unwrap();
        let repo = client.model("test", "repo");

        let err = repo.delete_cached_revision().revision("doesnotexist").send().await.unwrap_err();
        assert!(matches!(err, HFError::LocalEntryNotFound { .. }), "unexpected error: {err:?}");
    }
}
