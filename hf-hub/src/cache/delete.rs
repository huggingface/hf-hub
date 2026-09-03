//! Deletion of a single cached revision.
//!
//! Split into a planning pass that touches nothing and an apply pass that runs with every
//! doomed blob's lock held. The caller acquires those locks between the two.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::storage;
use crate::error::{HFError, HFResult};

/// Everything needed to delete one cached revision, computed by [`plan`] before any blob
/// lock is held and consumed by [`apply`] once they are.
#[derive(Debug)]
pub(crate) struct DeletePlan {
    pub(crate) repo: PathBuf,
    pub(crate) snapshots_dir: PathBuf,
    pub(crate) snap: PathBuf,
    pub(crate) commit: String,
    /// Cache folder name for the repo, e.g. `models--o--n`.
    pub(crate) folder: String,
    pub(crate) locks_dir: PathBuf,
    pub(crate) doomed: HashMap<PathBuf, u64>,
    pub(crate) snap_canon: Option<PathBuf>,
    pub(crate) blobs_dir_canon: Option<PathBuf>,
    /// Blob file names (etags) under `blobs/` that looked like candidates for removal at
    /// planning time — i.e. doomed blobs not referenced by any other revision then. Locks
    /// must be acquired for every one of these, sorted, before [`apply`] runs. [`apply`]
    /// rechecks references itself once those locks are held, since this snapshot of `keep`
    /// can go stale between planning and locking.
    pub(crate) candidate_etags: Vec<String>,
}

/// Result of [`apply`]: bytes freed, and whether the repo folder was removed because no
/// revisions were left. When it was, the caller still owes a removal of the plan's
/// `locks_dir`, which is safe only once every lock guard has been dropped.
#[derive(Debug)]
pub(crate) struct ApplyOutcome {
    pub(crate) freed: u64,
    pub(crate) repo_removed: bool,
}

/// Rejects an empty string, anything starting with `.`, and anything outside
/// `[A-Za-z0-9._-]`. A commit of `"."` would otherwise resolve back to the snapshots
/// directory itself and delete every revision in the repo; a leading `.` also rejects
/// `".."`, and the character allow-list rejects any path separator.
fn valid_commit(commit: &str) -> bool {
    !commit.is_empty()
        && !commit.starts_with('.')
        && commit
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn is_dir_no_follow(entry: &std::fs::DirEntry) -> bool {
    entry.file_type().is_ok_and(|ft| ft.is_dir())
}

/// Recursively maps every file under `dir` to the canonicalized path it resolves to and
/// that path's size. On Unix, files under a `snapshots/<commit>` directory are symlinks
/// that resolve to `blobs/<etag>`; on Windows they are plain copies, so canonicalization
/// resolves to the file itself.
///
/// Recursion only follows real directories (`DirEntry::file_type`, which does not follow
/// symlinks) so a directory symlink planted under `dir` can't walk this function outside
/// the cache.
fn collect_targets(dir: &Path) -> HashMap<PathBuf, u64> {
    let mut out = HashMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            if is_dir_no_follow(&entry) {
                stack.push(entry.path());
                continue;
            }
            let path = entry.path();
            if let Ok(canonical) = std::fs::canonicalize(&path) {
                let size = std::fs::metadata(&canonical).map(|m| m.len()).unwrap_or(0);
                out.insert(canonical, size);
            }
        }
    }
    out
}

/// Removes every ref file (recursively, since PR refs nest under `refs/pr/<n>`) whose
/// trimmed content names `commit`. Recursion follows only real directories, and only real
/// files (never symlinks) are read and removed, so a symlink planted under `refs_dir`
/// can't be used to read or delete something outside the cache.
fn remove_matching_refs(refs_dir: &Path, commit: &str) {
    let mut stack = vec![refs_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if is_dir_no_follow(&entry) {
                stack.push(entry.path());
                continue;
            }
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_file() {
                continue;
            }
            let path = entry.path();
            if std::fs::read_to_string(&path).is_ok_and(|c| c.trim() == commit) {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

/// Sum of file sizes under `dir`. Recursion follows only real directories.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            if is_dir_no_follow(&entry) {
                stack.push(entry.path());
                continue;
            }
            if let Ok(meta) = std::fs::metadata(entry.path()) {
                total += meta.len();
            }
        }
    }
    total
}

/// Removes `dir` and returns the number of bytes it held, or `0` if it doesn't exist.
fn remove_dir_and_size(dir: &Path) -> u64 {
    if !dir.is_dir() {
        return 0;
    }
    let size = dir_size(dir);
    let _ = std::fs::remove_dir_all(dir);
    size
}

/// Paths under `snapshots_dir` (other than `snap` itself) that are still referenced,
/// mapped to their size. Used both by [`plan`] (to pick an initial set of candidate blobs
/// to lock) and, recomputed, by [`apply`] (to make the final call on what's actually still
/// referenced once the locks are held).
fn collect_keep(snapshots_dir: &Path, snap: &Path) -> HashMap<PathBuf, u64> {
    let mut keep = HashMap::new();
    if let Ok(entries) = std::fs::read_dir(snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path == snap || !is_dir_no_follow(&entry) {
                continue;
            }
            keep.extend(collect_targets(&path));
        }
    }
    keep
}

/// Plan the deletion of `commit` from the repo cached at `<cache_dir>/<repo_folder>`.
///
/// Touches no state. Returns [`crate::error::HFError::LocalEntryNotFound`] when the revision
/// is not cached, and [`crate::error::HFError::InvalidParameter`] when `commit` is malformed
/// or the snapshot is not a real directory under `snapshots/`.
pub(crate) fn plan(cache_dir: &Path, repo_folder: &str, commit: &str) -> HFResult<DeletePlan> {
    if !valid_commit(commit) {
        return Err(HFError::InvalidParameter(format!("invalid commit: {commit:?}")));
    }

    let repo = cache_dir.join(repo_folder);
    let snapshots_dir = repo.join("snapshots");
    let snap = snapshots_dir.join(commit);

    if !snap.is_dir() {
        return Err(HFError::LocalEntryNotFound {
            path: snap.display().to_string(),
        });
    }
    // Defense in depth beyond `valid_commit`: a commit of "." normalizes `snap` back to
    // `snapshots_dir` itself, and a symlinked "snapshot" could point anywhere: both would
    // otherwise cause `remove_dir_all` in `apply` to destroy the wrong directory.
    if snap.parent() != Some(snapshots_dir.as_path()) {
        return Err(HFError::InvalidParameter(format!("invalid commit: {commit:?}")));
    }
    if !std::fs::symlink_metadata(&snap)?.file_type().is_dir() {
        return Err(HFError::InvalidParameter(format!("invalid commit: {commit:?}")));
    }

    let keep = collect_keep(&snapshots_dir, &snap);
    let doomed = collect_targets(&snap);
    // Captured now, before `remove_dir_all` in `apply` deletes `snap`, so it stays
    // resolvable for the Windows-copy case handled there.
    let snap_canon = std::fs::canonicalize(&snap).ok();
    let blobs_dir_canon = std::fs::canonicalize(repo.join("blobs")).ok();

    let mut candidate_etags: Vec<String> = Vec::new();
    for path in doomed.keys() {
        if keep.contains_key(path) {
            continue;
        }
        if blobs_dir_canon.as_ref().is_some_and(|b| path.starts_with(b))
            && let Some(etag) = path.file_name()
        {
            candidate_etags.push(etag.to_string_lossy().into_owned());
        }
    }

    Ok(DeletePlan {
        repo,
        snapshots_dir,
        snap,
        commit: commit.to_string(),
        folder: repo_folder.to_string(),
        locks_dir: locks_dir(cache_dir, repo_folder),
        doomed,
        snap_canon,
        blobs_dir_canon,
        candidate_etags,
    })
}

/// Execute a plan.
///
/// Every etag in [`DeletePlan::candidate_etags`] must be locked by the caller before this
/// runs, and those locks must stay held until it returns.
pub(crate) fn apply(plan: DeletePlan) -> HFResult<ApplyOutcome> {
    let DeletePlan {
        repo,
        snapshots_dir,
        snap,
        commit,
        folder: _,
        locks_dir: _,
        doomed,
        snap_canon,
        blobs_dir_canon,
        candidate_etags: _,
    } = plan;

    std::fs::remove_dir_all(&snap)?;
    remove_matching_refs(&repo.join("refs"), &commit);

    // Recompute `keep` now, with every doomed blob's lock held, instead of trusting the
    // set `plan` computed earlier. A concurrent download's content-addressed dedup fast
    // path can link a new snapshot pointer onto an existing blob without ever taking that
    // blob's lock (the content is already on disk, so there's nothing for it to write) —
    // so the set computed before locking can go stale by the time we get here. Rechecking
    // this late, under lock, closes nearly all of that race window.
    let keep = collect_keep(&snapshots_dir, &snap);

    let mut freed: u64 = 0;
    for (path, size) in doomed {
        if keep.contains_key(&path) {
            continue;
        }
        if blobs_dir_canon.as_ref().is_some_and(|b| path.starts_with(b)) {
            // Unix: pointer files are symlinks, so `path` is the real blob under
            // `blobs/`. Only count it as freed if we actually removed it. Its hf-hub
            // blob lock is held by the caller for the duration of this call.
            if std::fs::remove_file(&path).is_ok() {
                freed += size;
            }
        } else if snap_canon.as_ref().is_some_and(|s| path.starts_with(s)) {
            // Windows: pointer files are plain copies, not symlinks, so canonicalizing a
            // snapshot file returns the snapshot file itself. It was already removed
            // above by `remove_dir_all`, and — matching the Python reference
            // implementation — the blob under `blobs/` is left alone and only
            // reclaimed once every revision referencing it is gone, since a plain copy
            // carries no on-disk link back to it.
            freed += size;
        }
        // Anything else canonicalized somewhere outside both `blobs/` and this snapshot
        // (e.g. a symlink escaping the cache) is left untouched and not counted as freed.
    }

    freed += remove_dir_and_size(&repo.join(".no_exist").join(&commit));

    let snapshots_empty = std::fs::read_dir(&snapshots_dir)
        .map(|mut it| it.next().is_none())
        .unwrap_or(true);
    let mut repo_removed = false;
    if snapshots_empty {
        freed += remove_dir_and_size(&repo);
        repo_removed = true;
    }

    Ok(ApplyOutcome { freed, repo_removed })
}

/// Directory holding this repo's blob locks, for post-apply cleanup. Derived from
/// [`storage::lock_path`] rather than rebuilt independently, so a future layout change has
/// one home; the etag passed in is irrelevant since only the parent is kept.
pub(crate) fn locks_dir(cache_dir: &Path, repo_folder: &str) -> PathBuf {
    storage::lock_path(cache_dir, repo_folder, "")
        .parent()
        .expect("lock_path always nests the lock file under a directory")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_symlinked_repo(cache_dir: &Path) -> PathBuf {
        let repo = cache_dir.join("models--o--n");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("shared"), b"12345").unwrap();
        std::fs::write(blobs.join("only_a"), b"1234567").unwrap();

        let snap_a = repo.join("snapshots").join("aaa");
        std::fs::create_dir_all(&snap_a).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("../../blobs/shared", snap_a.join("s.txt")).unwrap();
            std::os::unix::fs::symlink("../../blobs/only_a", snap_a.join("a.txt")).unwrap();
        }

        let snap_b = repo.join("snapshots").join("bbb");
        std::fs::create_dir_all(&snap_b).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("../../blobs/shared", snap_b.join("s.txt")).unwrap();
        }

        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "aaa").unwrap();

        repo
    }

    #[test]
    fn plan_rejects_empty_commit() {
        let dir = tempfile::tempdir().unwrap();
        let err = plan(dir.path(), "models--o--n", "").unwrap_err();
        assert!(matches!(err, HFError::InvalidParameter(_)));
    }

    #[test]
    fn plan_rejects_dot_commit() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        std::fs::create_dir_all(repo.join("snapshots").join("aaa")).unwrap();

        let err = plan(dir.path(), "models--o--n", ".").unwrap_err();

        assert!(matches!(err, HFError::InvalidParameter(_)));
        assert!(repo.join("snapshots").join("aaa").is_dir());
    }

    #[test]
    fn plan_rejects_dotdot_commit() {
        let dir = tempfile::tempdir().unwrap();
        let err = plan(dir.path(), "models--o--n", "..").unwrap_err();
        assert!(matches!(err, HFError::InvalidParameter(_)));
    }

    #[test]
    fn plan_rejects_path_separator_commit() {
        let dir = tempfile::tempdir().unwrap();
        let err = plan(dir.path(), "models--o--n", "../x").unwrap_err();
        assert!(matches!(err, HFError::InvalidParameter(_)));
    }

    #[test]
    fn plan_returns_local_entry_not_found_for_missing_revision() {
        let dir = tempfile::tempdir().unwrap();
        let err = plan(dir.path(), "models--o--n", "nope").unwrap_err();
        assert!(matches!(err, HFError::LocalEntryNotFound { .. }));
    }

    #[cfg(not(windows))]
    #[test]
    fn plan_rejects_symlinked_snapshot_directory() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        let snapshots = repo.join("snapshots");
        std::fs::create_dir_all(&snapshots).unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), snapshots.join("aaa")).unwrap();

        let err = plan(dir.path(), "models--o--n", "aaa").unwrap_err();

        assert!(matches!(err, HFError::InvalidParameter(_)));
    }

    #[cfg(not(windows))]
    #[test]
    fn delete_frees_unreferenced_blob_but_keeps_shared_blob() {
        let dir = tempfile::tempdir().unwrap();
        let repo = write_symlinked_repo(dir.path());

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        assert_eq!(plan_result.candidate_etags, vec!["only_a".to_string()]);
        let outcome = apply(plan_result).unwrap();

        assert_eq!(outcome.freed, 7);
        assert!(!outcome.repo_removed);
        assert!(!repo.join("snapshots").join("aaa").exists());
        assert!(repo.join("snapshots").join("bbb").is_dir());
        assert!(repo.join("blobs").join("shared").exists(), "blob referenced by bbb must survive");
        assert!(!repo.join("blobs").join("only_a").exists(), "unreferenced blob must be freed");
    }

    #[test]
    fn delete_removes_matching_refs_including_nested_pr_refs() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        std::fs::create_dir_all(repo.join("snapshots").join("aaa")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join("bbb")).unwrap();
        std::fs::create_dir_all(repo.join("refs").join("pr")).unwrap();
        std::fs::write(repo.join("refs").join("main"), "bbb").unwrap();
        std::fs::write(repo.join("refs").join("pr").join("3"), "aaa").unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        apply(plan_result).unwrap();

        assert!(!repo.join("refs").join("pr").join("3").exists());
        assert!(repo.join("refs").join("main").exists());
        assert!(repo.join("snapshots").join("bbb").is_dir());
    }

    #[test]
    fn delete_removes_no_exist_dir_without_removing_repo() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        std::fs::create_dir_all(repo.join("snapshots").join("aaa")).unwrap();
        std::fs::create_dir_all(repo.join("snapshots").join("bbb")).unwrap();
        let no_exist_dir = repo.join(".no_exist").join("aaa");
        std::fs::create_dir_all(&no_exist_dir).unwrap();
        std::fs::write(no_exist_dir.join("missing.json"), b"").unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        let outcome = apply(plan_result).unwrap();

        assert!(!no_exist_dir.exists());
        assert!(!outcome.repo_removed);
        assert!(repo.join("snapshots").join("bbb").is_dir());
    }

    #[test]
    fn deleting_last_revision_removes_repo_folder_and_reports_repo_removed() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("only"), b"1234567").unwrap();
        let snap = repo.join("snapshots").join("aaa");
        std::fs::create_dir_all(&snap).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("../../blobs/only", snap.join("a.txt")).unwrap();
        #[cfg(not(unix))]
        std::fs::write(snap.join("a.txt"), b"1234567").unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        let outcome = apply(plan_result).unwrap();

        // On Unix the pointer is a symlink with no bytes of its own, so only the blob is
        // counted. On Windows it is a plain copy, so its bytes are counted when the snapshot
        // goes and the blob's bytes are counted again when the emptied repo folder goes —
        // both really are on disk, so both really are freed.
        #[cfg(unix)]
        assert_eq!(outcome.freed, 7);
        #[cfg(not(unix))]
        assert_eq!(outcome.freed, 14);
        assert!(outcome.repo_removed);
        assert!(!repo.exists());
    }

    #[test]
    fn windows_shaped_copies_free_snapshot_bytes_and_spare_shared_blob() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--o--n");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("etag1"), b"123456789").unwrap();

        let snap_a = repo.join("snapshots").join("aaa");
        std::fs::create_dir_all(&snap_a).unwrap();
        std::fs::write(snap_a.join("cfg.txt"), b"123456789").unwrap();

        let snap_b = repo.join("snapshots").join("bbb");
        std::fs::create_dir_all(&snap_b).unwrap();
        std::fs::write(snap_b.join("cfg.txt"), b"123456789").unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        let outcome = apply(plan_result).unwrap();

        assert_eq!(outcome.freed, 9);
        assert!(!repo.join("snapshots").join("aaa").exists());
        assert!(repo.join("snapshots").join("bbb").is_dir());
        assert!(
            blobs.join("etag1").exists(),
            "blob should survive until the last revision referencing it is removed"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn apply_rechecks_keep_set_and_spares_blob_that_gained_a_new_reference() {
        let dir = tempfile::tempdir().unwrap();
        let repo = write_symlinked_repo(dir.path());

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        assert_eq!(plan_result.candidate_etags, vec!["only_a".to_string()]);

        // Simulate a concurrent download's dedup fast path: it links a brand new
        // snapshot onto the blob we're about to delete without ever taking that blob's
        // lock (there's nothing for it to write, the content already exists), between
        // planning and locking.
        let snap_c = repo.join("snapshots").join("ccc");
        std::fs::create_dir_all(&snap_c).unwrap();
        std::os::unix::fs::symlink("../../blobs/only_a", snap_c.join("c.txt")).unwrap();

        let outcome = apply(plan_result).unwrap();

        assert_eq!(outcome.freed, 0);
        assert!(!outcome.repo_removed);
        assert!(
            repo.join("blobs").join("only_a").exists(),
            "blob gained a new reference after planning and must survive"
        );
        assert!(!repo.join("snapshots").join("aaa").exists());
        assert!(repo.join("snapshots").join("bbb").is_dir());
        assert!(repo.join("snapshots").join("ccc").is_dir());
    }

    #[cfg(not(windows))]
    #[test]
    fn refs_directory_symlink_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let repo = write_symlinked_repo(dir.path());

        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"do not touch").unwrap();
        std::os::unix::fs::symlink(outside.path(), repo.join("refs").join("evil")).unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        apply(plan_result).unwrap();

        assert!(outside.path().join("secret.txt").exists());
    }

    #[cfg(not(windows))]
    #[test]
    fn escaping_symlink_target_survives_and_is_not_counted() {
        let dir = tempfile::tempdir().unwrap();
        let repo = write_symlinked_repo(dir.path());

        let outside = tempfile::tempdir().unwrap();
        let outside_file = outside.path().join("escaped.bin");
        std::fs::write(&outside_file, b"0123456789").unwrap();
        std::os::unix::fs::symlink(&outside_file, repo.join("snapshots").join("aaa").join("e.bin")).unwrap();

        let plan_result = plan(dir.path(), "models--o--n", "aaa").unwrap();
        let outcome = apply(plan_result).unwrap();

        assert_eq!(outcome.freed, 7);
        assert!(outside_file.exists());
    }

    #[test]
    fn locks_dir_reuses_storage_lock_path_parent() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            locks_dir(dir.path(), "models--o--n"),
            storage::lock_path(dir.path(), "models--o--n", "etag").parent().unwrap()
        );
    }
}
