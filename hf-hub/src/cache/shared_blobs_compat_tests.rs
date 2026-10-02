//! Compatibility with the shared-blob store written by Python `huggingface_hub`
//! (`utils/_shared_blobs.py`).
//!
//! Xet payloads live once at `<cache>/blobs/<hash[:2]>/<hash>`. Each repo's
//! `blobs/<etag>` is a relative symlink into the store, and `<hash>.refs` lists every
//! `<repo_folder>/blobs/<etag>` that points at the payload, one per line.

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use super::{delete, shared_blobs, storage};

const REPO_A: &str = "models--o--a";
const REPO_B: &str = "models--o--b";
const SHARED_HASH: &str = "ab11111111111111111111111111111111111111111111111111111111111111";
const ONLY_A_HASH: &str = "cd22222222222222222222222222222222222222222222222222222222222222";
const ORPHAN_HASH: &str = "ef33333333333333333333333333333333333333333333333333333333333333";

fn store_entry(cache: &Path, hash: &str) -> PathBuf {
    cache.join("blobs").join(&hash[..2]).join(hash)
}

fn manifest(cache: &Path, hash: &str) -> PathBuf {
    cache.join("blobs").join(&hash[..2]).join(format!("{hash}.refs"))
}

fn write_store_entry(cache: &Path, hash: &str, payload: &[u8], refs: &[&str]) {
    let entry = store_entry(cache, hash);
    std::fs::create_dir_all(entry.parent().unwrap()).unwrap();
    std::fs::write(&entry, payload).unwrap();
    let lines: String = refs.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(manifest(cache, hash), lines).unwrap();
}

fn link_repo_blob(cache: &Path, repo: &str, etag: &str, hash: &str) {
    let blobs = cache.join(repo).join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    symlink(format!("../../blobs/{}/{hash}", &hash[..2]), blobs.join(etag)).unwrap();
}

fn link_snapshot_file(cache: &Path, repo: &str, commit: &str, file: &str, etag: &str) {
    let snap = cache.join(repo).join("snapshots").join(commit);
    std::fs::create_dir_all(&snap).unwrap();
    symlink(format!("../../blobs/{etag}"), snap.join(file)).unwrap();
}

/// Two repos sharing one payload, a payload only `REPO_A` uses, and an unreferenced
/// payload. `REPO_A` has revisions `c0` (x.bin) and `c1` (x.bin, y.bin); `REPO_B` has
/// only `c2` (x.bin).
fn write_python_shared_cache(cache: &Path) {
    std::fs::create_dir_all(cache.join("blobs")).unwrap();
    std::fs::write(cache.join("blobs").join(".huggingface-shared-blobs"), "1\n").unwrap();

    write_store_entry(
        cache,
        SHARED_HASH,
        b"0123456789",
        &["models--o--a/blobs/e_shared", "models--o--b/blobs/e_shared"],
    );
    write_store_entry(cache, ONLY_A_HASH, b"abcdefg", &["models--o--a/blobs/e_only_a"]);
    write_store_entry(cache, ORPHAN_HASH, b"xyz", &[]);

    link_repo_blob(cache, REPO_A, "e_shared", SHARED_HASH);
    link_repo_blob(cache, REPO_A, "e_only_a", ONLY_A_HASH);
    link_repo_blob(cache, REPO_B, "e_shared", SHARED_HASH);

    link_snapshot_file(cache, REPO_A, "c0", "x.bin", "e_shared");
    link_snapshot_file(cache, REPO_A, "c1", "x.bin", "e_shared");
    link_snapshot_file(cache, REPO_A, "c1", "y.bin", "e_only_a");
    link_snapshot_file(cache, REPO_B, "c2", "x.bin", "e_shared");

    for (repo, commit) in [(REPO_A, "c1"), (REPO_B, "c2")] {
        std::fs::create_dir_all(cache.join(repo).join("refs")).unwrap();
        std::fs::write(cache.join(repo).join("refs").join("main"), commit).unwrap();
    }
}

#[test]
fn snapshot_files_read_through_the_store() {
    let dir = tempfile::tempdir().unwrap();
    write_python_shared_cache(dir.path());

    let file = storage::snapshot_path(dir.path(), REPO_B, "c2", "x.bin");
    assert_eq!(std::fs::read(file).unwrap(), b"0123456789");
}

#[tokio::test]
async fn scan_lists_only_repos_and_reports_per_repo_blob_paths() {
    let dir = tempfile::tempdir().unwrap();
    write_python_shared_cache(dir.path());

    let info = storage::scan_cache_dir(dir.path()).await.unwrap();

    let mut repo_ids: Vec<&str> = info.repos.iter().map(|r| r.repo_id.as_str()).collect();
    repo_ids.sort();
    assert_eq!(repo_ids, ["o/a", "o/b"]);

    let repo_b = info.repos.iter().find(|r| r.repo_id == "o/b").unwrap();
    let file = &repo_b.revisions[0].files[0];
    assert_eq!(file.blob_path, dir.path().canonicalize().unwrap().join(REPO_B).join("blobs").join("e_shared"));
    assert_eq!(file.size_on_disk, 10);
}

#[tokio::test]
async fn scan_counts_each_store_payload_once() {
    let dir = tempfile::tempdir().unwrap();
    write_python_shared_cache(dir.path());

    let info = storage::scan_cache_dir(dir.path()).await.unwrap();

    let size_of = |id: &str| info.repos.iter().find(|r| r.repo_id == id).unwrap().size_on_disk;
    assert_eq!(size_of("o/a"), 17);
    assert_eq!(size_of("o/b"), 10);
    assert_eq!(info.size_on_disk, 10 + 7 + 3, "shared payload once, orphan payload included");
}

#[test]
fn delete_sweeps_store_entry_whose_last_reference_is_removed() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);

    let outcome = delete::apply(delete::plan(cache, REPO_A, "c1").unwrap()).unwrap();

    assert_eq!(outcome.freed, 7);
    assert!(!outcome.repo_removed);
    assert!(std::fs::symlink_metadata(cache.join(REPO_A).join("blobs").join("e_only_a")).is_err());
    assert!(!store_entry(cache, ONLY_A_HASH).exists());
    assert!(!manifest(cache, ONLY_A_HASH).exists());
    assert!(cache.join(REPO_A).join("blobs").join("e_shared").exists());
    assert!(store_entry(cache, SHARED_HASH).exists());
}

#[test]
fn delete_keeps_store_entry_still_referenced_by_another_repo() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);

    let outcome = delete::apply(delete::plan(cache, REPO_B, "c2").unwrap()).unwrap();

    assert!(outcome.repo_removed);
    assert_eq!(outcome.freed, 0, "the shared payload is still used by REPO_A");
    assert_eq!(std::fs::read(store_entry(cache, SHARED_HASH)).unwrap(), b"0123456789");
    assert_eq!(std::fs::read_to_string(manifest(cache, SHARED_HASH)).unwrap(), "models--o--a/blobs/e_shared\n");
}

#[test]
fn delete_keeps_payload_when_manifest_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    std::fs::remove_file(manifest(cache, ONLY_A_HASH)).unwrap();

    let outcome = delete::apply(delete::plan(cache, REPO_A, "c1").unwrap()).unwrap();

    assert_eq!(outcome.freed, 0);
    assert!(store_entry(cache, ONLY_A_HASH).exists());
}

#[tokio::test]
async fn scan_ignores_store_without_marker() {
    let dir = tempfile::tempdir().unwrap();
    write_python_shared_cache(dir.path());
    std::fs::remove_file(dir.path().join("blobs").join(".huggingface-shared-blobs")).unwrap();

    let info = storage::scan_cache_dir(dir.path()).await.unwrap();

    assert_eq!(info.size_on_disk, 17 + 10, "per-repo sizes summed, store payloads not added");
}

fn store(cache: &Path) -> PathBuf {
    shared_blobs::store_dir(cache).unwrap()
}

#[test]
fn sweep_rewrites_manifest_keeping_only_valid_references() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    std::fs::write(
        manifest(cache, SHARED_HASH),
        "models--o--a/blobs/e_shared\nmodels--o--gone/blobs/e_shared\n../models--o--b/blobs/e_shared\n/abs/blobs/x\nfoo/blobs/e_shared\nmodels--o--b/blobs/e_shared\n",
    )
    .unwrap();

    let store = store(cache);
    let entry = std::fs::canonicalize(store_entry(cache, SHARED_HASH)).unwrap();
    assert_eq!(shared_blobs::sweep(cache, &store, &entry), 0);

    assert_eq!(
        std::fs::read_to_string(manifest(cache, SHARED_HASH)).unwrap(),
        "models--o--a/blobs/e_shared\nmodels--o--b/blobs/e_shared\n"
    );
}

#[test]
fn sweep_refuses_symlinked_lock() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    let victim = cache.join("victim");
    std::fs::write(&victim, b"precious").unwrap();
    symlink(&victim, store_entry(cache, ORPHAN_HASH).with_extension("lock")).unwrap();

    let store = store(cache);
    let entry = std::fs::canonicalize(store_entry(cache, ORPHAN_HASH)).unwrap();
    assert_eq!(shared_blobs::sweep(cache, &store, &entry), 0);

    assert_eq!(std::fs::read(&victim).unwrap(), b"precious");
    assert!(store_entry(cache, ORPHAN_HASH).exists());
}

#[test]
fn sweep_keeps_payload_with_symlinked_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    let real = cache.join("real.refs");
    std::fs::write(&real, "").unwrap();
    std::fs::remove_file(manifest(cache, ORPHAN_HASH)).unwrap();
    symlink(&real, manifest(cache, ORPHAN_HASH)).unwrap();

    let store = store(cache);
    let entry = std::fs::canonicalize(store_entry(cache, ORPHAN_HASH)).unwrap();
    assert_eq!(shared_blobs::sweep(cache, &store, &entry), 0);
    assert!(store_entry(cache, ORPHAN_HASH).exists());
}

#[test]
fn sweep_keeps_payload_when_a_reference_cannot_be_checked() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    std::fs::write(manifest(cache, SHARED_HASH), "models--o--b/blobs/e_shared\n").unwrap();
    let blobs_b = cache.join(REPO_B).join("blobs");
    std::fs::set_permissions(&blobs_b, std::fs::Permissions::from_mode(0o000)).unwrap();

    let store = store(cache);
    let entry = std::fs::canonicalize(store_entry(cache, SHARED_HASH)).unwrap();
    let freed = shared_blobs::sweep(cache, &store, &entry);
    std::fs::set_permissions(&blobs_b, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert_eq!(freed, 0);
    assert!(store_entry(cache, SHARED_HASH).exists());
}

#[test]
fn sweep_does_not_remove_hash_named_directory() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    let entry = store_entry(cache, ORPHAN_HASH);
    std::fs::remove_file(&entry).unwrap();
    std::fs::create_dir(&entry).unwrap();

    let store = store(cache);
    let entry = std::fs::canonicalize(entry).unwrap();
    assert_eq!(shared_blobs::sweep(cache, &store, &entry), 0);
    assert!(entry.is_dir());
}

#[test]
fn delete_sweeps_leftover_store_links_when_repo_is_removed() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    std::fs::write(manifest(cache, ORPHAN_HASH), "models--o--b/blobs/e_orphan\n").unwrap();
    link_repo_blob(cache, REPO_B, "e_orphan", ORPHAN_HASH);

    let outcome = delete::apply(delete::plan(cache, REPO_B, "c2").unwrap()).unwrap();

    assert!(outcome.repo_removed);
    assert_eq!(outcome.freed, 3);
    assert!(!store_entry(cache, ORPHAN_HASH).exists());
    assert!(store_entry(cache, SHARED_HASH).exists());
}

#[test]
fn delete_frees_payload_once_when_two_etags_link_it() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    std::fs::write(manifest(cache, ONLY_A_HASH), "models--o--a/blobs/e_only_a\nmodels--o--a/blobs/e_only_a2\n")
        .unwrap();
    link_repo_blob(cache, REPO_A, "e_only_a2", ONLY_A_HASH);
    link_snapshot_file(cache, REPO_A, "c1", "z.bin", "e_only_a2");

    let outcome = delete::apply(delete::plan(cache, REPO_A, "c1").unwrap()).unwrap();

    assert_eq!(outcome.freed, 7);
    assert!(!store_entry(cache, ONLY_A_HASH).exists());
    for etag in ["e_only_a", "e_only_a2"] {
        assert!(std::fs::symlink_metadata(cache.join(REPO_A).join("blobs").join(etag)).is_err());
    }
}

#[test]
fn delete_does_not_unlink_another_repos_link() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    let snap = cache.join(REPO_A).join("snapshots").join("c3");
    std::fs::create_dir_all(&snap).unwrap();
    symlink("../../../models--o--b/blobs/e_shared", snap.join("x.bin")).unwrap();

    delete::apply(delete::plan(cache, REPO_A, "c3").unwrap()).unwrap();

    assert!(cache.join(REPO_B).join("blobs").join("e_shared").exists());
}

#[tokio::test]
async fn scan_counts_payload_once_when_pointer_skips_repo_link() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path();
    write_python_shared_cache(cache);
    let snap = cache.join(REPO_B).join("snapshots").join("c4");
    std::fs::create_dir_all(&snap).unwrap();
    symlink(format!("../../../blobs/ab/{SHARED_HASH}"), snap.join("direct.bin")).unwrap();

    let info = storage::scan_cache_dir(cache).await.unwrap();

    assert_eq!(info.size_on_disk, 10 + 7 + 3);
}
