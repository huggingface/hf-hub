//! Integration tests for the file system cache.
//!
//! Tests require HF_TOKEN and network access, skip if not set.
//! Interop tests additionally require python3, skip if not found.
//!
//! Run: HF_TOKEN=hf_xxx cargo test -p hf-hub --test cache_test

use std::path::Path;

use hf_hub::test_utils::*;
use hf_hub::{HFClient, HFClientBuilder, HFError, RepoDownloadFileParams, RepoSnapshotDownloadParams};
use serial_test::serial;

fn api() -> Option<HFClient> {
    if is_ci() {
        let token = resolve_prod_token()?;
        Some(
            HFClientBuilder::new()
                .token(token)
                .endpoint(PROD_ENDPOINT)
                .build()
                .expect("Failed to create HFClient"),
        )
    } else {
        if std::env::var(HF_TOKEN).is_err() {
            return None;
        }
        Some(HFClientBuilder::new().build().expect("Failed to create HFClient"))
    }
}

fn api_with_cache(cache_dir: &std::path::Path) -> HFClient {
    if is_ci() {
        let token = std::env::var(HF_PROD_TOKEN).expect("HF_PROD_TOKEN required in CI for prod repo tests");
        HFClientBuilder::new()
            .token(token)
            .endpoint(PROD_ENDPOINT)
            .cache_dir(cache_dir)
            .build()
            .expect("Failed to create HFClient")
    } else {
        HFClientBuilder::new()
            .cache_dir(cache_dir)
            .build()
            .expect("Failed to create HFClient")
    }
}

const TEST_MODEL_PARTS: (&str, &str) = ("hf-internal-testing", "tiny-gemma3");
const TEST_MODEL_REPO_ID: &str = "hf-internal-testing/tiny-gemma3";
const TEST_DATASET_PARTS: (&str, &str) = ("hf-internal-testing", "cats_vs_dogs_sample");
const TEST_DATASET_REPO_ID: &str = "hf-internal-testing/cats_vs_dogs_sample";
const TEST_MODEL_CACHE_FRAGMENT: &str = "hf-internal-testing--tiny-gemma3";
const TEST_DATASET_CACHE_FRAGMENT: &str = "datasets--hf-internal-testing--cats_vs_dogs_sample";

fn find_repo_folder(cache_dir: &Path, name_fragment: &str) -> std::path::PathBuf {
    std::fs::read_dir(cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().contains(name_fragment))
        .unwrap_or_else(|| panic!("repo folder containing '{name_fragment}' not found in {}", cache_dir.display()))
        .path()
}

fn find_single_blob(cache_dir: &Path, repo_name_fragment: &str) -> std::path::PathBuf {
    let repo_folder = find_repo_folder(cache_dir, repo_name_fragment);
    let blobs_dir = repo_folder.join("blobs");
    let blob = std::fs::read_dir(&blobs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .next()
        .unwrap_or_else(|| panic!("no blobs found in {}", blobs_dir.display()));
    blob.path()
}

fn walk_find(dir: &Path, filename: &str) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().map(|n| n == filename).unwrap_or(false) {
                    return true;
                }
            }
        }
    }
    false
}

fn list_files_recursive(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p.strip_prefix(dir).unwrap_or(&p);
                    files.push(rel.to_string_lossy().to_string());
                }
            }
        }
    }
    files
}

// =============================================================================
// Cache-mode download tests
// =============================================================================

#[tokio::test]
async fn test_download_file_to_cache() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
    assert!(path.to_string_lossy().contains("snapshots"));

    let repo_folder = std::fs::read_dir(cache_dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().contains(TEST_MODEL_CACHE_FRAGMENT))
        .expect("repo folder not found");
    let blobs_dir = repo_folder.path().join("blobs");
    assert!(blobs_dir.exists());
    let blob_count = std::fs::read_dir(&blobs_dir).unwrap().count();
    assert_eq!(blob_count, 1);
}

#[tokio::test]
async fn test_download_file_cache_hit() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    let path1 = repo
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    let path2 = repo
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    assert_eq!(path1, path2);
}

#[tokio::test]
async fn test_download_file_local_files_only_miss() {
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let result = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_files_only(true)
                .build(),
        )
        .await;
    assert!(matches!(result, Err(hf_hub::HFError::LocalEntryNotFound { .. })));
}

#[tokio::test]
async fn test_download_file_local_files_only_hit() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    let path1 = repo
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let path2 = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_files_only(true)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(path1, path2);
}

#[cfg(not(windows))]
#[tokio::test]
async fn test_download_file_cache_symlink_structure() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let meta = std::fs::symlink_metadata(&path).unwrap();
    assert!(meta.file_type().is_symlink());
    let target = std::fs::read_link(&path).unwrap();
    assert!(target.to_string_lossy().contains("blobs"));
}

#[tokio::test]
async fn test_snapshot_download() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let snapshot_dir = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["*.json".to_string()])
                .build(),
        )
        .await
        .unwrap();

    assert!(snapshot_dir.exists());
    assert!(snapshot_dir.to_string_lossy().contains("snapshots"));
    let config = snapshot_dir.join("config.json");
    assert!(config.exists());
}

// =============================================================================
// Conditional requests & ETag
// =============================================================================

#[tokio::test]
async fn test_cache_hit_no_redownload() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    repo.download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let blob = find_single_blob(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let mtime_before = std::fs::metadata(&blob).unwrap().modified().unwrap();

    // Second download should use 304 and not touch the blob
    repo.download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    let mtime_after = std::fs::metadata(&blob).unwrap().modified().unwrap();
    assert_eq!(mtime_before, mtime_after);
}

#[tokio::test]
async fn test_force_download_bypasses_cache() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    repo.download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let blob = find_single_blob(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let mtime_before = std::fs::metadata(&blob).unwrap().modified().unwrap();

    // Small delay so mtime can differ
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    repo.download_file(
        &RepoDownloadFileParams::builder()
            .filename("config.json")
            .force_download(true)
            .build(),
    )
    .await
    .unwrap();

    let mtime_after = std::fs::metadata(&blob).unwrap().modified().unwrap();
    assert!(mtime_after > mtime_before, "force_download should rewrite the blob");
}

#[tokio::test]
async fn test_force_download_ignores_no_exist() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    // Create a stale .no_exist marker — network download should succeed
    // regardless since .no_exist is only consulted via resolve_from_cache_only
    let repo_folder = format!("models--{}", TEST_MODEL_CACHE_FRAGMENT);
    let fake_commit = "0000000000000000000000000000000000000000";
    let no_exist_dir = cache_dir.path().join(&repo_folder).join(".no_exist").join(fake_commit);
    std::fs::create_dir_all(&no_exist_dir).unwrap();
    std::fs::write(no_exist_dir.join("config.json"), b"").unwrap();
    let refs_dir = cache_dir.path().join(&repo_folder).join("refs");
    std::fs::create_dir_all(&refs_dir).unwrap();
    std::fs::write(refs_dir.join("main"), fake_commit).unwrap();

    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .force_download(true)
                .build(),
        )
        .await
        .unwrap();
    assert!(path.exists());
}

// =============================================================================
// .no_exist markers
// =============================================================================

#[tokio::test]
async fn test_no_exist_marker_on_404() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let result = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("this_file_does_not_exist_abc123.txt")
                .build(),
        )
        .await;
    assert!(matches!(result, Err(HFError::EntryNotFound { .. })));

    // .no_exist marker should have been written
    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let no_exist_dir = repo_folder.join(".no_exist");
    assert!(no_exist_dir.exists(), ".no_exist directory should exist");

    // Should have exactly one commit dir with the marker file
    let marker_found = walk_find(&no_exist_dir, "this_file_does_not_exist_abc123.txt");
    assert!(marker_found, ".no_exist marker file should exist");
}

#[tokio::test]
async fn test_no_exist_marker_prevents_request() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);

    // First download: 404 creates the .no_exist marker
    let _ = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("nonexistent_file_xyz789.txt")
                .build(),
        )
        .await;

    // .no_exist is checked via resolve_from_cache_only (local_files_only or
    // offline fallback), matching Python's try_to_load_from_cache behavior.
    let result = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("nonexistent_file_xyz789.txt")
                .local_files_only(true)
                .build(),
        )
        .await;
    assert!(
        matches!(result, Err(HFError::EntryNotFound { .. })),
        "Should return EntryNotFound from .no_exist marker via local_files_only: {result:?}"
    );
}

#[tokio::test]
async fn test_no_exist_writes_ref_on_404() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let _ = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("no_such_file_ref_test.txt").build())
        .await;

    // The refs/main file should have been written even though the file 404'd
    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let main_ref = repo_folder.join("refs").join("main");
    assert!(main_ref.exists(), "refs/main should be written on 404 with commit hash header");
    let commit = std::fs::read_to_string(&main_ref).unwrap();
    let commit = commit.trim();
    assert_eq!(commit.len(), 40, "ref should contain a 40-char commit hash, got: {commit}");
    assert!(commit.chars().all(|c| c.is_ascii_hexdigit()), "ref should be hex, got: {commit}");
}

// =============================================================================
// Ref file handling
// =============================================================================

#[tokio::test]
async fn test_ref_written_for_branch_download() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let main_ref = repo_folder.join("refs").join("main");
    assert!(main_ref.exists());
    let commit = std::fs::read_to_string(&main_ref).unwrap();
    let commit = commit.trim();
    assert_eq!(commit.len(), 40);
    assert!(commit.chars().all(|c| c.is_ascii_hexdigit()));
}

#[tokio::test]
async fn test_no_ref_for_commit_hash_download() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    // First get the commit hash via a normal download
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let commit_hash = std::fs::read_to_string(repo_folder.join("refs").join("main"))
        .unwrap()
        .trim()
        .to_string();

    // Now download in a fresh cache using the commit hash directly
    let cache_dir2 = tempfile::tempdir().unwrap();
    let api2 = api_with_cache(cache_dir2.path());

    api2.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .revision(&commit_hash)
                .build(),
        )
        .await
        .unwrap();

    let repo_folder2 = find_repo_folder(cache_dir2.path(), TEST_MODEL_CACHE_FRAGMENT);
    let refs_dir = repo_folder2.join("refs");
    // refs dir should not exist (or be empty) when downloading by commit hash
    if refs_dir.exists() {
        let count = std::fs::read_dir(&refs_dir).unwrap().count();
        assert_eq!(count, 0, "No refs should be written for commit hash downloads");
    }
}

#[tokio::test]
async fn test_download_by_commit_hash() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    // Get commit hash from a normal download
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let commit_hash = std::fs::read_to_string(repo_folder.join("refs").join("main"))
        .unwrap()
        .trim()
        .to_string();

    // Download by commit hash in fresh cache
    let cache_dir2 = tempfile::tempdir().unwrap();
    let api2 = api_with_cache(cache_dir2.path());
    let path = api2
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .revision(&commit_hash)
                .build(),
        )
        .await
        .unwrap();

    assert!(path.exists());
    assert!(
        path.to_string_lossy().contains(&format!("snapshots/{commit_hash}")),
        "Path should contain snapshots/<commit_hash>: {}",
        path.display()
    );
}

// =============================================================================
// Transient error fallback
// =============================================================================

#[tokio::test]
async fn test_offline_fallback_with_cached_file() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    // Populate cache
    let original_path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    // Create API with bogus endpoint (will fail to connect)
    let api_broken = HFClientBuilder::new()
        .cache_dir(cache_dir.path())
        .endpoint("http://localhost:1")
        .build()
        .unwrap();
    let result = api_broken
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await;
    assert!(result.is_ok(), "Should fall back to cached file, got: {result:?}");
    assert_eq!(result.unwrap(), original_path);
}

#[tokio::test]
async fn test_offline_fallback_without_cache_propagates_error() {
    let cache_dir = tempfile::tempdir().unwrap();
    let api_broken = HFClientBuilder::new()
        .cache_dir(cache_dir.path())
        .endpoint("http://localhost:1")
        .build()
        .unwrap();

    let result = api_broken
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await;
    assert!(result.is_err(), "Should propagate error when no cache available");
    // Should NOT be LocalEntryNotFound — should be the original connection error
    assert!(
        !matches!(result, Err(HFError::LocalEntryNotFound { .. })),
        "Error should be the original network error, not LocalEntryNotFound"
    );
}

// =============================================================================
// Snapshot download
// =============================================================================

#[tokio::test]
async fn test_snapshot_download_ignore_patterns() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let snapshot_dir = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .ignore_patterns(vec!["*.md".to_string()])
                .allow_patterns(vec!["*.json".to_string(), "*.md".to_string()])
                .build(),
        )
        .await
        .unwrap();

    // No .md files should be present
    let files = list_files_recursive(&snapshot_dir);
    assert!(files.iter().all(|f| !f.ends_with(".md")), "No .md files should exist: {files:?}");
    assert!(files.iter().any(|f| f.ends_with(".json")), "Should have .json files: {files:?}");
}

#[tokio::test]
async fn test_snapshot_download_local_files_only_miss() {
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let result = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(&RepoSnapshotDownloadParams::builder().local_files_only(true).build())
        .await;
    assert!(matches!(result, Err(HFError::LocalEntryNotFound { .. })));
}

#[tokio::test]
async fn test_snapshot_download_local_files_only_hit() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    let dir1 = repo
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["config.json".to_string()])
                .build(),
        )
        .await
        .unwrap();

    let dir2 = repo
        .snapshot_download(&RepoSnapshotDownloadParams::builder().local_files_only(true).build())
        .await
        .unwrap();
    assert_eq!(dir1, dir2);
}

#[tokio::test]
async fn test_snapshot_download_by_commit_hash() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    // First get the commit hash
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let commit_hash = std::fs::read_to_string(repo_folder.join("refs").join("main"))
        .unwrap()
        .trim()
        .to_string();

    // Snapshot download in fresh cache by commit hash
    let cache_dir2 = tempfile::tempdir().unwrap();
    let api2 = api_with_cache(cache_dir2.path());
    let snapshot_dir = api2
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .revision(&commit_hash)
                .allow_patterns(vec!["config.json".to_string()])
                .build(),
        )
        .await
        .unwrap();

    assert!(snapshot_dir.join("config.json").exists());
    assert!(snapshot_dir.to_string_lossy().contains(&commit_hash), "Snapshot dir should contain commit hash");

    // No ref should be written for commit hash revision
    let repo_folder2 = find_repo_folder(cache_dir2.path(), TEST_MODEL_CACHE_FRAGMENT);
    let refs_dir = repo_folder2.join("refs");
    if refs_dir.exists() {
        let count = std::fs::read_dir(&refs_dir).unwrap().count();
        assert_eq!(count, 0, "No refs should be written for commit hash snapshot");
    }
}

#[tokio::test]
async fn test_snapshot_download_force_download() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    repo.snapshot_download(
        &RepoSnapshotDownloadParams::builder()
            .allow_patterns(vec!["config.json".to_string()])
            .build(),
    )
    .await
    .unwrap();

    let blob = find_single_blob(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let mtime_before = std::fs::metadata(&blob).unwrap().modified().unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    repo.snapshot_download(
        &RepoSnapshotDownloadParams::builder()
            .allow_patterns(vec!["config.json".to_string()])
            .force_download(true)
            .build(),
    )
    .await
    .unwrap();

    let mtime_after = std::fs::metadata(&blob).unwrap().modified().unwrap();
    assert!(mtime_after > mtime_before, "force_download should rewrite the blob");
}

#[tokio::test]
async fn test_snapshot_download_returns_correct_path() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let snapshot_dir = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["config.json".to_string()])
                .build(),
        )
        .await
        .unwrap();

    let path_str = snapshot_dir.to_string_lossy();
    let expected_fragment = format!("models--{}", TEST_MODEL_CACHE_FRAGMENT);
    assert!(path_str.contains(&expected_fragment), "Should contain {expected_fragment}: {path_str}");
    assert!(path_str.contains("snapshots"), "Should contain snapshots: {path_str}");
}

// =============================================================================
// Cache structure verification
// =============================================================================

#[tokio::test]
async fn test_cache_directory_layout() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    assert!(repo_folder.join("blobs").exists(), "blobs/ should exist");
    assert!(repo_folder.join("snapshots").exists(), "snapshots/ should exist");
    assert!(repo_folder.join("refs").exists(), "refs/ should exist");

    // Exactly one snapshot dir (a commit hash dir)
    let snap_entries: Vec<_> = std::fs::read_dir(repo_folder.join("snapshots"))
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(snap_entries.len(), 1);
    let commit_dir_name = snap_entries[0].file_name().to_string_lossy().to_string();
    assert_eq!(commit_dir_name.len(), 40);
}

#[cfg(not(windows))]
#[tokio::test]
async fn test_blob_deduplication_across_downloads() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);

    // Download same file via single file download
    repo.download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let blob_count_before = std::fs::read_dir(repo_folder.join("blobs")).unwrap().count();

    // Download again via snapshot (should reuse the same blob)
    repo.snapshot_download(
        &RepoSnapshotDownloadParams::builder()
            .allow_patterns(vec!["config.json".to_string()])
            .build(),
    )
    .await
    .unwrap();

    let blob_count_after = std::fs::read_dir(repo_folder.join("blobs")).unwrap().count();
    assert_eq!(blob_count_before, blob_count_after, "Blob should be reused, not duplicated");
}

#[tokio::test]
async fn test_dataset_repo_type_cache_folder() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let path = api
        .dataset(TEST_DATASET_PARTS.0, TEST_DATASET_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("README.md").build())
        .await
        .unwrap();
    assert!(path.exists());

    let repo_folder = find_repo_folder(cache_dir.path(), TEST_DATASET_CACHE_FRAGMENT);
    assert!(repo_folder.exists(), "Dataset cache folder not found");
}

#[tokio::test]
async fn test_download_to_local_dir_no_cache() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let local_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(local_dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();

    assert!(path.exists());
    assert!(path.starts_with(local_dir.path()), "File should be in local_dir");

    // Cache should be empty (no repo folders)
    let cache_entries: Vec<_> = std::fs::read_dir(cache_dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains("--"))
        .collect();
    assert!(cache_entries.is_empty(), "Cache should have no repo folders when using local_dir");
}

// =============================================================================
// Concurrent downloads
// =============================================================================

#[tokio::test]
async fn test_concurrent_downloads_same_file() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let mut handles = Vec::new();
    for _ in 0..4 {
        let api_clone = api.clone();
        let handle = tokio::spawn(async move {
            api_clone
                .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
                .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
                .await
        });
        handles.push(handle);
    }

    let mut paths = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        let path = result.unwrap();
        assert!(path.exists());
        paths.push(path);
    }

    // All should return the same path
    for p in &paths {
        assert_eq!(p, &paths[0]);
    }

    // Only one blob should exist
    let repo_folder = find_repo_folder(cache_dir.path(), TEST_MODEL_CACHE_FRAGMENT);
    let blob_count = std::fs::read_dir(repo_folder.join("blobs")).unwrap().count();
    assert_eq!(blob_count, 1, "Concurrent downloads should produce exactly 1 blob");
}

// =============================================================================
// Environment variable handling
// =============================================================================

#[test]
#[serial]
fn test_hf_hub_cache_env_var() {
    let dir = tempfile::tempdir().unwrap();
    // Save and set env
    let old_val = std::env::var(HF_HUB_CACHE).ok();
    // SAFETY: test runs serially (#[serial]) so no concurrent env access
    unsafe { std::env::set_var(HF_HUB_CACHE, dir.path()) };

    let api = HFClientBuilder::new().build().unwrap();
    // Verify through a download attempt that would use the cache dir
    // We can't easily inspect the private field, but we can check the
    // builder override works by using an explicit cache_dir
    let api2 = api_with_cache(dir.path());
    // Both should work without error
    drop(api);
    drop(api2);

    // Restore env
    match old_val {
        Some(v) => unsafe { std::env::set_var(HF_HUB_CACHE, v) },
        None => unsafe { std::env::remove_var(HF_HUB_CACHE) },
    }
}

#[test]
#[serial]
fn test_xdg_cache_home_env_var() {
    let dir = tempfile::tempdir().unwrap();
    // Save existing env vars
    let old_hub_cache = std::env::var(HF_HUB_CACHE).ok();
    let old_hf_home = std::env::var(HF_HOME).ok();
    let old_xdg = std::env::var(XDG_CACHE_HOME).ok();

    // SAFETY: test runs serially (#[serial]) so no concurrent env access
    unsafe {
        std::env::remove_var(HF_HUB_CACHE);
        std::env::remove_var(HF_HOME);
        std::env::set_var(XDG_CACHE_HOME, dir.path());
    }

    let api = HFClientBuilder::new().build().unwrap();
    drop(api);

    // Restore env
    // SAFETY: test runs serially (#[serial]) so no concurrent env access
    unsafe {
        match old_hub_cache {
            Some(v) => std::env::set_var(HF_HUB_CACHE, v),
            None => std::env::remove_var(HF_HUB_CACHE),
        }
        match old_hf_home {
            Some(v) => std::env::set_var(HF_HOME, v),
            None => std::env::remove_var(HF_HOME),
        }
        match old_xdg {
            Some(v) => std::env::set_var(XDG_CACHE_HOME, v),
            None => std::env::remove_var(XDG_CACHE_HOME),
        }
    }
}

// =============================================================================
// Cross-library interoperability tests (Python huggingface_hub)
// =============================================================================

fn python_available() -> bool {
    std::process::Command::new("python3").arg("--version").output().is_ok()
}

fn setup_python_venv(base_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    if !python_available() {
        return None;
    }
    let venv_dir = base_dir.join("venv");
    let status = std::process::Command::new("python3")
        .args(["-m", "venv", &venv_dir.to_string_lossy()])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }

    let pip = venv_dir.join("bin").join("pip");
    let status = std::process::Command::new(&pip)
        .args(["install", "--upgrade", "pip"])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }

    let status = std::process::Command::new(&pip)
        .args(["install", "-q", "huggingface_hub"])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }

    Some(venv_dir)
}

fn python_bin(venv_dir: &std::path::Path) -> std::path::PathBuf {
    venv_dir.join("bin").join("python")
}

#[tokio::test]
async fn test_interop_python_downloads_first() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "config.json")
print(path)
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python failed: {}", String::from_utf8_lossy(&output.stderr));

    let repo_folder = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().contains(TEST_MODEL_CACHE_FRAGMENT))
        .unwrap();
    let blob_count_before = std::fs::read_dir(repo_folder.path().join("blobs")).unwrap().count();

    let api = api_with_cache(&cache_dir);
    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    assert!(path.exists());

    let blob_count_after = std::fs::read_dir(repo_folder.path().join("blobs")).unwrap().count();
    assert_eq!(blob_count_before, blob_count_after);
}

#[tokio::test]
async fn test_interop_rust_downloads_first() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    let api = api_with_cache(&cache_dir);
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "config.json", local_files_only=True)
print(path)
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(
        output.status.success(),
        "Python local_files_only failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn test_interop_mixed_partial_downloads() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
hf_hub_download("{repo_id}", "README.md")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success());

    let api = api_with_cache(&cache_dir);
    let repo = api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1);
    repo.download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let readme_path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("README.md")
                .local_files_only(true)
                .build(),
        )
        .await
        .unwrap();
    assert!(readme_path.exists());

    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "config.json", local_files_only=True)
print(path)
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(
        output.status.success(),
        "Python can't find Rust's file: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn test_interop_python_snapshot_rust_snapshot() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import snapshot_download
path = snapshot_download("{repo_id}", allow_patterns=["*.json"])
print(path)
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(
        output.status.success(),
        "Python snapshot_download failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let repo_folder = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().contains(TEST_MODEL_CACHE_FRAGMENT))
        .unwrap();
    let blob_count_before = std::fs::read_dir(repo_folder.path().join("blobs")).unwrap().count();

    let api = api_with_cache(&cache_dir);
    let snapshot_dir = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["*.json".to_string()])
                .build(),
        )
        .await
        .unwrap();
    assert!(snapshot_dir.exists());

    let blob_count_after = std::fs::read_dir(repo_folder.path().join("blobs")).unwrap().count();
    assert_eq!(blob_count_before, blob_count_after);
}

#[tokio::test]
async fn test_interop_rust_writes_python_validates_cache() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Rust snapshot_download: multiple files into cache
    let api = api_with_cache(&cache_dir);
    let snapshot_dir = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["*.json".to_string(), "*.md".to_string()])
                .build(),
        )
        .await
        .unwrap();
    assert!(snapshot_dir.exists());

    // Collect filenames Rust cached
    let mut rust_files: Vec<String> = Vec::new();
    let mut stack = vec![snapshot_dir.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(&snapshot_dir).unwrap();
                rust_files.push(rel.to_string_lossy().to_string());
            }
        }
    }
    assert!(rust_files.len() >= 2, "Expected at least 2 cached files, got: {rust_files:?}");

    // Python validates the cache structure and reads every file
    let rust_files_json = serde_json::to_string(&rust_files).unwrap();
    let script = format!(
        r#"
import json
import os
import sys

os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"

from huggingface_hub import hf_hub_download, scan_cache_dir

# 1. scan_cache_dir must find the repo with correct structure
cache_info = scan_cache_dir("{cache}")
repos = [r for r in cache_info.repos if "{repo_id}" in r.repo_id]
assert len(repos) == 1, f"Expected 1 gpt2 repo, found {{len(repos)}}"
repo = repos[0]
assert len(repo.revisions) >= 1, f"Expected >=1 revision, found {{len(repo.revisions)}}"

revision = next(iter(repo.revisions))
snapshot_path = str(revision.snapshot_path)
cached_rel_paths = set()
for f in revision.files:
    rel = os.path.relpath(str(f.file_path), snapshot_path)
    cached_rel_paths.add(rel)
rust_files = set(json.loads('{rust_files_json}'))
assert rust_files.issubset(cached_rel_paths), (
    f"Rust files {{rust_files}} not all found in Python scan: {{cached_rel_paths}}"
)

# 2. Every file must be readable via hf_hub_download with local_files_only=True
for filename in rust_files:
    path = hf_hub_download("{repo_id}", filename, local_files_only=True)
    size = os.path.getsize(path)
    assert size > 0, f"File {{filename}} is empty at {{path}}"

# 3. Verify config.json content is valid JSON with expected field
config_path = hf_hub_download("{repo_id}", "config.json", local_files_only=True)
with open(config_path) as f:
    config = json.load(f)
assert "model_type" in config, f"config.json missing model_type: {{list(config.keys())}}"

print("ALL_CHECKS_PASSED")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
        rust_files_json = rust_files_json,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Python validation failed.\nstdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("ALL_CHECKS_PASSED"),
        "Python did not complete all checks.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

// =============================================================================
// Xet cache tests
// =============================================================================

#[tokio::test]
async fn test_xet_download_to_cache() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let path = match api
        .model("hf-internal-testing", "tiny-gemma3")
        .download_file(&RepoDownloadFileParams::builder().filename("model.safetensors").build())
        .await
    {
        Ok(p) => p,
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("not found") || err_str.contains("Not Found") {
                eprintln!("Skipping xet cache test: repo not found");
                return;
            }
            panic!("Unexpected error: {err_str}");
        },
    };

    assert!(path.exists());
    assert!(path.to_string_lossy().contains("snapshots"));

    let file_size = std::fs::metadata(&path).unwrap().len();
    assert!(file_size > 1_000_000, "Expected large file, got {file_size} bytes");

    // Blob should exist (LFS files use SHA-256 etag = 64 hex chars)
    let repo_folder = std::fs::read_dir(cache_dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().contains("tiny-gemma3"))
        .expect("repo folder not found");
    let blobs_dir = repo_folder.path().join("blobs");
    assert!(blobs_dir.exists());
    let blobs: Vec<_> = std::fs::read_dir(&blobs_dir).unwrap().filter_map(|e| e.ok()).collect();
    assert!(!blobs.is_empty());

    // Symlink should point to blob
    #[cfg(not(windows))]
    {
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(meta.file_type().is_symlink());
        let target = std::fs::read_link(&path).unwrap();
        assert!(target.to_string_lossy().contains("blobs"));
    }

    // Ref should exist
    let refs_dir = repo_folder.path().join("refs");
    assert!(refs_dir.exists());
    let main_ref = refs_dir.join("main");
    assert!(main_ref.exists());

    // Second download should be a cache hit (same path returned)
    let path2 = api
        .model("hf-internal-testing", "tiny-gemma3")
        .download_file(&RepoDownloadFileParams::builder().filename("model.safetensors").build())
        .await
        .unwrap();
    assert_eq!(path, path2);

    // local_files_only should work
    let path3 = api
        .model("hf-internal-testing", "tiny-gemma3")
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("model.safetensors")
                .local_files_only(true)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(path, path3);
}

#[tokio::test]
async fn test_xet_snapshot_download_to_cache() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let snapshot_dir = match api
        .model("hf-internal-testing", "tiny-gemma3")
        .snapshot_download(&RepoSnapshotDownloadParams::default())
        .await
    {
        Ok(d) => d,
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("not found") || err_str.contains("Not Found") {
                eprintln!("Skipping xet snapshot test: repo not found");
                return;
            }
            panic!("Unexpected error: {err_str}");
        },
    };

    assert!(snapshot_dir.exists());
    assert!(snapshot_dir.to_string_lossy().contains("snapshots"));

    let repo_folder = find_repo_folder(cache_dir.path(), "tiny-gemma3");
    assert!(repo_folder.join("blobs").exists());
    assert!(repo_folder.join("refs").join("main").exists());

    let files = list_files_recursive(&snapshot_dir);
    assert!(!files.is_empty(), "Snapshot should contain files");
}

#[tokio::test]
async fn test_xet_cache_hit_second_download() {
    let Some(_) = api() else { return };
    let cache_dir = tempfile::tempdir().unwrap();
    let api = api_with_cache(cache_dir.path());

    let repo = api.model("hf-internal-testing", "tiny-gemma3");

    let path1 = match repo
        .download_file(&RepoDownloadFileParams::builder().filename("model.safetensors").build())
        .await
    {
        Ok(p) => p,
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("not found") || err_str.contains("Not Found") {
                eprintln!("Skipping xet cache hit test: repo not found");
                return;
            }
            panic!("Unexpected error: {err_str}");
        },
    };

    let blob = find_single_blob(cache_dir.path(), "tiny-gemma3");
    let mtime_before = std::fs::metadata(&blob).unwrap().modified().unwrap();

    // Second download: should be a cache hit (blob not rewritten)
    let path2 = repo
        .download_file(&RepoDownloadFileParams::builder().filename("model.safetensors").build())
        .await
        .unwrap();
    assert_eq!(path1, path2);

    let mtime_after = std::fs::metadata(&blob).unwrap().modified().unwrap();
    assert_eq!(mtime_before, mtime_after, "Blob should not be rewritten on cache hit");
}

// =============================================================================
// New interop tests: .no_exist and ref interop
// =============================================================================

#[tokio::test]
async fn test_interop_rust_no_exist_python_reads() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Rust: trigger a 404 to create a .no_exist marker
    let api = api_with_cache(&cache_dir);
    let _ = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("interop_no_exist_test_file.txt")
                .build(),
        )
        .await;

    // Python: try_to_load_from_cache should recognize the .no_exist marker
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub.file_download import try_to_load_from_cache
result = try_to_load_from_cache("{repo_id}", "interop_no_exist_test_file.txt")
# result should be _CACHED_NO_EXIST (a special sentinel) or None
# _CACHED_NO_EXIST is not None and not a string path
if result is None:
    print("NOT_FOUND_IN_CACHE")
elif isinstance(result, str):
    print(f"FOUND_FILE:{{result}}")
else:
    print("CACHED_NO_EXIST")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Python failed.\nstdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        stdout, "CACHED_NO_EXIST",
        "Python should recognize Rust's .no_exist marker.\nstdout: {stdout}\nstderr: {stderr}"
    );
}

#[tokio::test]
async fn test_interop_rust_ref_python_reads() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Rust: download to create refs/main
    let api = api_with_cache(&cache_dir);
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    // Python: hf_hub_download with local_files_only should find it via the ref
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "config.json", local_files_only=True)
assert os.path.exists(path), f"File not found: {{path}}"
print("REF_INTEROP_OK")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Python ref interop failed.\nstdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("REF_INTEROP_OK"));
}

#[tokio::test]
async fn test_interop_python_no_exist_rust_reads() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Python: trigger 404 to create .no_exist marker
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("{repo_id}", "python_no_exist_interop_test.txt")
except Exception:
    pass
print("DONE")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python failed: {}", String::from_utf8_lossy(&output.stderr));

    // Rust: local_files_only should find the .no_exist marker via resolve_from_cache_only
    let api = api_with_cache(&cache_dir);
    let result = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("python_no_exist_interop_test.txt")
                .local_files_only(true)
                .build(),
        )
        .await;
    assert!(
        matches!(result, Err(HFError::EntryNotFound { .. })),
        "Rust should recognize Python's .no_exist marker via local_files_only: {result:?}"
    );
}

#[tokio::test]
async fn test_interop_python_ref_rust_local_files_only() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Python downloads file (creates refs/main + blob + symlink)
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
hf_hub_download("{repo_id}", "config.json")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python failed: {}", String::from_utf8_lossy(&output.stderr));

    // Rust: local_files_only should find the file via Python's ref
    let api = api_with_cache(&cache_dir);
    let path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_files_only(true)
                .build(),
        )
        .await
        .unwrap();
    assert!(path.exists());
    let content: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(content.get("model_type").is_some());
}

#[tokio::test]
async fn test_interop_rust_snapshot_python_snapshot_reuse() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Rust snapshot_download first
    let api = api_with_cache(&cache_dir);
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["*.json".to_string()])
                .build(),
        )
        .await
        .unwrap();

    let repo_folder = find_repo_folder(&cache_dir, TEST_MODEL_CACHE_FRAGMENT);
    let blob_count_before = std::fs::read_dir(repo_folder.join("blobs")).unwrap().count();

    // Python snapshot_download same patterns — should reuse Rust's blobs
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import snapshot_download
snapshot_download("{repo_id}", allow_patterns=["*.json"])
print("OK")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python snapshot failed: {}", String::from_utf8_lossy(&output.stderr));

    let blob_count_after = std::fs::read_dir(repo_folder.join("blobs")).unwrap().count();
    assert_eq!(blob_count_before, blob_count_after, "Python should reuse Rust's blobs");
}

#[tokio::test]
async fn test_interop_dataset_repo_type() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Rust downloads a dataset file
    let api = api_with_cache(&cache_dir);
    api.dataset(TEST_DATASET_PARTS.0, TEST_DATASET_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("README.md").build())
        .await
        .unwrap();

    // Python reads the same file with local_files_only
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{dataset_repo_id}", "README.md", repo_type="dataset", local_files_only=True)
assert os.path.exists(path), f"Not found: {{path}}"
assert os.path.getsize(path) > 0
print("DATASET_INTEROP_OK")
"#,
        cache = cache_dir.display(),
        token = token,
        dataset_repo_id = TEST_DATASET_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Python dataset interop failed.\nstdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("DATASET_INTEROP_OK"));
}

#[cfg(not(windows))]
#[tokio::test]
async fn test_interop_symlink_target_format() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Python downloads file — creates the canonical symlink format
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
path = hf_hub_download("{repo_id}", "config.json")
link = os.readlink(path)
print(link)
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python failed: {}", String::from_utf8_lossy(&output.stderr));
    let python_link_target = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Rust downloads same file into a fresh cache
    let cache_dir2 = base_dir.path().join("cache2");
    std::fs::create_dir_all(&cache_dir2).unwrap();
    let api = api_with_cache(&cache_dir2);
    let rust_path = api
        .model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();
    let rust_link_target = std::fs::read_link(&rust_path).unwrap().to_string_lossy().to_string();

    // Both should use the same relative symlink format (e.g. "../../blobs/<etag>")
    assert_eq!(python_link_target, rust_link_target, "Symlink target format should match between Python and Rust");
}

#[tokio::test]
async fn test_interop_conditional_request_reuse() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Python downloads file — creates blob + symlink + ref
    let script = format!(
        r#"
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import hf_hub_download
hf_hub_download("{repo_id}", "config.json")
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    assert!(output.status.success(), "Python failed: {}", String::from_utf8_lossy(&output.stderr));

    // Record blob mtime after Python's download
    let repo_folder = find_repo_folder(&cache_dir, TEST_MODEL_CACHE_FRAGMENT);
    let blobs_dir = repo_folder.join("blobs");
    let blob = std::fs::read_dir(&blobs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .next()
        .unwrap()
        .path();
    let mtime_before = std::fs::metadata(&blob).unwrap().modified().unwrap();

    // Rust downloads same file — should read etag from Python's symlink,
    // send If-None-Match, get 304, and NOT rewrite the blob
    let api = api_with_cache(&cache_dir);
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .download_file(&RepoDownloadFileParams::builder().filename("config.json").build())
        .await
        .unwrap();

    let mtime_after = std::fs::metadata(&blob).unwrap().modified().unwrap();
    assert_eq!(mtime_before, mtime_after, "Rust should use conditional request (304) and not rewrite Python's blob");
}

#[tokio::test]
async fn test_interop_scan_cache_counts_match() {
    let Some(_) = api() else { return };
    let base_dir = tempfile::tempdir().unwrap();
    let cache_dir = base_dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let Some(venv_dir) = setup_python_venv(base_dir.path()) else {
        return;
    };
    let python = python_bin(&venv_dir);
    let token = resolve_prod_token().expect("HF_TOKEN or HF_PROD_TOKEN required");

    // Download multiple files via both libraries to populate the cache
    let api = api_with_cache(&cache_dir);
    api.model(TEST_MODEL_PARTS.0, TEST_MODEL_PARTS.1)
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .allow_patterns(vec!["*.json".to_string(), "*.md".to_string()])
                .build(),
        )
        .await
        .unwrap();

    // Python: scan_cache_dir and report metrics
    let script = format!(
        r#"
import json
import os
os.environ["HF_HUB_CACHE"] = "{cache}"
os.environ["HF_TOKEN"] = "{token}"
from huggingface_hub import scan_cache_dir
info = scan_cache_dir("{cache}")
repos = [r for r in info.repos if "{repo_id}" in r.repo_id]
assert len(repos) == 1
repo = repos[0]
print(json.dumps({{
    "nb_files": repo.nb_files,
    "size_on_disk": repo.size_on_disk,
    "num_revisions": len(repo.revisions),
    "repo_id": repo.repo_id,
}}))
"#,
        cache = cache_dir.display(),
        token = token,
        repo_id = TEST_MODEL_REPO_ID,
    );
    let output = std::process::Command::new(&python).args(["-c", &script]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "Python scan failed.\nstdout: {stdout}\nstderr: {stderr}");

    let python_metrics: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    // Count unique blobs and total size from Rust's perspective
    let repo_folder = find_repo_folder(&cache_dir, TEST_MODEL_CACHE_FRAGMENT);
    let blobs_dir = repo_folder.join("blobs");
    let mut rust_nb_files = 0usize;
    let mut rust_size: u64 = 0;
    for entry in std::fs::read_dir(&blobs_dir).unwrap().filter_map(|e| e.ok()) {
        rust_nb_files += 1;
        rust_size += std::fs::metadata(entry.path()).unwrap().len();
    }

    assert_eq!(
        rust_nb_files,
        python_metrics["nb_files"].as_u64().unwrap() as usize,
        "nb_files should match between Rust blob count and Python scan_cache_dir"
    );
    assert_eq!(
        rust_size,
        python_metrics["size_on_disk"].as_u64().unwrap(),
        "size_on_disk should match between Rust and Python"
    );
}
