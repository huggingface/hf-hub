//! Integration tests for bucket sync operations.
//!
//! Requires:
//!   - HF_TOKEN environment variable
//!   - HF_TEST_WRITE=1 (creates and deletes buckets)
//!
//! Run: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p hf-hub --test bucket_sync_test --
//! --nocapture

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::TryStreamExt;
use hf_hub::buckets::BucketTreeEntry;
use hf_hub::buckets::sync::{BucketSyncAction, BucketSyncDirection};
use hf_hub::{HFBucket, HFClient, HFClientBuilder};
use integration_tests::test_utils::*;
use rand::RngExt;
use tokio::sync::OnceCell;

static WHOAMI_USERNAME: OnceCell<String> = OnceCell::const_new();
static COUNTER: AtomicU32 = AtomicU32::new(0);

fn api() -> Option<HFClient> {
    if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        Some(
            HFClientBuilder::new()
                .token(token)
                .endpoint(HUB_CI_ENDPOINT)
                .build()
                .expect("Failed to create HFClient"),
        )
    } else {
        let token = std::env::var(HF_TOKEN).ok()?;
        Some(HFClientBuilder::new().token(token).build().expect("Failed to create HFClient"))
    }
}

fn unique_suffix() -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}{:x}-{count}", t.as_secs(), t.subsec_nanos())
}

async fn get_username(client: &HFClient) -> String {
    WHOAMI_USERNAME
        .get_or_init(|| async { client.whoami().send().await.expect("whoami failed").username })
        .await
        .clone()
}

async fn create_test_bucket(client: &HFClient, suffix: &str) -> (String, String) {
    let username = get_username(client).await;
    let name = format!("hfrs-sync-test-{suffix}");
    client
        .create_bucket()
        .namespace(&username)
        .name(&name)
        .private(true)
        .exist_ok(true)
        .send()
        .await
        .expect("create_bucket failed");
    (username, name)
}

async fn delete_test_bucket(client: &HFClient, namespace: &str, name: &str) {
    let bucket_id = format!("{namespace}/{name}");
    let _ = client.delete_bucket().bucket_id(&bucket_id).missing_ok(true).send().await;
}

/// Poll the bucket tree until `path` is visible. The batch endpoint returns before
/// the tree index is consistent on the CI backend, so a tiny upload can race with a
/// subsequent `list_tree` call that still sees an empty bucket.
async fn wait_for_bucket_file(bucket: &HFBucket, path: &str) {
    for _ in 0..20 {
        let entries: Vec<BucketTreeEntry> =
            bucket.list_tree().recursive(true).send().unwrap().try_collect().await.unwrap();
        if entries
            .iter()
            .any(|e| matches!(e, BucketTreeEntry::File { path: p, .. } if p == path))
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("bucket tree never reflected uploaded file {path}");
}

fn create_local_files(dir: &std::path::Path, files: &[(&str, &[u8])]) {
    for (path, content) in files {
        let full_path = dir.join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(full_path, content).unwrap();
    }
}

const RANDOM_BIN_SIZE: usize = 1024 * 1024; // 1 MB

fn generate_random_bin(dir: &std::path::Path) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut data = vec![0u8; RANDOM_BIN_SIZE];
    rng.fill(&mut data[..]);
    std::fs::write(dir.join("d.bin"), &data).unwrap();
    data
}

#[tokio::test]
async fn test_sync_upload_new_files() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(local_dir.path(), &[("file1.txt", b"hello world"), ("subdir/file2.txt", b"nested content")]);
    generate_random_bin(local_dir.path());

    let plan = bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.uploads(), 3);
    assert_eq!(plan.downloads(), 0);
    assert_eq!(plan.deletes(), 0);
    assert!(plan.operations.iter().all(|op| op.action == BucketSyncAction::Upload));
    assert!(plan.operations.iter().all(|op| op.reason == "new file"));

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_upload_then_download() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let upload_dir = tempfile::tempdir().unwrap();
    create_local_files(
        upload_dir.path(),
        &[
            ("a.txt", b"content a"),
            ("b.txt", b"content b"),
            ("sub/c.txt", b"content c"),
        ],
    );
    let bin_data = generate_random_bin(upload_dir.path());
    let bin_hash = sha256_hex(&bin_data);

    bucket
        .sync()
        .local_path(upload_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();

    let download_dir = tempfile::tempdir().unwrap();
    let plan = bucket
        .sync()
        .local_path(download_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Download)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.downloads(), 4);
    assert_eq!(plan.uploads(), 0);

    assert_eq!(std::fs::read_to_string(download_dir.path().join("a.txt")).unwrap(), "content a");
    assert_eq!(std::fs::read_to_string(download_dir.path().join("b.txt")).unwrap(), "content b");
    assert_eq!(std::fs::read_to_string(download_dir.path().join("sub/c.txt")).unwrap(), "content c");
    let downloaded_bin = std::fs::read(download_dir.path().join("d.bin")).unwrap();
    assert_eq!(downloaded_bin.len(), RANDOM_BIN_SIZE);
    assert_eq!(sha256_hex(&downloaded_bin), bin_hash);

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_upload_skip_identical() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(local_dir.path(), &[("file.txt", b"same content")]);

    bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();

    // Second sync: file should be skipped (same size, ignore_times to avoid mtime issues)
    let plan = bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .ignore_times(true)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.uploads(), 0);
    assert_eq!(plan.skips(), 1);
    assert_eq!(plan.operations[0].reason, "same size");

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_upload_with_delete() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    // Upload two files
    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(local_dir.path(), &[("keep.txt", b"keep me"), ("remove.txt", b"remove me")]);

    bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();

    // Remove one local file, sync with --delete
    std::fs::remove_file(local_dir.path().join("remove.txt")).unwrap();

    let plan = bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .delete(true)
        .ignore_times(true)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.deletes(), 1);
    let delete_op = plan.operations.iter().find(|op| op.action == BucketSyncAction::Delete).unwrap();
    assert_eq!(delete_op.path, "remove.txt");

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_with_include_filter() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(
        local_dir.path(),
        &[
            ("data.txt", b"include me"),
            ("image.png", b"exclude me"),
            ("notes.txt", b"include me too"),
        ],
    );

    let plan = bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .include(vec!["*.txt".to_string()])
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.uploads(), 2);
    assert!(plan.operations.iter().all(|op| op.path.ends_with(".txt")));

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_with_exclude_filter() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(
        local_dir.path(),
        &[
            ("data.txt", b"include me"),
            ("secret.key", b"exclude me"),
            ("notes.txt", b"include me too"),
        ],
    );

    let plan = bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .exclude(vec!["*.key".to_string()])
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.uploads(), 2);
    assert!(plan.operations.iter().all(|op| !op.path.ends_with(".key")));

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_with_prefix() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    let local_dir = tempfile::tempdir().unwrap();
    create_local_files(local_dir.path(), &[("file1.txt", b"in prefix"), ("file2.txt", b"also in prefix")]);

    // Upload to a prefix
    bucket
        .sync()
        .local_path(local_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .prefix("my-prefix")
        .send()
        .await
        .unwrap();

    // Download from that prefix to a new dir
    let download_dir = tempfile::tempdir().unwrap();
    let plan = bucket
        .sync()
        .local_path(download_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Download)
        .prefix("my-prefix")
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.downloads(), 2);
    assert_eq!(std::fs::read_to_string(download_dir.path().join("file1.txt")).unwrap(), "in prefix");

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_download_with_delete() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    // Upload a file
    let upload_dir = tempfile::tempdir().unwrap();
    create_local_files(upload_dir.path(), &[("remote.txt", b"from bucket")]);

    bucket
        .sync()
        .local_path(upload_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();
    wait_for_bucket_file(&bucket, "remote.txt").await;

    // Create local dir with an extra file
    let download_dir = tempfile::tempdir().unwrap();
    create_local_files(download_dir.path(), &[("local_only.txt", b"should be deleted")]);

    let plan = bucket
        .sync()
        .local_path(download_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Download)
        .delete(true)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.downloads(), 1);
    assert_eq!(plan.deletes(), 1);

    assert!(download_dir.path().join("remote.txt").exists());
    assert!(!download_dir.path().join("local_only.txt").exists());

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_existing_flag() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    // Upload one file
    let upload_dir = tempfile::tempdir().unwrap();
    create_local_files(upload_dir.path(), &[("existing.txt", b"already here")]);

    bucket
        .sync()
        .local_path(upload_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();

    // Now try to upload two files with --existing: only existing.txt should sync
    let upload_dir2 = tempfile::tempdir().unwrap();
    create_local_files(upload_dir2.path(), &[("existing.txt", b"updated content"), ("new.txt", b"brand new")]);

    let plan = bucket
        .sync()
        .local_path(upload_dir2.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .existing(true)
        .verbose(true)
        .send()
        .await
        .unwrap();

    // existing.txt should be uploaded (size differs), new.txt should be skipped
    assert!(plan.uploads() <= 1);
    let new_file_ops: Vec<_> = plan.operations.iter().filter(|op| op.path == "new.txt").collect();
    assert!(new_file_ops.is_empty() || new_file_ops[0].action == BucketSyncAction::Skip);

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_sync_ignore_existing_flag() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let suffix = unique_suffix();
    let (namespace, name) = create_test_bucket(&client, &suffix).await;
    let bucket = client.bucket(&namespace, &name);

    // Upload one file
    let upload_dir = tempfile::tempdir().unwrap();
    create_local_files(upload_dir.path(), &[("existing.txt", b"original")]);

    bucket
        .sync()
        .local_path(upload_dir.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .send()
        .await
        .unwrap();

    // Upload with --ignore-existing: existing.txt skipped, new.txt uploaded
    let upload_dir2 = tempfile::tempdir().unwrap();
    create_local_files(upload_dir2.path(), &[("existing.txt", b"modified"), ("new.txt", b"brand new")]);

    let plan = bucket
        .sync()
        .local_path(upload_dir2.path().to_path_buf())
        .direction(BucketSyncDirection::Upload)
        .ignore_existing(true)
        .verbose(true)
        .send()
        .await
        .unwrap();

    assert_eq!(plan.uploads(), 1);
    let uploaded: Vec<_> = plan
        .operations
        .iter()
        .filter(|op| op.action == BucketSyncAction::Upload)
        .collect();
    assert_eq!(uploaded[0].path, "new.txt");

    let skipped: Vec<_> = plan
        .operations
        .iter()
        .filter(|op| op.action == BucketSyncAction::Skip)
        .collect();
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].path, "existing.txt");

    delete_test_bucket(&client, &namespace, &name).await;
}
