//! Integration tests for bucket xet-based file transfers.
//!
//! Directly exercises `HFBucket::upload_files` (which fans out to
//! `HFBucket::xet_upload`) and `HFBucket::download_files` without going
//! through the `sync` planner.
//!
//! Requires:
//!   - HF_TOKEN environment variable
//!   - HF_TEST_WRITE=1 (creates and deletes buckets)
//!
//! Run: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p hf-hub --test
//! bucket_xet_transfer_test -- --nocapture

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::TryStreamExt;
use hf_hub::buckets::BucketTreeEntry;
use hf_hub::{HFBucket, HFClient, HFClientBuilder, HFError};
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
    let name = format!("hfrs-xet-upload-test-{suffix}");
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
    let _ = client.delete_bucket().bucket_id(bucket_id).missing_ok(true).send().await;
}

/// Poll the bucket tree until `path` is visible — the batch endpoint can
/// return before the tree index is consistent on the CI backend.
async fn wait_for_bucket_file(bucket: &HFBucket, path: &str) -> BucketTreeEntry {
    for _ in 0..40 {
        let entries: Vec<BucketTreeEntry> =
            bucket.list_tree().recursive(true).send().unwrap().try_collect().await.unwrap();
        if let Some(entry) = entries
            .iter()
            .find(|e| matches!(e, BucketTreeEntry::File { path: p, .. } if p == path))
        {
            return entry.clone();
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("bucket tree never reflected uploaded file {path}");
}

fn generate_random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut data = vec![0u8; size];
    rng.fill(&mut data[..]);
    data
}

#[tokio::test]
async fn test_bucket_upload_small_text_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    let tmp = tempfile::tempdir().unwrap();
    let data = b"Hello from the bucket xet transfer test!".to_vec();
    let local_file = tmp.path().join("greeting.txt");
    std::fs::write(&local_file, &data).unwrap();

    bucket
        .upload_files()
        .files(vec![(local_file, "greeting.txt".to_string())])
        .send()
        .await
        .expect("bucket upload_files should succeed");

    let entry = wait_for_bucket_file(&bucket, "greeting.txt").await;
    match entry {
        BucketTreeEntry::File { size, .. } => assert_eq!(size, data.len() as u64),
        BucketTreeEntry::Directory { .. } => panic!("expected file entry"),
    }

    let dl_dir = tempfile::tempdir().unwrap();
    bucket
        .download_files()
        .files(vec![("greeting.txt".to_string(), dl_dir.path().join("greeting.txt"))])
        .send()
        .await
        .unwrap();
    assert_eq!(std::fs::read(dl_dir.path().join("greeting.txt")).unwrap(), data);

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_bucket_upload_multiple_files() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    let tmp = tempfile::tempdir().unwrap();
    let files: Vec<(&str, &[u8])> = vec![
        ("a.txt", b"alpha content"),
        ("b.txt", b"bravo content"),
        ("subdir/nested.txt", b"nested content"),
    ];
    let mut upload_args: Vec<(std::path::PathBuf, String)> = Vec::new();
    for (remote, content) in &files {
        let local = tmp.path().join(remote);
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&local, content).unwrap();
        upload_args.push((local, remote.to_string()));
    }

    bucket
        .upload_files()
        .files(upload_args)
        .send()
        .await
        .expect("bucket upload_files with multiple files should succeed");

    for (remote, content) in &files {
        let entry = wait_for_bucket_file(&bucket, remote).await;
        match entry {
            BucketTreeEntry::File { size, .. } => assert_eq!(size, content.len() as u64),
            BucketTreeEntry::Directory { .. } => panic!("expected file entry for {remote}"),
        }
    }

    let dl_dir = tempfile::tempdir().unwrap();
    let download_pairs: Vec<(String, std::path::PathBuf)> = files
        .iter()
        .map(|(remote, _)| (remote.to_string(), dl_dir.path().join(remote)))
        .collect();
    bucket.download_files().files(download_pairs).send().await.unwrap();
    for (remote, content) in &files {
        let downloaded = std::fs::read(dl_dir.path().join(remote)).unwrap();
        assert_eq!(downloaded.as_slice(), *content, "content mismatch for {remote}");
    }

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_bucket_upload_large_random_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    const SIZE: usize = 20 * 1024 * 1024;
    let data = generate_random_bytes(SIZE);
    let expected_hash = sha256_hex(&data);

    let tmp = tempfile::tempdir().unwrap();
    let local_file = tmp.path().join("big.bin");
    std::fs::write(&local_file, &data).unwrap();
    drop(data);

    bucket
        .upload_files()
        .files(vec![(local_file, "big.bin".to_string())])
        .send()
        .await
        .expect("large bucket upload via xet should succeed");

    let entry = wait_for_bucket_file(&bucket, "big.bin").await;
    match entry {
        BucketTreeEntry::File { size, .. } => assert_eq!(size, SIZE as u64),
        BucketTreeEntry::Directory { .. } => panic!("expected file entry"),
    }

    let dl_dir = tempfile::tempdir().unwrap();
    bucket
        .download_files()
        .files(vec![("big.bin".to_string(), dl_dir.path().join("big.bin"))])
        .send()
        .await
        .unwrap();
    let downloaded = std::fs::read(dl_dir.path().join("big.bin")).unwrap();
    assert_eq!(downloaded.len(), SIZE);
    assert_eq!(sha256_hex(&downloaded), expected_hash);

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_bucket_upload_empty_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    let tmp = tempfile::tempdir().unwrap();
    let local_file = tmp.path().join("empty.bin");
    std::fs::write(&local_file, b"").unwrap();

    bucket
        .upload_files()
        .files(vec![(local_file, "empty.bin".to_string())])
        .send()
        .await
        .expect("empty-file bucket upload should succeed");

    let entry = wait_for_bucket_file(&bucket, "empty.bin").await;
    match entry {
        BucketTreeEntry::File { size, .. } => assert_eq!(size, 0),
        BucketTreeEntry::Directory { .. } => panic!("expected file entry"),
    }

    let dl_dir = tempfile::tempdir().unwrap();
    bucket
        .download_files()
        .files(vec![("empty.bin".to_string(), dl_dir.path().join("empty.bin"))])
        .send()
        .await
        .unwrap();
    assert!(std::fs::read(dl_dir.path().join("empty.bin")).unwrap().is_empty());

    delete_test_bucket(&client, &namespace, &name).await;
}

/// `delete_bucket` should remove a bucket even when it still contains uploaded files.
/// After deletion, follow-up operations against the same bucket id must surface a
/// `BucketNotFound` (or `Http`) error rather than appearing to succeed.
#[tokio::test]
async fn test_delete_bucket_with_files() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);
    let bucket_id = format!("{namespace}/{name}");

    let tmp = tempfile::tempdir().unwrap();
    let files: Vec<(&str, &[u8])> = vec![
        ("first.txt", b"first contents"),
        ("nested/second.bin", b"second contents"),
    ];
    let mut upload_args: Vec<(std::path::PathBuf, String)> = Vec::new();
    for (remote, content) in &files {
        let local = tmp.path().join(remote);
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&local, content).unwrap();
        upload_args.push((local, remote.to_string()));
    }
    bucket
        .upload_files()
        .files(upload_args)
        .send()
        .await
        .expect("upload to bucket failed");
    for (remote, _) in &files {
        wait_for_bucket_file(&bucket, remote).await;
    }

    // Delete the bucket while it still has files.
    client
        .delete_bucket()
        .bucket_id(&bucket_id)
        .send()
        .await
        .expect("delete_bucket should succeed even when the bucket has files");

    // A second delete (without missing_ok) must surface BucketNotFound.
    let err = client.delete_bucket().bucket_id(&bucket_id).send().await.unwrap_err();
    assert!(
        matches!(err, HFError::BucketNotFound { .. } | HFError::Http { .. }),
        "expected BucketNotFound after deletion, got {err:?}",
    );

    // missing_ok=true should still succeed against the now-deleted bucket.
    client
        .delete_bucket()
        .bucket_id(&bucket_id)
        .missing_ok(true)
        .send()
        .await
        .expect("delete_bucket with missing_ok should succeed when bucket is gone");
}
