//! Integration tests for `HFBucket::download_file_stream`.
//!
//! Exercises the wasm-friendly streaming bucket-download path against the
//! live Hub. Uploads a known payload, calls the streaming API, drains the
//! stream, and asserts on byte equality and reported `content_length`.
//!
//! Requires:
//!   - HF_TOKEN environment variable
//!   - HF_TEST_WRITE=1 (creates and deletes buckets)
//!
//! Run: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p integration-tests \
//!      --test bucket_download_stream_test -- --nocapture

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::{StreamExt, TryStreamExt};
use hf_hub::buckets::{BucketTreeEntry, BucketUpload};
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
    let name = format!("hfrs-bucket-stream-dl-test-{suffix}");
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

async fn drain(stream: hf_hub::repository::download::HFByteStream) -> Vec<u8> {
    let mut stream = stream;
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.expect("stream chunk error"));
    }
    out
}

#[tokio::test]
async fn test_bucket_download_stream_full_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    let tmp = tempfile::tempdir().unwrap();
    let payload = b"Hello from the bucket download_file_stream test!\n".to_vec();
    let local_file = tmp.path().join("greeting.txt");
    std::fs::write(&local_file, &payload).unwrap();

    bucket
        .upload_files()
        .files(vec![BucketUpload::new(local_file, "greeting.txt")])
        .send()
        .await
        .expect("bucket upload_files should succeed");

    wait_for_bucket_file(&bucket, "greeting.txt").await;

    let (content_length, stream) = bucket
        .download_file_stream()
        .remote_path("greeting.txt")
        .send()
        .await
        .expect("download_file_stream should succeed");

    assert_eq!(content_length, Some(payload.len() as u64), "reported content_length must match upload size");
    let bytes = drain(stream).await;
    assert_eq!(bytes, payload, "downloaded bytes must match uploaded payload");

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_bucket_download_stream_xet_backed_large_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    const SIZE: usize = 20 * 1024 * 1024;
    let payload = generate_random_bytes(SIZE);
    let expected_len = payload.len();

    let tmp = tempfile::tempdir().unwrap();
    let local_file = tmp.path().join("big.bin");
    std::fs::write(&local_file, &payload).unwrap();

    bucket
        .upload_files()
        .files(vec![BucketUpload::new(local_file, "big.bin")])
        .send()
        .await
        .expect("large bucket upload via xet should succeed");

    let entry = wait_for_bucket_file(&bucket, "big.bin").await;
    let xet_hash = match &entry {
        BucketTreeEntry::File { xet_hash, .. } => xet_hash.clone(),
        _ => panic!("expected file entry"),
    };
    assert!(!xet_hash.is_empty(), "20MB bucket upload should be xet-backed");

    let (content_length, stream) = bucket
        .download_file_stream()
        .remote_path("big.bin")
        .send()
        .await
        .expect("download_file_stream should succeed for xet-backed file");

    assert_eq!(content_length, Some(expected_len as u64));
    let bytes = drain(stream).await;
    assert_eq!(bytes.len(), expected_len, "byte count mismatch");
    assert_eq!(bytes, payload, "downloaded bytes must match uploaded payload");

    delete_test_bucket(&client, &namespace, &name).await;
}

#[tokio::test]
async fn test_bucket_download_stream_nonexistent_file_returns_error() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (namespace, name) = create_test_bucket(&client, &unique_suffix()).await;
    let bucket = client.bucket(&namespace, &name);

    let result = bucket.download_file_stream().remote_path("does/not/exist.bin").send().await;

    match result {
        Ok(_) => panic!("download_file_stream of missing file must return an error"),
        Err(HFError::EntryNotFound { .. }) => {},
        Err(other) => panic!("expected EntryNotFound, got: {other:?}"),
    }

    delete_test_bucket(&client, &namespace, &name).await;
}
