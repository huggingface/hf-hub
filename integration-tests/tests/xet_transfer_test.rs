//! Integration tests for xet-based file transfers.
//!
//! Tests uploading large/binary files that require xet storage and
//! downloading files from xet-enabled repositories.
//!
//! Requires:
//!   - HF_TOKEN environment variable
//!   - HF_TEST_WRITE=1 (creates and deletes repos)
//!
//! Run: source ~/hf/prod_token && HF_TEST_WRITE=1 cargo test -p hf-hub --test xet_transfer_test
//! -- --nocapture
//!
//! These tests are slow (uploading large files) and create real repositories.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use hf_hub::repository::AddSource;
use hf_hub::{HFClient, HFClientBuilder, HFRepository, RepoType};
use integration_tests::test_utils::*;
use rand::RngExt;
use tokio::sync::OnceCell;

static WHOAMI_USERNAME: OnceCell<String> = OnceCell::const_new();

/// Client for write tests — hub-ci in CI, default endpoint locally.
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

/// Client for read-only tests against hardcoded production repos.
fn prod_api() -> Option<HFClient> {
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
        let token = std::env::var(HF_TOKEN).ok()?;
        Some(HFClientBuilder::new().token(token).build().expect("Failed to create HFClient"))
    }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn unique_suffix() -> String {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}{:x}-{count}", t.as_secs(), t.subsec_nanos())
}

/// Split an `"owner/name"` repo_id into an [`HFRepository`] handle.
fn repo_handle(client: &HFClient, owner: &str, name: &str) -> HFRepository {
    client.repo(RepoType::Model, owner, name)
}

async fn create_test_repo(client: &HFClient, suffix: &str) -> (String, String) {
    let username = WHOAMI_USERNAME
        .get_or_init(|| async { client.whoami().send().await.expect("whoami failed").username })
        .await;
    let name = format!("hf-hub-xet-test-{suffix}");
    let repo_id = format!("{username}/{name}");
    client
        .create_repo()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(true)
        .send()
        .await
        .expect("create_repo failed");
    (username.clone(), name)
}

async fn delete_test_repo(client: &HFClient, repo_id: &str) {
    let _ = client.delete_repo().repo_id(repo_id).send().await;
}

fn generate_random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::rng();
    let mut data = vec![0u8; size];
    rng.fill(&mut data[..]);
    data
}

// --- Small file tests (inline NDJSON, no xet needed) ---

#[tokio::test]
async fn test_upload_small_text_file_roundtrip() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    let data = b"Hello from the xet transfer test!".to_vec();
    let commit = repo
        .upload_file()
        .source(AddSource::bytes(data.clone()))
        .path_in_repo("greeting.txt")
        .commit_message("upload small text file")
        .send()
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("greeting.txt")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), data);

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_upload_empty_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    repo.upload_file()
        .source(AddSource::bytes(vec![]))
        .path_in_repo("empty.bin")
        .commit_message("upload empty file")
        .send()
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("empty.bin")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert!(std::fs::read(&path).unwrap().is_empty());

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_upload_then_overwrite_same_path() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    repo.upload_file()
        .source(AddSource::bytes(b"version 1"))
        .path_in_repo("versioned.txt")
        .commit_message("v1")
        .send()
        .await
        .unwrap();

    repo.upload_file()
        .source(AddSource::bytes(b"version 2 updated"))
        .path_in_repo("versioned.txt")
        .commit_message("v2")
        .send()
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("versioned.txt")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "version 2 updated");

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_upload_file_with_nested_path() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    let data = b"deeply nested content".to_vec();
    repo.upload_file()
        .source(AddSource::bytes(data.clone()))
        .path_in_repo("a/b/c/d/deep.txt")
        .commit_message("upload nested file")
        .send()
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("a/b/c/d/deep.txt")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert_eq!(path, dir.path().join("a/b/c/d/deep.txt"));
    assert_eq!(std::fs::read(&path).unwrap(), data);

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_upload_from_file_path() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    let tmp = tempfile::tempdir().unwrap();
    let data = b"content from a local file on disk".to_vec();
    let expected_hash = sha256_hex(&data);
    let local_file = tmp.path().join("upload_me.txt");
    std::fs::write(&local_file, &data).unwrap();

    repo.upload_file()
        .source(AddSource::file(local_file))
        .path_in_repo("uploaded_from_path.txt")
        .commit_message("upload from file path")
        .send()
        .await
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file()
        .filename("uploaded_from_path.txt")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert_eq!(sha256_hex(&std::fs::read(&path).unwrap()), expected_hash);

    delete_test_repo(&client, &repo_id).await;
}

// --- Large file / xet tests ---
// These test the xet upload path for files too large for inline NDJSON.
// The Hub rejects binary files > ~10MB via regular commit and requires xet.

#[tokio::test]
async fn test_download_from_known_xet_repo() {
    let Some(client) = prod_api() else { return };

    let dir = tempfile::tempdir().unwrap();
    let result = repo_handle(&client, "hf-internal-testing", "tiny-gemma3")
        .download_file()
        .filename("model.safetensors")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await;
    match result {
        Ok(path) => {
            assert!(path.exists());
            let metadata = std::fs::metadata(&path).unwrap();
            assert!(metadata.len() > 0);
        },
        Err(e) => {
            let err_str = e.to_string();
            assert!(
                err_str.contains("not found") || err_str.contains("Not Found"),
                "Expected success or not-found for xet repo, got: {err_str}"
            );
        },
    }
}

#[tokio::test]
async fn test_upload_75mb_random_data_and_verify() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let (owner, name) = create_test_repo(&client, &unique_suffix()).await;
    let repo_id = format!("{owner}/{name}");
    let repo = repo_handle(&client, &owner, &name);

    let data_75mb = generate_random_bytes(75 * 1024 * 1024);
    let expected_hash = sha256_hex(&data_75mb);

    let tmp = tempfile::tempdir().unwrap();
    let local_file = tmp.path().join("model.safetensors");
    std::fs::write(&local_file, &data_75mb).unwrap();
    drop(data_75mb);

    let commit = repo
        .upload_file()
        .source(AddSource::file(local_file))
        .path_in_repo("model.safetensors")
        .commit_message("upload 75MB random data")
        .send()
        .await
        .expect("Large file upload via xet should succeed");
    assert!(commit.commit_oid.is_some());

    assert!(repo.file_exists().filename("model.safetensors").send().await.unwrap());

    let dl_dir = tempfile::tempdir().unwrap();
    let downloaded_path = repo
        .download_file()
        .filename("model.safetensors")
        .local_dir(dl_dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert!(downloaded_path.exists());

    let downloaded_data = std::fs::read(&downloaded_path).unwrap();
    assert_eq!(downloaded_data.len(), 75 * 1024 * 1024);
    assert_eq!(sha256_hex(&downloaded_data), expected_hash);

    delete_test_repo(&client, &repo_id).await;
}

// --- Xet streaming / range download tests ---

#[tokio::test]
async fn test_xet_download_stream_full() {
    let Some(client) = prod_api() else { return };

    let repo = repo_handle(&client, "hf-internal-testing", "tiny-gemma3");

    let result = repo.download_file_stream().filename("model.safetensors").send().await;

    match result {
        Ok((content_length, stream)) => {
            assert!(content_length.is_some());
            let len = content_length.unwrap();
            assert!(len > 0);

            futures::pin_mut!(stream);
            let mut total = 0u64;
            while let Some(chunk) = stream.next().await {
                total += chunk.unwrap().len() as u64;
            }
            assert_eq!(total, len);
        },
        Err(e) => {
            let err_str = e.to_string();
            assert!(
                err_str.contains("not found") || err_str.contains("Not Found"),
                "Expected success or not-found for xet repo, got: {err_str}"
            );
        },
    }
}

#[tokio::test]
async fn test_xet_download_stream_range() {
    let Some(client) = prod_api() else { return };

    let repo = repo_handle(&client, "hf-internal-testing", "tiny-gemma3");

    // Download first 1024 bytes via range
    let result = repo
        .download_file_stream()
        .filename("model.safetensors")
        .range(0..1024u64)
        .send()
        .await;

    match result {
        Ok((content_length, stream)) => {
            assert_eq!(content_length, Some(1024));

            futures::pin_mut!(stream);
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                bytes.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(bytes.len(), 1024);
        },
        Err(e) => {
            let err_str = e.to_string();
            assert!(
                err_str.contains("not found") || err_str.contains("Not Found"),
                "Expected success or not-found for xet repo, got: {err_str}"
            );
        },
    }
}

#[tokio::test]
async fn test_xet_download_stream_range_middle() {
    let Some(client) = prod_api() else { return };

    let repo = repo_handle(&client, "hf-internal-testing", "tiny-gemma3");

    // Download bytes 1000..2000
    let result = repo
        .download_file_stream()
        .filename("model.safetensors")
        .range(1000..2000u64)
        .send()
        .await;

    match result {
        Ok((content_length, stream)) => {
            assert_eq!(content_length, Some(1000));

            futures::pin_mut!(stream);
            let mut bytes = Vec::new();
            while let Some(chunk) = stream.next().await {
                bytes.extend_from_slice(&chunk.unwrap());
            }
            assert_eq!(bytes.len(), 1000);
        },
        Err(e) => {
            let err_str = e.to_string();
            assert!(
                err_str.contains("not found") || err_str.contains("Not Found"),
                "Expected success or not-found for xet repo, got: {err_str}"
            );
        },
    }
}
