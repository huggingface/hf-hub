//! Integration tests for xet-based file transfers.
//!
//! Tests uploading large/binary files that require xet storage, and
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
use hf_hub::test_utils::*;
use hf_hub::types::{
    AddSource, CreateRepoParams, DeleteRepoParams, RepoDownloadFileParams, RepoDownloadFileStreamParams,
    RepoFileExistsParams, RepoUploadFileParams,
};
use hf_hub::{HFClient, HFClientBuilder, HFRepository};
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

/// Split a `"owner/name"` repo_id into an [`HFRepository`] handle.
fn repo_handle(client: &HFClient, owner: &str, name: &str) -> HFRepository {
    client.model(owner, name)
}

async fn create_test_repo(client: &HFClient, suffix: &str) -> (String, String) {
    let username = WHOAMI_USERNAME
        .get_or_init(|| async { client.whoami().await.expect("whoami failed").username })
        .await;
    let name = format!("hf-hub-xet-test-{suffix}");
    let repo_id = format!("{username}/{name}");
    let params = CreateRepoParams::builder()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(true)
        .build();
    client.create_repo(&params).await.expect("create_repo failed");
    (username.clone(), name)
}

async fn delete_test_repo(client: &HFClient, repo_id: &str) {
    let params = DeleteRepoParams::builder().repo_id(repo_id).build();
    let _ = client.delete_repo(&params).await;
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
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(data.clone()))
                .path_in_repo("greeting.txt")
                .commit_message("upload small text file")
                .build(),
        )
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("greeting.txt")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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

    repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(vec![]))
            .path_in_repo("empty.bin")
            .commit_message("upload empty file")
            .build(),
    )
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("empty.bin")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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

    repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(b"version 1".to_vec()))
            .path_in_repo("versioned.txt")
            .commit_message("v1")
            .build(),
    )
    .await
    .unwrap();

    repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(b"version 2 updated".to_vec()))
            .path_in_repo("versioned.txt")
            .commit_message("v2")
            .build(),
    )
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("versioned.txt")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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
    repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(data.clone()))
            .path_in_repo("a/b/c/d/deep.txt")
            .commit_message("upload nested file")
            .build(),
    )
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("a/b/c/d/deep.txt")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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

    repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::File(local_file))
            .path_in_repo("uploaded_from_path.txt")
            .commit_message("upload from file path")
            .build(),
    )
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("uploaded_from_path.txt")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("model.safetensors")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
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
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::File(local_file))
                .path_in_repo("model.safetensors")
                .commit_message("upload 75MB random data")
                .build(),
        )
        .await
        .expect("Large file upload via xet should succeed");
    assert!(commit.commit_oid.is_some());

    assert!(
        repo.file_exists(&RepoFileExistsParams::builder().filename("model.safetensors").build())
            .await
            .unwrap()
    );

    let dl_dir = tempfile::tempdir().unwrap();
    let downloaded_path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("model.safetensors")
                .local_dir(dl_dir.path().to_path_buf())
                .build(),
        )
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

    let result = repo
        .download_file_stream(&RepoDownloadFileStreamParams::builder().filename("model.safetensors").build())
        .await;

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
        .download_file_stream(
            &RepoDownloadFileStreamParams::builder()
                .filename("model.safetensors")
                .range(0..1024u64)
                .build(),
        )
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
        .download_file_stream(
            &RepoDownloadFileStreamParams::builder()
                .filename("model.safetensors")
                .range(1000..2000u64)
                .build(),
        )
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
