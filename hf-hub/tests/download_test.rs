//! Integration tests for downloads and progress tracking.
//!
//! Read-only tests (downloads from hardcoded repos) use **prod** (huggingface.co).
//! Write tests (upload progress) create temporary repos on **hub-ci** and require HF_TEST_WRITE=1.
//!
//! Run read-only: HF_TOKEN=hf_xxx cargo test -p hf-hub --test download_test
//! Run all: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p hf-hub --test download_test
//!
//! CI: read-only tests use HF_PROD_TOKEN, write tests use HF_CI_TOKEN against hub-ci.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use hf_hub::repository::HFRepository;
use hf_hub::test_utils::*;
use hf_hub::{
    AddSource, CommitOperation, CreateRepoParams, DeleteRepoParams, DownloadEvent, FileStatus, HFClient,
    HFClientBuilder, ProgressEvent, ProgressHandler, RepoCreateCommitParams, RepoDownloadFileParams,
    RepoDownloadFileStreamParams, RepoSnapshotDownloadParams, RepoUploadFileParams, UploadEvent, UploadPhase,
};
use sha2::{Digest, Sha256};

fn prod_api() -> Option<HFClient> {
    if is_ci() {
        let token = resolve_prod_token()?;
        Some(build_client(&token, PROD_ENDPOINT))
    } else {
        default_api()
    }
}

fn hub_ci_api() -> Option<HFClient> {
    if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        Some(build_client(&token, HUB_CI_ENDPOINT))
    } else {
        default_api()
    }
}

fn default_api() -> Option<HFClient> {
    let token = std::env::var(HF_TOKEN).ok()?;
    let endpoint = std::env::var(HF_ENDPOINT).unwrap_or_else(|_| PROD_ENDPOINT.to_string());
    Some(build_client(&token, &endpoint))
}

fn build_client(token: &str, endpoint: &str) -> HFClient {
    HFClientBuilder::new()
        .token(token)
        .endpoint(endpoint)
        .build()
        .expect("Failed to create HFClient")
}

fn uuid_short() -> String {
    format!("{:016x}", rand::random::<u64>())
}

async fn cached_username(api: &HFClient) -> String {
    api.whoami().await.expect("whoami failed").username
}

async fn create_test_repo(api: &HFClient) -> String {
    let username = cached_username(api).await;
    let repo_id = format!("{}/hfrs-progress-test-{}", username, uuid_short());
    let params = CreateRepoParams::builder()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(false)
        .build();
    api.create_repo(&params).await.expect("create_repo failed");
    repo_id
}

async fn delete_test_repo(api: &HFClient, repo_id: &str) {
    let params = DeleteRepoParams::builder().repo_id(repo_id).build();
    let _ = api.delete_repo(&params).await;
}

const TEST_MODEL_PARTS: (&str, &str) = ("hf-internal-testing", "tiny-gemma3");
const TEST_DATASET_PARTS: (&str, &str) = ("hf-internal-testing", "cats_vs_dogs_sample");

fn model(api: &HFClient, owner: &str, name: &str) -> HFRepository {
    api.model(owner, name)
}

fn dataset(api: &HFClient, owner: &str, name: &str) -> HFRepository {
    api.dataset(owner, name)
}

#[tokio::test]
async fn test_download_small_json_file() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_preserves_subdirectory_structure() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();

    assert_eq!(path, dir.path().join("config.json"));
    assert!(path.exists());
}

#[tokio::test]
async fn test_download_with_specific_revision() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let path = model(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .revision("main")
                .build(),
        )
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_dataset_file() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_DATASET_PARTS;

    let path = dataset(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("README.md")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();

    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_download_nonexistent_file_returns_error() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let result = model(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("this_file_does_not_exist_at_all.bin")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_download_from_nonexistent_repo_returns_error() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();

    let result = model(&api, "this-user-does-not-exist-99999", "this-repo-does-not-exist")
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("anything.txt")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_download_multiple_files_to_same_dir() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    for filename in &["config.json", "README.md"] {
        let path = repo
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(*filename)
                    .local_dir(dir.path().to_path_buf())
                    .build(),
            )
            .await
            .unwrap();
        assert!(path.exists());
    }

    assert!(dir.path().join("config.json").exists());
    assert!(dir.path().join("README.md").exists());
}

#[tokio::test]
async fn test_download_file_content_is_deterministic() {
    let Some(api) = prod_api() else { return };
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    for dir in [&dir1, &dir2] {
        repo.download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();
    }

    let content1 = std::fs::read(dir1.path().join("config.json")).unwrap();
    let content2 = std::fs::read(dir2.path().join("config.json")).unwrap();

    let hash1 = Sha256::digest(&content1);
    let hash2 = Sha256::digest(&content2);
    assert_eq!(hash1, hash2);
}

#[tokio::test]
async fn test_download_overwrites_existing_file() {
    let Some(api) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let (owner, name) = TEST_MODEL_PARTS;

    let dest = dir.path().join("config.json");
    std::fs::write(&dest, "old content").unwrap();

    model(&api, owner, name)
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();

    let content = std::fs::read_to_string(&dest).unwrap();
    assert_ne!(content, "old content");
    assert!(content.contains("model_type"));
}

// --- Range / partial download tests (non-xet) ---

#[tokio::test]
async fn test_download_stream_full_file() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    let (content_length, stream) = repo
        .download_file_stream(&RepoDownloadFileStreamParams::builder().filename("config.json").build())
        .await
        .unwrap();

    assert!(content_length.is_some());

    futures::pin_mut!(stream);
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }

    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json.get("model_type").is_some());
}

#[tokio::test]
async fn test_download_stream_range_first_bytes() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    // Download just the first 20 bytes
    let (content_length, stream) = repo
        .download_file_stream(
            &RepoDownloadFileStreamParams::builder()
                .filename("config.json")
                .range(0..20u64)
                .build(),
        )
        .await
        .unwrap();

    assert!(content_length.unwrap() <= 20);

    futures::pin_mut!(stream);
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(bytes.len(), 20);
}

#[tokio::test]
async fn test_download_stream_range_middle_bytes() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    // First download the full file for comparison
    let (_len, full_stream) = repo
        .download_file_stream(&RepoDownloadFileStreamParams::builder().filename("config.json").build())
        .await
        .unwrap();
    futures::pin_mut!(full_stream);
    let mut full_bytes = Vec::new();
    while let Some(chunk) = full_stream.next().await {
        full_bytes.extend_from_slice(&chunk.unwrap());
    }

    // Now download a range from the middle
    let start = 10u64;
    let end = 50u64;
    let (_len, range_stream) = repo
        .download_file_stream(
            &RepoDownloadFileStreamParams::builder()
                .filename("config.json")
                .range(start..end)
                .build(),
        )
        .await
        .unwrap();

    futures::pin_mut!(range_stream);
    let mut range_bytes = Vec::new();
    while let Some(chunk) = range_stream.next().await {
        range_bytes.extend_from_slice(&chunk.unwrap());
    }

    assert_eq!(range_bytes.len(), (end - start) as usize);
    assert_eq!(range_bytes, &full_bytes[start as usize..end as usize]);
}

#[tokio::test]
async fn test_download_stream_range_content_matches_full_download() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);
    let dir = tempfile::tempdir().unwrap();

    // Download full file to disk for reference
    let path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(dir.path().to_path_buf())
                .build(),
        )
        .await
        .unwrap();
    let full_bytes = std::fs::read(&path).unwrap();

    // Stream the first 100 bytes
    let range_end = 100u64.min(full_bytes.len() as u64);
    let (_len, stream) = repo
        .download_file_stream(
            &RepoDownloadFileStreamParams::builder()
                .filename("config.json")
                .range(0..range_end)
                .build(),
        )
        .await
        .unwrap();

    futures::pin_mut!(stream);
    let mut streamed = Vec::new();
    while let Some(chunk) = stream.next().await {
        streamed.extend_from_slice(&chunk.unwrap());
    }

    assert_eq!(streamed, &full_bytes[..range_end as usize]);
}

// --- Progress tracking tests ---

struct RecordingHandler {
    events: Mutex<Vec<ProgressEvent>>,
}

impl RecordingHandler {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<ProgressEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl ProgressHandler for RecordingHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

#[tokio::test]
async fn test_download_file_with_progress_to_local_dir() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let dir = tempfile::tempdir().unwrap();
    let params = RepoDownloadFileParams::builder()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .progress(Some(handler.clone()))
        .build();

    let path = repo.download_file(&params).await.unwrap();
    assert!(path.exists());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");

    // First event should be Download(Start)
    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start), got {:?}",
        &events[0]
    );

    // Last event should be Download(Complete)
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );

    // Should have at least one Progress event with InProgress or Complete
    let has_progress = events
        .iter()
        .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Progress { .. })));
    assert!(has_progress, "should have at least one Progress event");

    // Should have a Complete file status
    let has_file_complete = events.iter().any(|e| {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = e {
            files.iter().any(|f| f.status == FileStatus::Complete)
        } else {
            false
        }
    });
    assert!(has_file_complete, "should have a file Complete status event");
}

#[tokio::test]
async fn test_download_file_with_progress_to_cache() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    let handler = Arc::new(RecordingHandler::new());

    let params = RepoDownloadFileParams::builder()
        .filename("config.json")
        .force_download(true)
        .progress(Some(handler.clone()))
        .build();

    let path = repo.download_file(&params).await.unwrap();
    assert!(path.exists());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received progress events");

    assert!(
        matches!(&events[0], ProgressEvent::Download(DownloadEvent::Start { total_files: 1, .. })),
        "first event should be Download(Start)"
    );
    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Download(DownloadEvent::Complete)),
        "last event should be Download(Complete)"
    );
}

#[tokio::test]
async fn test_download_with_no_progress_handler() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    let dir = tempfile::tempdir().unwrap();
    let params = RepoDownloadFileParams::builder()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .build();

    let path = repo.download_file(&params).await.unwrap();
    assert!(path.exists());
}

// --- Upload progress tests (write to hub-ci) ---

fn repo_from_id(api: &HFClient, repo_id: &str) -> HFRepository {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    api.model(parts[0], parts[1])
}

#[tokio::test]
async fn test_upload_file_with_progress() {
    let Some(api) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&api).await;
    let repo = repo_from_id(&api, &repo_id);

    let handler = Arc::new(RecordingHandler::new());

    let result = repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"hello from progress test".to_vec()))
                .path_in_repo("progress_test.txt")
                .commit_message("upload with progress tracking")
                .progress(Some(handler.clone()))
                .build(),
        )
        .await;

    delete_test_repo(&api, &repo_id).await;
    let commit = result.unwrap();
    assert!(commit.commit_oid.is_some());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received upload progress events");

    assert!(
        matches!(&events[0], ProgressEvent::Upload(UploadEvent::Start { total_files: 1, .. })),
        "first event should be Upload(Start), got {:?}",
        &events[0]
    );

    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Upload(UploadEvent::Complete)),
        "last event should be Upload(Complete)"
    );

    let has_preparing = events.iter().any(|e| {
        matches!(
            e,
            ProgressEvent::Upload(UploadEvent::Progress {
                phase: UploadPhase::Preparing,
                ..
            })
        )
    });
    assert!(has_preparing, "should have a Preparing phase event");

    let has_committing = events.iter().any(|e| {
        matches!(
            e,
            ProgressEvent::Upload(UploadEvent::Progress {
                phase: UploadPhase::Committing,
                ..
            })
        )
    });
    assert!(has_committing, "should have a Committing phase event");
}

#[tokio::test]
async fn test_create_commit_with_progress_multiple_files() {
    let Some(api) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&api).await;
    let repo = repo_from_id(&api, &repo_id);

    let handler = Arc::new(RecordingHandler::new());

    let result = repo
        .create_commit(
            &RepoCreateCommitParams::builder()
                .operations(vec![
                    CommitOperation::Add {
                        path_in_repo: "file_a.txt".to_string(),
                        source: AddSource::Bytes(b"content a".to_vec()),
                    },
                    CommitOperation::Add {
                        path_in_repo: "file_b.txt".to_string(),
                        source: AddSource::Bytes(b"content b".to_vec()),
                    },
                ])
                .commit_message("multi-file commit with progress")
                .progress(Some(handler.clone()))
                .build(),
        )
        .await;

    delete_test_repo(&api, &repo_id).await;
    let commit = result.unwrap();
    assert!(commit.commit_oid.is_some());

    let events = handler.events();
    assert!(!events.is_empty(), "should have received upload progress events");

    if let ProgressEvent::Upload(UploadEvent::Start {
        total_files,
        total_bytes,
    }) = &events[0]
    {
        assert_eq!(*total_files, 2);
        assert_eq!(*total_bytes, 18); // "content a" + "content b" = 9 + 9
    } else {
        panic!("first event should be Upload(Start), got {:?}", &events[0]);
    }

    assert!(
        matches!(events.last().unwrap(), ProgressEvent::Upload(UploadEvent::Complete)),
        "last event should be Upload(Complete)"
    );
}

#[tokio::test]
async fn test_upload_with_no_progress_handler() {
    let Some(api) = hub_ci_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&api).await;
    let repo = repo_from_id(&api, &repo_id);

    let result = repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"no handler test".to_vec()))
                .path_in_repo("no_handler.txt")
                .commit_message("upload without progress handler")
                .build(),
        )
        .await;

    delete_test_repo(&api, &repo_id).await;
    result.unwrap();
}

#[tokio::test]
async fn test_snapshot_download_exactly_one_complete_per_file() {
    let Some(api) = prod_api() else { return };
    let (owner, name) = TEST_MODEL_PARTS;
    let repo = model(&api, owner, name);

    let handler = Arc::new(RecordingHandler::new());
    let dir = tempfile::tempdir().unwrap();

    let params = RepoSnapshotDownloadParams::builder()
        .local_dir(dir.path().to_path_buf())
        .allow_patterns(vec!["*.json".to_string()])
        .force_download(true)
        .progress(Some(handler.clone()))
        .build();

    repo.snapshot_download(&params).await.unwrap();

    let events = handler.events();

    // Count Complete events per filename
    let mut complete_counts: HashMap<String, usize> = HashMap::new();
    for event in &events {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event {
            for fp in files {
                if fp.status == FileStatus::Complete {
                    *complete_counts.entry(fp.filename.clone()).or_default() += 1;
                }
            }
        }
    }

    assert!(!complete_counts.is_empty(), "should have at least one file Complete event");

    for (filename, count) in &complete_counts {
        assert_eq!(*count, 1, "file '{filename}' had {count} Complete events, expected exactly 1");
    }

    // The files_bar count should match: total_files from Start == number of distinct Complete files
    if let Some(ProgressEvent::Download(DownloadEvent::Start { total_files, .. })) = events.first() {
        assert_eq!(
            *total_files,
            complete_counts.len(),
            "total_files in Start ({total_files}) should match number of completed files ({})",
            complete_counts.len()
        );
    }
}
