//! Integration tests for `HFRepository::upload_large_folder`.
//!
//! Tests upload, resume (no second commit), dedup savings, and cross-tool
//! resume from a Python-written cache file.
//!
//! Requires:
//!   - HF_TOKEN (local) or HF_CI_TOKEN (CI) environment variable
//!   - HF_TEST_WRITE=1 (creates and deletes repos)
//!
//! Run: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p integration-tests upload_large_folder -- --nocapture

use hf_hub::{HFClient, RepoTypeDataset};
use integration_tests::test_utils::{
    HF_CI_TOKEN, HF_TOKEN, HUB_CI_ENDPOINT, is_ci, resolve_hub_ci_token, write_enabled,
};

fn unique_repo_name(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{prefix}-{nanos}")
}

fn make_client() -> Option<HFClient> {
    if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        Some(
            HFClient::builder()
                .token(token)
                .endpoint(HUB_CI_ENDPOINT)
                .build()
                .expect("failed to create HFClient"),
        )
    } else {
        let token = std::env::var(HF_TOKEN).ok()?;
        Some(HFClient::builder().token(token).build().expect("failed to create HFClient"))
    }
}

/// Upload a folder, verify report fields, then re-upload and verify no new commits are created.
#[tokio::test]
async fn upload_large_folder_uploads_and_resumes() {
    let Some(_token) = resolve_hub_ci_token() else {
        eprintln!("skipping: no token");
        return;
    };
    if !write_enabled() {
        eprintln!("skipping: HF_TEST_WRITE not set");
        return;
    }
    let Some(client) = make_client() else {
        eprintln!("skipping: no token");
        return;
    };

    let owner = client.whoami().send().await.unwrap().username;
    let name = unique_repo_name("rs-ulf-test");
    let repo = client.dataset(&owner, &name);

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("readme.md"), b"hello world").unwrap();
    std::fs::create_dir_all(dir.path().join("data")).unwrap();
    let big: Vec<u8> = (0..(8 * 1024 * 1024)).map(|i| (i % 251) as u8).collect();
    std::fs::write(dir.path().join("data/a.bin"), &big).unwrap();
    let mut dup = big.clone();
    dup.extend_from_slice(b"tail");
    std::fs::write(dir.path().join("data/b.bin"), &dup).unwrap();

    let report = repo
        .upload_large_folder()
        .folder_path(dir.path().to_path_buf())
        .send()
        .await
        .expect("upload_large_folder failed");

    assert_eq!(report.total_files, 3);
    assert!(report.files_uploaded_lfs >= 2, "expected the two big files via lfs");
    assert!(!report.commits.is_empty());
    assert!(report.dedup_bytes_saved > 0, "expected dedup savings from b.bin duplicating a.bin");

    let meta = std::fs::read_to_string(dir.path().join(".cache/huggingface/upload/data/a.bin.metadata")).unwrap();
    let committed_line = meta.split('\n').nth(7).unwrap();
    assert_eq!(committed_line, "1", "a.bin should be marked committed");

    // Resume: no new commits expected since everything is already committed.
    let report2 = repo
        .upload_large_folder()
        .folder_path(dir.path().to_path_buf())
        .send()
        .await
        .expect("resume upload_large_folder failed");
    assert_eq!(report2.total_files, 3);
    assert!(report2.commits.is_empty(), "resume should create no new commits");

    let _ = client
        .delete_repository()
        .repo_id(format!("{owner}/{name}"))
        .repo_type(RepoTypeDataset)
        .send()
        .await;
}

/// Pre-seed a Python-style metadata file marking a file as already committed;
/// the Rust implementation should respect it and produce no new commits.
#[tokio::test]
async fn resumes_python_written_cache() {
    let Some(_token) = resolve_hub_ci_token() else {
        return;
    };
    if !write_enabled() {
        return;
    }
    let Some(client) = make_client() else {
        return;
    };

    let owner = client.whoami().send().await.unwrap().username;
    let name = unique_repo_name("rs-ulf-resume");
    let repo = client.dataset(&owner, &name);

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("x.txt"), b"abc").unwrap();

    // Pre-seed a Python-style metadata file marking x.txt as already committed.
    let upload_dir = dir.path().join(".cache/huggingface/upload");
    std::fs::create_dir_all(&upload_dir).unwrap();
    // 8-line format: timestamp, size, should_ignore, sha256, upload_mode, remote_oid, is_uploaded, is_committed
    std::fs::write(upload_dir.join("x.txt.metadata"), "99999999999.0\n3\n0\n\nregular\n\n0\n1\n").unwrap();

    let report = repo
        .upload_large_folder()
        .folder_path(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert_eq!(report.total_files, 1);
    assert!(report.commits.is_empty(), "expected no commit for an already-committed file");

    let _ = client
        .delete_repository()
        .repo_id(format!("{owner}/{name}"))
        .repo_type(RepoTypeDataset)
        .send()
        .await;
}

/// Many small (regular) files plus explicit num_workers. Exercises parallel
/// classify, the committer's batching, and produces ≥1 commit with all files.
#[tokio::test]
async fn upload_large_folder_many_small_files_with_workers() {
    let Some(_token) = resolve_hub_ci_token() else {
        eprintln!("skipping: no token");
        return;
    };
    if !write_enabled() {
        eprintln!("skipping: HF_TEST_WRITE not set");
        return;
    }
    let Some(client) = make_client() else {
        eprintln!("skipping: no token");
        return;
    };

    let owner = client.whoami().send().await.unwrap().username;
    let name = unique_repo_name("rs-ulf-small");
    let repo = client.dataset(&owner, &name);

    let dir = tempfile::tempdir().unwrap();
    for i in 0..25 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), format!("content {i}")).unwrap();
    }

    let report = repo
        .upload_large_folder()
        .folder_path(dir.path().to_path_buf())
        .num_workers(3)
        .send()
        .await
        .expect("upload_large_folder failed");

    assert_eq!(report.total_files, 25);
    assert_eq!(report.files_committed_inline, 25);
    assert_eq!(report.files_uploaded_lfs, 0);
    assert!(!report.commits.is_empty());

    // Resume: nothing new to commit.
    let report2 = repo
        .upload_large_folder()
        .folder_path(dir.path().to_path_buf())
        .num_workers(1)
        .send()
        .await
        .expect("resume failed");
    assert!(report2.commits.is_empty(), "resume should create no new commits");

    let _ = client
        .delete_repository()
        .repo_id(format!("{owner}/{name}"))
        .repo_type(RepoTypeDataset)
        .send()
        .await;
}
