//! Integration tests against the live Hugging Face Hub API.
//!
//! ## Local development
//!
//! Read-only tests: require HF_TOKEN, skip if not set.
//! Write tests: require HF_TOKEN + HF_TEST_WRITE=1, skip otherwise.
//!
//! Run read-only: HF_TOKEN=hf_xxx cargo test -p hf-hub --test integration_test
//! Run all: HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p hf-hub --test integration_test
//!
//! ## CI (GITHUB_ACTIONS=true)
//!
//! Read-only tests use HF_PROD_TOKEN against https://huggingface.co.
//! Write tests use HF_CI_TOKEN against https://hub-ci.huggingface.co.
//!
//! Feature-gated tests: enable with --features, e.g.:
//!   HF_TOKEN=hf_xxx cargo test -p hf-hub --all-features --test integration_test

use futures::StreamExt;
use hf_hub::repository::*;
use hf_hub::{HFClient, HFClientBuilder, HFRepository, RepoType};
use integration_tests::test_utils::*;

fn api() -> Option<HFClient> {
    if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        Some(build_client(&token, HUB_CI_ENDPOINT))
    } else {
        default_api()
    }
}

fn prod_api() -> Option<HFClient> {
    if is_ci() {
        let token = resolve_prod_token()?;
        Some(build_client(&token, PROD_ENDPOINT))
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

const TEST_ORG: &str = "huggingface";
const TEST_USER: &str = "julien-c";
const TEST_MODEL_AUTHOR: &str = "openai-community";
const TEST_MODEL_REPO: &str = "hf-internal-testing/tiny-gemma3";
const TEST_SPACE_REPO: (&str, &str) = ("huggingface-projects", "diffusers-gallery");
const TEST_SPACE_INFO_REPO: &str = "HuggingFaceFW/blogpost-fineweb-v1";
const TEST_DATASET_REPO: &str = "hf-internal-testing/cats_vs_dogs_sample";

/// Cached whoami username, fetched once and reused across write tests.
async fn cached_username() -> &'static str {
    static USERNAME: tokio::sync::OnceCell<String> = tokio::sync::OnceCell::const_new();
    USERNAME
        .get_or_init(|| async {
            let client = api().expect("API client required for cached_username");
            client.whoami().send().await.expect("whoami failed").username
        })
        .await
}

/// Create an HFRepository handle from a full `owner/name` repo_id string.
fn repo(client: &HFClient, repo_id: &str) -> HFRepository {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    if parts.len() == 2 {
        client.model(parts[0], parts[1])
    } else {
        client.model("", repo_id)
    }
}

/// Create an HFRepository handle with a specific repo type.
fn repo_typed(client: &HFClient, repo_id: &str, repo_type: RepoType) -> HFRepository {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let (owner, name) = if parts.len() == 2 {
        (parts[0], parts[1])
    } else {
        ("", repo_id)
    };
    client.repo(repo_type, owner, name)
}

#[tokio::test]
async fn test_model_info() {
    let Some(client) = prod_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    let info = repo(&client, model_repo).info().send().await.unwrap();
    match info {
        RepoInfo::Model(model) => assert!(model.id.contains("tiny-gemma3")),
        _ => panic!("expected model info"),
    }
}

#[tokio::test]
async fn test_repo_handle_info_and_file_exists() {
    let Some(client) = prod_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    let repo = repo(&client, model_repo);

    let info = repo.info().send().await.unwrap();
    match info {
        RepoInfo::Model(model) => assert_eq!(model.id, model_repo),
        _ => panic!("expected model info"),
    }

    let exists = repo.file_exists().filename("config.json").send().await.unwrap();
    assert!(exists);
}

#[tokio::test]
async fn test_dataset_info() {
    let Some(client) = prod_api() else { return };
    let dataset_repo = TEST_DATASET_REPO;
    let info = repo_typed(&client, dataset_repo, RepoType::Dataset)
        .info()
        .send()
        .await
        .unwrap();
    match info {
        RepoInfo::Dataset(ds) => assert_eq!(ds.id, dataset_repo),
        _ => panic!("expected dataset info"),
    }
}

#[tokio::test]
async fn test_repo_exists() {
    let Some(client) = prod_api() else { return };
    assert!(repo(&client, TEST_MODEL_REPO).exists().send().await.unwrap());
    assert!(
        !repo(&client, "this-repo-definitely-does-not-exist-12345")
            .exists()
            .send()
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn test_file_exists() {
    let Some(client) = prod_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    assert!(
        repo(&client, model_repo)
            .file_exists()
            .filename("config.json")
            .send()
            .await
            .unwrap()
    );

    assert!(
        !repo(&client, model_repo)
            .file_exists()
            .filename("nonexistent_file.xyz")
            .send()
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn test_list_models() {
    let Some(client) = prod_api() else { return };
    let author = TEST_MODEL_AUTHOR;
    let stream = client.list_models().author(author).limit(3_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(model) = stream.next().await {
        let model = model.unwrap();
        assert!(model.id.starts_with(&format!("{}/", author)));
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn test_list_repo_files() {
    let Some(client) = prod_api() else { return };
    let files = repo(&client, TEST_MODEL_REPO).list_files().send().await.unwrap();
    assert!(files.contains(&"config.json".to_string()));
    assert!(files.contains(&"README.md".to_string()));
}

#[tokio::test]
async fn test_list_repo_tree() {
    let Some(client) = prod_api() else { return };
    let r = repo(&client, TEST_MODEL_REPO);
    let stream = r.list_tree().send().unwrap();
    futures::pin_mut!(stream);

    let mut found_config = false;
    while let Some(entry) = stream.next().await {
        let entry = entry.unwrap();
        if let RepoTreeEntry::File { path, .. } = &entry
            && path == "config.json"
        {
            found_config = true;
            break;
        }
    }
    assert!(found_config);
}

#[tokio::test]
async fn test_list_repo_commits() {
    let Some(client) = prod_api() else { return };
    let r = repo(&client, TEST_MODEL_REPO);
    let stream = r.list_commits().send().unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    assert!(!first.id.is_empty());
    assert!(!first.title.is_empty());
}

#[tokio::test]
async fn test_list_repo_refs() {
    let Some(client) = prod_api() else { return };
    let refs = repo(&client, TEST_MODEL_REPO).list_refs().send().await.unwrap();
    assert!(!refs.branches.is_empty());
    // "main" branch should exist
    assert!(refs.branches.iter().any(|b| b.name == "main"));
}

#[tokio::test]
async fn test_revision_exists() {
    let Some(client) = prod_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    assert!(
        repo(&client, model_repo)
            .revision_exists()
            .revision("main")
            .send()
            .await
            .unwrap()
    );

    assert!(
        !repo(&client, model_repo)
            .revision_exists()
            .revision("nonexistent-branch-xyz")
            .send()
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn test_download_file() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let path = repo(&client, TEST_MODEL_REPO)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .await
        .unwrap();
    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

// --- User operations ---

#[tokio::test]
async fn test_whoami() {
    let Some(client) = api() else { return };
    let user = client.whoami().send().await.unwrap();
    assert!(!user.username.is_empty());
}

#[tokio::test]
async fn test_auth_check() {
    let Some(client) = api() else { return };
    client.auth_check().send().await.unwrap();
}

#[tokio::test]
async fn test_get_user_overview() {
    let Some(client) = prod_api() else { return };
    let username = TEST_USER;
    let user = client.get_user_overview().username(username).send().await.unwrap();
    assert_eq!(user.username, username);
}

#[tokio::test]
async fn test_get_organization_overview() {
    let Some(client) = prod_api() else { return };
    let org_name = TEST_ORG;
    let org = client.get_organization_overview().organization(org_name).send().await.unwrap();
    assert_eq!(org.name, org_name);
}

#[tokio::test]
async fn test_list_user_followers() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_user_followers().username(TEST_USER).send().unwrap();
    futures::pin_mut!(stream);
    let first = stream.next().await;
    assert!(first.is_some());
    first.unwrap().unwrap();
}

#[tokio::test]
async fn test_list_user_following() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_user_following().username(TEST_USER).send().unwrap();
    futures::pin_mut!(stream);
    let first = stream.next().await;
    assert!(first.is_some());
    first.unwrap().unwrap();
}

#[tokio::test]
async fn test_list_organization_members() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_organization_members().organization(TEST_ORG).send().unwrap();
    futures::pin_mut!(stream);
    let first = stream.next().await;
    assert!(first.is_some());
    first.unwrap().unwrap();
}

// --- Additional repo info tests ---

#[tokio::test]
async fn test_space_info() {
    let Some(client) = prod_api() else { return };
    let space_repo = TEST_SPACE_INFO_REPO;
    let info = repo_typed(&client, space_repo, RepoType::Space).info().send().await.unwrap();
    match info {
        RepoInfo::Space(space) => assert_eq!(space.id, space_repo),
        _ => panic!("expected space info"),
    }
}

#[tokio::test]
async fn test_list_datasets() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_datasets().author(TEST_ORG).limit(3_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(ds) = stream.next().await {
        ds.unwrap();
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn test_list_spaces() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_spaces().author(TEST_ORG).limit(3_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(space) = stream.next().await {
        space.unwrap();
        count += 1;
    }
    assert!(count > 0);
}

// --- File info tests ---

#[tokio::test]
async fn test_get_paths_info() {
    let Some(client) = prod_api() else { return };
    let entries = repo(&client, TEST_MODEL_REPO)
        .get_paths_info()
        .paths(vec!["config.json".to_string(), "README.md".to_string()])
        .send()
        .await
        .unwrap();
    assert_eq!(entries.len(), 2);
    let paths: Vec<String> = entries
        .iter()
        .map(|e| match e {
            RepoTreeEntry::File { path, .. } => path.clone(),
            RepoTreeEntry::Directory { path, .. } => path.clone(),
        })
        .collect();
    assert!(paths.contains(&"config.json".to_string()));
    assert!(paths.contains(&"README.md".to_string()));
}

#[tokio::test]
async fn test_get_file_metadata() {
    let Some(client) = prod_api() else { return };
    let meta = repo(&client, TEST_MODEL_REPO)
        .get_file_metadata()
        .filepath("config.json")
        .send()
        .await
        .unwrap();
    assert_eq!(meta.filename, "config.json");
    assert!(!meta.etag.is_empty());
    assert!(!meta.commit_hash.is_empty());
    assert!(meta.file_size > 0);
}

#[tokio::test]
async fn test_get_file_metadata_with_revision() {
    let Some(client) = prod_api() else { return };
    let model = repo(&client, TEST_MODEL_REPO);
    let meta_default = model.get_file_metadata().filepath("config.json").send().await.unwrap();
    let meta_main = model
        .get_file_metadata()
        .filepath("config.json")
        .revision("main")
        .send()
        .await
        .unwrap();
    assert_eq!(meta_default.commit_hash, meta_main.commit_hash);
    assert_eq!(meta_default.etag, meta_main.etag);

    let pinned = model
        .get_file_metadata()
        .filepath("config.json")
        .revision(meta_main.commit_hash.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(pinned.commit_hash, meta_main.commit_hash);
}

#[tokio::test]
async fn test_get_file_metadata_missing() {
    let Some(client) = prod_api() else { return };
    let err = repo(&client, TEST_MODEL_REPO)
        .get_file_metadata()
        .filepath("this-file-does-not-exist.bin")
        .send()
        .await
        .unwrap_err();
    match err {
        hf_hub::HFError::EntryNotFound { path, .. } => {
            assert_eq!(path, "this-file-does-not-exist.bin");
        },
        other => panic!("expected EntryNotFound, got {other:?}"),
    }
}

// --- Commit and diff tests ---

#[tokio::test]
async fn test_get_commit_diff() {
    let Some(client) = prod_api() else { return };

    let gpt2 = repo(&client, TEST_MODEL_REPO);
    let stream = gpt2.list_commits().send().unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let diff = gpt2
        .get_commit_diff()
        .compare(format!("{}..{}", second.id, first.id))
        .send()
        .await
        .unwrap();
    assert!(!diff.is_empty());
}

#[tokio::test]
async fn test_get_raw_diff() {
    let Some(client) = prod_api() else { return };

    let gpt2 = repo(&client, TEST_MODEL_REPO);
    let stream = gpt2.list_commits().send().unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let raw = gpt2
        .get_raw_diff()
        .compare(format!("{}..{}", second.id, first.id))
        .send()
        .await
        .unwrap();
    assert!(!raw.is_empty());
}

#[tokio::test]
async fn test_diff_against_empty_tree_all_additions() {
    let Some(client) = prod_api() else { return };

    const GIT_EMPTY_TREE_HASH: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
    const ZERO_BLOB_ID: &str = "0000000000000000000000000000000000000000";

    let test_cases: Vec<(&str, &str)> = vec![
        ("espnet/yodas_owsmv4", "3bc62fd77f22f5bc2f116c6c21bcb12d5b85ab07"),
        ("ppbrown/pexels-photos-janpf", "e2629eb62efab2bf1e0fa7c9ebf2a5e75acf91f4"),
        ("tropos-labs/eigen-face-dataset-256", "341f2d0097d3a26ba4bd9e82dc8dac03b9a75f3d"),
    ];

    for (repo_id, revision) in &test_cases {
        let dataset = repo_typed(&client, repo_id, RepoType::Dataset);
        let stream = dataset
            .get_raw_diff_stream()
            .compare(format!("{GIT_EMPTY_TREE_HASH}..{revision}"))
            .send()
            .await
            .expect("get_raw_diff_stream failed");

        let diffs: Vec<_> = stream.map(|r| r.expect("stream item error")).collect().await;

        assert!(!diffs.is_empty(), "[{repo_id}] diff should not be empty");

        for diff in &diffs {
            assert_eq!(
                diff.status,
                hf_hub::repository::GitStatus::Addition,
                "[{repo_id}] file '{}' should be an Addition, got {:?}",
                diff.file_path,
                diff.status,
            );
            assert!(!diff.file_path.is_empty(), "[{repo_id}] file_path should not be empty");
            assert!(
                !diff.new_blob_id.is_empty(),
                "[{repo_id}] new_blob_id should not be empty for '{}'",
                diff.file_path,
            );
            assert_eq!(
                diff.old_blob_id, ZERO_BLOB_ID,
                "[{repo_id}] old_blob_id for '{}' should be all zeros, got '{}'",
                diff.file_path, diff.old_blob_id,
            );
        }
    }
}

// --- Write operation tests (require HF_TEST_WRITE=1) ---

#[tokio::test]
async fn test_create_and_delete_repo() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }

    let username = cached_username().await;

    let repo_id = format!("{}/hf-hub-test-{}", username, uuid_v4_short());

    // Create
    let url = client
        .create_repo()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(true)
        .send()
        .await
        .unwrap();
    assert!(url.url.contains(&repo_id));

    // Upload a file
    let test_repo = repo(&client, &repo_id);
    let commit = test_repo
        .upload_file()
        .source(AddSource::Bytes(b"hello world".to_vec()))
        .path_in_repo("test.txt")
        .commit_message("test upload")
        .send()
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    // Verify file exists
    assert!(test_repo.file_exists().filename("test.txt").send().await.unwrap());

    // Delete repo
    client.delete_repo().repo_id(&repo_id).send().await.unwrap();
}

fn uuid_v4_short() -> String {
    format!("{:016x}", rand::random::<u64>())
}

async fn create_test_repo(client: &HFClient) -> String {
    let username = cached_username().await;
    let repo_id = format!("{}/hf-hub-test-{}", username, uuid_v4_short());
    client
        .create_repo()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(false)
        .send()
        .await
        .expect("create_repo failed");

    let test_repo = repo(client, &repo_id);
    test_repo
        .upload_file()
        .source(AddSource::Bytes(b"initial content".to_vec()))
        .path_in_repo("README.md")
        .commit_message("initial commit")
        .send()
        .await
        .expect("seed upload failed");

    repo_id
}

async fn delete_test_repo(client: &HFClient, repo_id: &str) {
    let _ = client.delete_repo().repo_id(repo_id).send().await;
}

#[tokio::test]
async fn test_create_commit() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    let commit = test_repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("file_a.txt", b"content a".to_vec()),
            CommitOperation::add_bytes("file_b.txt", b"content b".to_vec()),
        ])
        .commit_message("add two files")
        .send()
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = test_repo.list_files().send().await.unwrap();
    assert!(files.contains(&"file_a.txt".to_string()));
    assert!(files.contains(&"file_b.txt".to_string()));

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_upload_folder() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hello").unwrap();
    std::fs::create_dir_all(dir.path().join("subdir")).unwrap();
    std::fs::write(dir.path().join("subdir/nested.txt"), "nested").unwrap();

    let test_repo = repo(&client, &repo_id);
    let commit = test_repo
        .upload_folder()
        .folder_path(dir.path().to_path_buf())
        .commit_message("upload folder")
        .send()
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = test_repo.list_files().send().await.unwrap();
    assert!(files.contains(&"hello.txt".to_string()));
    assert!(files.contains(&"subdir/nested.txt".to_string()));

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_delete_file() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    test_repo
        .upload_file()
        .source(AddSource::Bytes(b"to delete".to_vec()))
        .path_in_repo("deleteme.txt")
        .commit_message("add file to delete")
        .send()
        .await
        .unwrap();

    test_repo
        .delete_file()
        .path_in_repo("deleteme.txt")
        .commit_message("delete file")
        .send()
        .await
        .unwrap();

    assert!(!test_repo.file_exists().filename("deleteme.txt").send().await.unwrap());

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_delete_folder() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    test_repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("folder/a.txt", b"a".to_vec()),
            CommitOperation::add_bytes("folder/b.txt", b"b".to_vec()),
        ])
        .commit_message("add folder")
        .send()
        .await
        .unwrap();

    test_repo
        .delete_folder()
        .path_in_repo("folder")
        .commit_message("delete folder")
        .send()
        .await
        .unwrap();

    assert!(!test_repo.file_exists().filename("folder/a.txt").send().await.unwrap());

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_create_and_delete_branch() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    test_repo.create_branch().branch("test-branch").send().await.unwrap();

    let refs = test_repo.list_refs().send().await.unwrap();
    assert!(refs.branches.iter().any(|b| b.name == "test-branch"));

    test_repo.delete_branch().branch("test-branch").send().await.unwrap();

    let refs = test_repo.list_refs().send().await.unwrap();
    assert!(!refs.branches.iter().any(|b| b.name == "test-branch"));

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_create_and_delete_tag() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    test_repo.create_tag().tag("v1.0").send().await.unwrap();

    let refs = test_repo.list_refs().send().await.unwrap();
    assert!(refs.tags.iter().any(|t| t.name == "v1.0"));

    test_repo.delete_tag().tag("v1.0").send().await.unwrap();

    let refs = test_repo.list_refs().send().await.unwrap();
    assert!(!refs.tags.iter().any(|t| t.name == "v1.0"));

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_update_repo_settings() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    test_repo
        .update_settings()
        .description("test description from integration test")
        .send()
        .await
        .unwrap();

    // Verify we can still get info after update
    let _info = test_repo.info().send().await.unwrap();

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_move_repo() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let username = cached_username().await;
    let original_name = format!("{}/hf-hub-move-src-{}", username, uuid_v4_short());
    let new_name = format!("{}/hf-hub-move-dst-{}", username, uuid_v4_short());

    client.create_repo().repo_id(&original_name).private(true).send().await.unwrap();

    client
        .move_repo()
        .from_id(&original_name)
        .to_id(&new_name)
        .send()
        .await
        .unwrap();

    assert!(repo(&client, &new_name).exists().send().await.unwrap());

    client.delete_repo().repo_id(&new_name).send().await.unwrap();
}

// =============================================================================
// Spaces management tests
// =============================================================================

#[tokio::test]
async fn test_get_space_runtime() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_SPACE_REPO;
    let space = client.space(owner, name);
    let runtime = space.runtime().send().await.unwrap();
    assert!(runtime.stage.is_some());
}

#[tokio::test]
async fn test_duplicate_space() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let username = cached_username().await;

    // Create a minimal source space to duplicate.
    let source_id = format!("{}/hub-rust-test-dup-src-{}", username, uuid_v4_short());
    client
        .create_repo()
        .repo_id(&source_id)
        .repo_type(RepoType::Space)
        .private(true)
        .space_sdk("static")
        .send()
        .await
        .unwrap();

    let to_id = format!("{}/hub-rust-test-dup-space-{}", username, uuid_v4_short());
    let (owner, name) = source_id.split_once('/').unwrap();
    let source = client.space(owner, name);
    let result = source
        .duplicate()
        .to_id(&to_id)
        .private(true)
        .hardware("cpu-basic")
        .send()
        .await
        .unwrap();
    assert!(result.url.contains(&to_id));

    // Clean up both spaces.
    let _ = client.delete_repo().repo_id(&to_id).repo_type(RepoType::Space).send().await;
    let _ = client.delete_repo().repo_id(&source_id).repo_type(RepoType::Space).send().await;
}

#[tokio::test]
async fn test_space_secrets_and_variables() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let username = cached_username().await;
    let space = client.space(username, format!("hub-rust-test-space-{}", uuid_v4_short()));
    client
        .create_repo()
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
        .private(true)
        .space_sdk("static")
        .send()
        .await
        .unwrap();

    space
        .add_secret()
        .key("TEST_SECRET")
        .value("secret_value")
        .send()
        .await
        .unwrap();

    space.delete_secret().key("TEST_SECRET").send().await.unwrap();

    space.add_variable().key("TEST_VAR").value("var_value").send().await.unwrap();

    space.delete_variable().key("TEST_VAR").send().await.unwrap();

    let _ = client
        .delete_repo()
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
        .send()
        .await;
}
