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
use hf_hub::test_utils::*;
use hf_hub::types::*;
use hf_hub::{HFClient, HFClientBuilder, HFRepository};

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
            client.whoami().await.expect("whoami failed").username
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
    let info = repo(&client, model_repo).info(&RepoInfoParams::default()).await.unwrap();
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

    let info = repo.info(&RepoInfoParams::default()).await.unwrap();
    match info {
        RepoInfo::Model(model) => assert_eq!(model.id, model_repo),
        _ => panic!("expected model info"),
    }

    let exists = repo
        .file_exists(&RepoFileExistsParams::builder().filename("config.json").build())
        .await
        .unwrap();
    assert!(exists);
}

#[tokio::test]
async fn test_dataset_info() {
    let Some(client) = prod_api() else { return };
    let dataset_repo = TEST_DATASET_REPO;
    let info = repo_typed(&client, dataset_repo, RepoType::Dataset)
        .info(&RepoInfoParams::default())
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
    assert!(repo(&client, TEST_MODEL_REPO).exists().await.unwrap());
    assert!(
        !repo(&client, "this-repo-definitely-does-not-exist-12345")
            .exists()
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
            .file_exists(&RepoFileExistsParams::builder().filename("config.json").build())
            .await
            .unwrap()
    );

    assert!(
        !repo(&client, model_repo)
            .file_exists(&RepoFileExistsParams::builder().filename("nonexistent_file.xyz").build())
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn test_list_models() {
    let Some(client) = prod_api() else { return };
    let author = TEST_MODEL_AUTHOR;
    let params = ListModelsParams::builder().author(author).limit(3_usize).build();
    let stream = client.list_models(&params).unwrap();
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
    let files = repo(&client, TEST_MODEL_REPO)
        .list_files(&RepoListFilesParams::default())
        .await
        .unwrap();
    assert!(files.contains(&"config.json".to_string()));
    assert!(files.contains(&"README.md".to_string()));
}

#[tokio::test]
async fn test_list_repo_tree() {
    let Some(client) = prod_api() else { return };
    let r = repo(&client, TEST_MODEL_REPO);
    let stream = r.list_tree(&RepoListTreeParams::default()).unwrap();
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
    let stream = r.list_commits(&RepoListCommitsParams::default()).unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    assert!(!first.id.is_empty());
    assert!(!first.title.is_empty());
}

#[tokio::test]
async fn test_list_repo_refs() {
    let Some(client) = prod_api() else { return };
    let refs = repo(&client, TEST_MODEL_REPO)
        .list_refs(&RepoListRefsParams::default())
        .await
        .unwrap();
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
            .revision_exists(&RepoRevisionExistsParams::builder().revision("main").build())
            .await
            .unwrap()
    );

    assert!(
        !repo(&client, model_repo)
            .revision_exists(&RepoRevisionExistsParams::builder().revision("nonexistent-branch-xyz").build())
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn test_download_file() {
    let Some(client) = prod_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let path = repo(&client, TEST_MODEL_REPO)
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

// --- User operations ---

#[tokio::test]
async fn test_whoami() {
    let Some(client) = api() else { return };
    let user = client.whoami().await.unwrap();
    assert!(!user.username.is_empty());
}

#[tokio::test]
async fn test_auth_check() {
    let Some(client) = api() else { return };
    client.auth_check().await.unwrap();
}

#[tokio::test]
async fn test_get_user_overview() {
    let Some(client) = prod_api() else { return };
    let username = TEST_USER;
    let user = client.get_user_overview(username).await.unwrap();
    assert_eq!(user.username, username);
}

#[tokio::test]
async fn test_get_organization_overview() {
    let Some(client) = prod_api() else { return };
    let org_name = TEST_ORG;
    let org = client.get_organization_overview(org_name).await.unwrap();
    assert_eq!(org.name, org_name);
}

#[tokio::test]
async fn test_list_user_followers() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_user_followers(TEST_USER, None).unwrap();
    futures::pin_mut!(stream);
    let first = stream.next().await;
    assert!(first.is_some());
    first.unwrap().unwrap();
}

#[tokio::test]
async fn test_list_user_following() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_user_following(TEST_USER, None).unwrap();
    futures::pin_mut!(stream);
    let first = stream.next().await;
    assert!(first.is_some());
    first.unwrap().unwrap();
}

#[tokio::test]
async fn test_list_organization_members() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_organization_members(TEST_ORG, None).unwrap();
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
    let info = repo_typed(&client, space_repo, RepoType::Space)
        .info(&RepoInfoParams::default())
        .await
        .unwrap();
    match info {
        RepoInfo::Space(space) => assert_eq!(space.id, space_repo),
        _ => panic!("expected space info"),
    }
}

#[tokio::test]
async fn test_list_datasets() {
    let Some(client) = prod_api() else { return };
    let params = ListDatasetsParams::builder().author(TEST_ORG).limit(3_usize).build();
    let stream = client.list_datasets(&params).unwrap();
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
    let params = ListSpacesParams::builder().author(TEST_ORG).limit(3_usize).build();
    let stream = client.list_spaces(&params).unwrap();
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
        .get_paths_info(
            &RepoGetPathsInfoParams::builder()
                .paths(vec!["config.json".to_string(), "README.md".to_string()])
                .build(),
        )
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
        .get_file_metadata(&RepoGetFileMetadataParams::builder().filepath("config.json").build())
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
    let meta_default = model
        .get_file_metadata(&RepoGetFileMetadataParams::builder().filepath("config.json").build())
        .await
        .unwrap();
    let meta_main = model
        .get_file_metadata(
            &RepoGetFileMetadataParams::builder()
                .filepath("config.json")
                .revision("main")
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(meta_default.commit_hash, meta_main.commit_hash);
    assert_eq!(meta_default.etag, meta_main.etag);

    let pinned = model
        .get_file_metadata(
            &RepoGetFileMetadataParams::builder()
                .filepath("config.json")
                .revision(meta_main.commit_hash.clone())
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(pinned.commit_hash, meta_main.commit_hash);
}

#[tokio::test]
async fn test_get_file_metadata_missing() {
    let Some(client) = prod_api() else { return };
    let err = repo(&client, TEST_MODEL_REPO)
        .get_file_metadata(
            &RepoGetFileMetadataParams::builder()
                .filepath("this-file-does-not-exist.bin")
                .build(),
        )
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
    let stream = gpt2.list_commits(&RepoListCommitsParams::default()).unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let diff = gpt2
        .get_commit_diff(
            &RepoGetCommitDiffParams::builder()
                .compare(format!("{}..{}", second.id, first.id))
                .build(),
        )
        .await
        .unwrap();
    assert!(!diff.is_empty());
}

#[tokio::test]
async fn test_get_raw_diff() {
    let Some(client) = prod_api() else { return };

    let gpt2 = repo(&client, TEST_MODEL_REPO);
    let stream = gpt2.list_commits(&RepoListCommitsParams::default()).unwrap();
    futures::pin_mut!(stream);

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let raw = gpt2
        .get_raw_diff(
            &RepoGetRawDiffParams::builder()
                .compare(format!("{}..{}", second.id, first.id))
                .build(),
        )
        .await
        .unwrap();
    assert!(!raw.is_empty());
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
    let params = CreateRepoParams::builder()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(true)
        .build();
    let url = client.create_repo(&params).await.unwrap();
    assert!(url.url.contains(&repo_id));

    // Upload a file
    let test_repo = repo(&client, &repo_id);
    let commit = test_repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"hello world".to_vec()))
                .path_in_repo("test.txt")
                .commit_message("test upload")
                .build(),
        )
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    // Verify file exists
    assert!(
        test_repo
            .file_exists(&RepoFileExistsParams::builder().filename("test.txt").build())
            .await
            .unwrap()
    );

    // Delete repo
    let params = DeleteRepoParams::builder().repo_id(&repo_id).build();
    client.delete_repo(&params).await.unwrap();
}

fn uuid_v4_short() -> String {
    format!("{:016x}", rand::random::<u64>())
}

async fn create_test_repo(client: &HFClient) -> String {
    let username = cached_username().await;
    let repo_id = format!("{}/hf-hub-test-{}", username, uuid_v4_short());
    let params = CreateRepoParams::builder()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(false)
        .build();
    client.create_repo(&params).await.expect("create_repo failed");

    let test_repo = repo(client, &repo_id);
    test_repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"initial content".to_vec()))
                .path_in_repo("README.md")
                .commit_message("initial commit")
                .build(),
        )
        .await
        .expect("seed upload failed");

    repo_id
}

async fn delete_test_repo(client: &HFClient, repo_id: &str) {
    let params = DeleteRepoParams::builder().repo_id(repo_id).build();
    let _ = client.delete_repo(&params).await;
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
                .commit_message("add two files")
                .build(),
        )
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = test_repo.list_files(&RepoListFilesParams::default()).await.unwrap();
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
        .upload_folder(
            &RepoUploadFolderParams::builder()
                .folder_path(dir.path().to_path_buf())
                .commit_message("upload folder")
                .build(),
        )
        .await
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = test_repo.list_files(&RepoListFilesParams::default()).await.unwrap();
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
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"to delete".to_vec()))
                .path_in_repo("deleteme.txt")
                .commit_message("add file to delete")
                .build(),
        )
        .await
        .unwrap();

    test_repo
        .delete_file(
            &RepoDeleteFileParams::builder()
                .path_in_repo("deleteme.txt")
                .commit_message("delete file")
                .build(),
        )
        .await
        .unwrap();

    assert!(
        !test_repo
            .file_exists(&RepoFileExistsParams::builder().filename("deleteme.txt").build())
            .await
            .unwrap()
    );

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
        .create_commit(
            &RepoCreateCommitParams::builder()
                .operations(vec![
                    CommitOperation::Add {
                        path_in_repo: "folder/a.txt".to_string(),
                        source: AddSource::Bytes(b"a".to_vec()),
                    },
                    CommitOperation::Add {
                        path_in_repo: "folder/b.txt".to_string(),
                        source: AddSource::Bytes(b"b".to_vec()),
                    },
                ])
                .commit_message("add folder")
                .build(),
        )
        .await
        .unwrap();

    test_repo
        .delete_folder(
            &RepoDeleteFolderParams::builder()
                .path_in_repo("folder")
                .commit_message("delete folder")
                .build(),
        )
        .await
        .unwrap();

    assert!(
        !test_repo
            .file_exists(&RepoFileExistsParams::builder().filename("folder/a.txt").build())
            .await
            .unwrap()
    );

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
    test_repo
        .create_branch(&RepoCreateBranchParams::builder().branch("test-branch").build())
        .await
        .unwrap();

    let refs = test_repo.list_refs(&RepoListRefsParams::default()).await.unwrap();
    assert!(refs.branches.iter().any(|b| b.name == "test-branch"));

    test_repo
        .delete_branch(&RepoDeleteBranchParams::builder().branch("test-branch").build())
        .await
        .unwrap();

    let refs = test_repo.list_refs(&RepoListRefsParams::default()).await.unwrap();
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
    test_repo
        .create_tag(&RepoCreateTagParams::builder().tag("v1.0").build())
        .await
        .unwrap();

    let refs = test_repo.list_refs(&RepoListRefsParams::default()).await.unwrap();
    assert!(refs.tags.iter().any(|t| t.name == "v1.0"));

    test_repo
        .delete_tag(&RepoDeleteTagParams::builder().tag("v1.0").build())
        .await
        .unwrap();

    let refs = test_repo.list_refs(&RepoListRefsParams::default()).await.unwrap();
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
        .update_settings(
            &RepoUpdateSettingsParams::builder()
                .description("test description from integration test")
                .build(),
        )
        .await
        .unwrap();

    // Verify we can still get info after update
    let _info = test_repo.info(&RepoInfoParams::default()).await.unwrap();

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

    let create_params = CreateRepoParams::builder().repo_id(&original_name).private(true).build();
    client.create_repo(&create_params).await.unwrap();

    let move_params = MoveRepoParams::builder().from_id(&original_name).to_id(&new_name).build();
    client.move_repo(&move_params).await.unwrap();

    assert!(repo(&client, &new_name).exists().await.unwrap());

    let delete_params = DeleteRepoParams::builder().repo_id(&new_name).build();
    client.delete_repo(&delete_params).await.unwrap();
}

// =============================================================================
// Spaces management tests
// =============================================================================

#[tokio::test]
async fn test_get_space_runtime() {
    let Some(client) = prod_api() else { return };
    let (owner, name) = TEST_SPACE_REPO;
    let space = client.space(owner, name);
    let runtime = space.runtime().await.unwrap();
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
    let create_params = CreateRepoParams::builder()
        .repo_id(&source_id)
        .repo_type(RepoType::Space)
        .private(true)
        .space_sdk("static")
        .build();
    client.create_repo(&create_params).await.unwrap();

    let to_id = format!("{}/hub-rust-test-dup-space-{}", username, uuid_v4_short());
    let params = DuplicateSpaceParams::builder()
        .to_id(&to_id)
        .private(true)
        .hardware("cpu-basic")
        .build();
    let (owner, name) = source_id.split_once('/').unwrap();
    let source = client.space(owner, name);
    let result = source.duplicate(&params).await.unwrap();
    assert!(result.url.contains(&to_id));

    // Clean up both spaces.
    let delete_dup = DeleteRepoParams::builder().repo_id(&to_id).repo_type(RepoType::Space).build();
    let delete_src = DeleteRepoParams::builder()
        .repo_id(&source_id)
        .repo_type(RepoType::Space)
        .build();
    let _ = client.delete_repo(&delete_dup).await;
    let _ = client.delete_repo(&delete_src).await;
}

#[tokio::test]
async fn test_space_secrets_and_variables() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let username = cached_username().await;
    let space = client.space(username, format!("hub-rust-test-space-{}", uuid_v4_short()));
    let create_params = CreateRepoParams::builder()
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
        .private(true)
        .space_sdk("static")
        .build();
    client.create_repo(&create_params).await.unwrap();

    let add_secret = SpaceSecretParams::builder().key("TEST_SECRET").value("secret_value").build();
    space.add_secret(&add_secret).await.unwrap();

    let del_secret = SpaceSecretDeleteParams::builder().key("TEST_SECRET").build();
    space.delete_secret(&del_secret).await.unwrap();

    let add_var = SpaceVariableParams::builder().key("TEST_VAR").value("var_value").build();
    space.add_variable(&add_var).await.unwrap();

    let del_var = SpaceVariableDeleteParams::builder().key("TEST_VAR").build();
    space.delete_variable(&del_var).await.unwrap();

    let delete_params = DeleteRepoParams::builder()
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
        .build();
    let _ = client.delete_repo(&delete_params).await;
}
