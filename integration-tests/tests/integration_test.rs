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

async fn collect_file_paths(test_repo: &HFRepository) -> std::collections::HashSet<String> {
    let stream = test_repo.list_tree().recursive(true).send().unwrap();
    futures::pin_mut!(stream);
    let mut files = std::collections::HashSet::new();
    while let Some(entry) = stream.next().await {
        if let RepoTreeEntry::File { path, .. } = entry.unwrap() {
            files.insert(path);
        }
    }
    files
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
    let info = client.model_info().repo_id(model_repo).send().await.unwrap();
    assert!(info.id.contains("tiny-gemma3"));
}

#[tokio::test]
async fn test_repo_handle_info_and_file_exists() {
    let Some(client) = prod_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    let repo = repo(&client, model_repo);

    let info = client.model_info().repo_id(model_repo).send().await.unwrap();
    assert_eq!(info.id, model_repo);

    let exists = repo.file_exists().filename("config.json").send().await.unwrap();
    assert!(exists);
}

#[tokio::test]
async fn test_dataset_info() {
    let Some(client) = prod_api() else { return };
    let dataset_repo = TEST_DATASET_REPO;
    let info = client.dataset_info().repo_id(dataset_repo).send().await.unwrap();
    assert_eq!(info.id, dataset_repo);
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
async fn test_list_models_with_pipeline_tag() {
    let Some(client) = prod_api() else { return };
    let stream = client
        .list_models()
        .pipeline_tag("text-classification")
        .limit(5_usize)
        .send()
        .unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(model) = stream.next().await {
        model.unwrap();
        count += 1;
    }
    assert!(count > 0, "pipeline_tag=text-classification should return at least one model");
}

#[tokio::test]
async fn test_list_models_with_filter_tag() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_models().filter("transformers").limit(5_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(model) = stream.next().await {
        model.unwrap();
        count += 1;
    }
    assert!(count > 0, "filter=transformers should return at least one model");
}

#[tokio::test]
async fn test_list_models_with_search() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_models().search("bert").limit(5_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    let mut any_match = false;
    while let Some(model) = stream.next().await {
        let model = model.unwrap();
        if model.id.to_lowercase().contains("bert") {
            any_match = true;
        }
        count += 1;
    }
    assert!(count > 0, "search=bert should return at least one model");
    assert!(any_match, "at least one returned id should contain the search term");
}

#[tokio::test]
async fn test_list_models_sort_by_downloads() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_models().sort("downloads").full(true).limit(3_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut downloads: Vec<u64> = Vec::new();
    while let Some(model) = stream.next().await {
        let model = model.unwrap();
        if let Some(d) = model.downloads {
            downloads.push(d);
        }
    }
    assert!(!downloads.is_empty(), "sort=downloads with full=true should return populated download counts");
    let sorted = {
        let mut s = downloads.clone();
        s.sort_by(|a, b| b.cmp(a));
        s
    };
    assert_eq!(downloads, sorted, "results should be in descending download order");
}

#[tokio::test]
async fn test_list_models_full_includes_siblings() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_models().full(true).limit(1_usize).send().unwrap();
    futures::pin_mut!(stream);

    let model = stream.next().await.expect("stream should yield at least one model").unwrap();
    let siblings = model.siblings.as_ref().expect("full=true should populate siblings");
    assert!(!siblings.is_empty(), "siblings should not be empty when full=true");
}

#[tokio::test]
async fn test_list_datasets_with_filter() {
    let Some(client) = prod_api() else { return };
    let stream = client
        .list_datasets()
        .filter("task_categories:text-classification")
        .limit(5_usize)
        .send()
        .unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    while let Some(ds) = stream.next().await {
        ds.unwrap();
        count += 1;
    }
    assert!(count > 0, "filter=task_categories:text-classification should return at least one dataset");
}

#[tokio::test]
async fn test_list_spaces_with_sdk_filter() {
    let Some(client) = prod_api() else { return };
    let stream = client.list_spaces().filter("gradio").limit(5_usize).send().unwrap();
    futures::pin_mut!(stream);

    let mut count = 0;
    let mut any_with_sdk = false;
    while let Some(space) = stream.next().await {
        let space = space.unwrap();
        if space.sdk.as_deref() == Some("gradio") {
            any_with_sdk = true;
        }
        count += 1;
    }
    assert!(count > 0, "filter=gradio should return at least one Space");
    assert!(any_with_sdk, "expected at least one returned Space to have sdk=gradio");
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
async fn test_user_overview() {
    let Some(client) = prod_api() else { return };
    let username = TEST_USER;
    let user = client.user_overview().username(username).send().await.unwrap();
    assert_eq!(user.username, username);
}

#[tokio::test]
async fn test_organization_overview() {
    let Some(client) = prod_api() else { return };
    let org_name = TEST_ORG;
    let org = client.organization_overview().organization(org_name).send().await.unwrap();
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
    let info = client.space_info().repo_id(space_repo).send().await.unwrap();
    assert_eq!(info.id, space_repo);
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
async fn test_get_paths_info_mixed_inputs() {
    let Some(client) = prod_api() else { return };
    let r = repo(&client, TEST_MODEL_REPO);

    // Find an actual directory in the repo tree so the test isn't pinned to a specific
    // layout. tiny-gemma3 has subdirectories (e.g. `1_Pooling`); fall back gracefully if
    // it doesn't.
    let stream = r.list_tree().recursive(false).send().unwrap();
    futures::pin_mut!(stream);
    let mut directory_path: Option<String> = None;
    while let Some(entry) = stream.next().await {
        if let RepoTreeEntry::Directory { path, .. } = entry.unwrap() {
            directory_path = Some(path);
            break;
        }
    }

    let mut requested = vec!["config.json".to_string(), "this-path-does-not-exist.bin".to_string()];
    if let Some(ref dir) = directory_path {
        requested.push(dir.clone());
    }
    let entries = r.get_paths_info().paths(requested.clone()).send().await.unwrap();

    // The Hub silently drops missing paths, so we expect the file plus any directory we asked about.
    let expected_count = if directory_path.is_some() { 2 } else { 1 };
    assert_eq!(entries.len(), expected_count, "got {entries:?}");

    let returned_paths: Vec<&str> = entries
        .iter()
        .map(|e| match e {
            RepoTreeEntry::File { path, .. } => path.as_str(),
            RepoTreeEntry::Directory { path, .. } => path.as_str(),
        })
        .collect();
    assert!(returned_paths.contains(&"config.json"));
    assert!(!returned_paths.contains(&"this-path-does-not-exist.bin"));
    if let Some(ref dir) = directory_path {
        assert!(returned_paths.contains(&dir.as_str()));
        let dir_entry = entries
            .iter()
            .find(|e| matches!(e, RepoTreeEntry::Directory { path, .. } if path == dir));
        assert!(dir_entry.is_some(), "expected RepoTreeEntry::Directory for {dir}");
    }
}

#[tokio::test]
async fn test_get_paths_info_returns_lfs_metadata() {
    let Some(client) = prod_api() else { return };
    let r = repo(&client, TEST_MODEL_REPO);

    // Discover a file that has either LFS or Xet metadata so we have something concrete to assert.
    let stream = r.list_tree().recursive(true).expand(true).send().unwrap();
    futures::pin_mut!(stream);
    let mut large_file: Option<String> = None;
    while let Some(entry) = stream.next().await {
        if let RepoTreeEntry::File {
            path, lfs, xet_hash, ..
        } = entry.unwrap()
            && (lfs.is_some() || xet_hash.is_some())
        {
            large_file = Some(path);
            break;
        }
    }
    let Some(filepath) = large_file else {
        eprintln!("skipping: no LFS/xet-backed file found in {TEST_MODEL_REPO}");
        return;
    };

    let entries = r.get_paths_info().paths(vec![filepath.clone()]).send().await.unwrap();
    assert_eq!(entries.len(), 1);
    match &entries[0] {
        RepoTreeEntry::File {
            path, lfs, xet_hash, ..
        } => {
            assert_eq!(path, &filepath);
            assert!(
                lfs.is_some() || xet_hash.is_some(),
                "expected LFS or xet metadata for {filepath}, got entry without either",
            );
        },
        other => panic!("expected File entry, got {other:?}"),
    }
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

#[tokio::test]
async fn test_get_file_metadata_bogus_revision() {
    let Some(client) = prod_api() else { return };
    let err = repo(&client, TEST_MODEL_REPO)
        .get_file_metadata()
        .filepath("config.json")
        .revision("definitely-not-a-real-revision-xyz")
        .send()
        .await
        .unwrap_err();
    // The HEAD endpoint returns 404 either for a missing entry or a missing revision; both
    // are reasonable mappings.
    assert!(
        matches!(
            err,
            hf_hub::HFError::EntryNotFound { .. }
                | hf_hub::HFError::RevisionNotFound { .. }
                | hf_hub::HFError::Http { .. }
        ),
        "unexpected error variant: {err:?}",
    );
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
        ("hf-internal-testing/fixtures_ocr", "28fe12cdf7816b5dde94e22051b2ec8dc74267b7"),
        ("hf-internal-testing/example-documents", "5a0a43c6006b31a6ddbfac7d69234925741a40f6"),
        ("hf-internal-testing/dummy_image_text_data", "d5acb3a48d3127b59457e627c6dce975c20675b0"),
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
        .source(AddSource::bytes(b"hello world"))
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
        .source(AddSource::bytes(b"initial content"))
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

    let files = collect_file_paths(&test_repo).await;
    assert!(files.contains("file_a.txt"));
    assert!(files.contains("file_b.txt"));

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

    let files = collect_file_paths(&test_repo).await;
    assert!(files.contains("hello.txt"));
    assert!(files.contains("subdir/nested.txt"));

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
        .source(AddSource::bytes(b"to delete"))
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
async fn test_delete_branch_missing_returns_error() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    let err = test_repo
        .delete_branch()
        .branch("definitely-not-a-real-branch-xyz")
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, hf_hub::HFError::RepoNotFound { .. } | hf_hub::HFError::Http { .. }),
        "unexpected error variant: {err:?}",
    );

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_delete_tag_missing_returns_error() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    let err = test_repo
        .delete_tag()
        .tag("definitely-not-a-real-tag-xyz")
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, hf_hub::HFError::RepoNotFound { .. } | hf_hub::HFError::Http { .. }),
        "unexpected error variant: {err:?}",
    );

    delete_test_repo(&client, &repo_id).await;
}

#[tokio::test]
async fn test_create_tag_invalid_name_returns_error() {
    let Some(client) = api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client).await;

    let test_repo = repo(&client, &repo_id);
    // Tag names with whitespace/control characters are rejected by the Hub.
    let err = test_repo.create_tag().tag("invalid tag with spaces").send().await.unwrap_err();
    // The Hub may return 400 (Http) or 422 (Http) — assert it's a non-success error.
    assert!(
        matches!(
            err,
            hf_hub::HFError::Http { .. } | hf_hub::HFError::Conflict { .. } | hf_hub::HFError::RepoNotFound { .. }
        ),
        "unexpected error variant: {err:?}",
    );

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
    let _info = client.model_info().repo_id(test_repo.repo_path()).send().await.unwrap();

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
    assert!(!runtime.stage.is_empty());
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
