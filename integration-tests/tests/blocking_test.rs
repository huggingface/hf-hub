//! Integration tests for the synchronous HFClientSync wrapper.
//!
//! Read-only tests use hardcoded prod repos via `prod_sync_api()`.
//! Write tests using hub-ci via `ci_sync_api()` (require HF_TEST_WRITE=1).
//!
//! Local: HF_TOKEN=hf_xxx cargo test -p hf-hub --features blocking --test blocking_test
//! CI: The workflow sets HF_CI_TOKEN + HF_PROD_TOKEN.

use hf_hub::repository::*;
use hf_hub::{HFClientBuilder, HFClientSync};
use integration_tests::test_utils::*;

fn prod_sync_api() -> Option<HFClientSync> {
    let client = if is_ci() {
        let token = resolve_prod_token()?;
        HFClientBuilder::new()
            .token(token)
            .endpoint(PROD_ENDPOINT)
            .build()
            .expect("Failed to create HFClient")
    } else {
        if std::env::var(HF_TOKEN).is_err() {
            return None;
        }
        HFClientBuilder::new().build().expect("Failed to create HFClient")
    };
    Some(HFClientSync::from_inner(client).expect("Failed to create HFClientSync"))
}

fn ci_sync_api() -> Option<HFClientSync> {
    let client = if is_ci() {
        let token = std::env::var(HF_CI_TOKEN).ok()?;
        HFClientBuilder::new()
            .token(token)
            .endpoint(HUB_CI_ENDPOINT)
            .build()
            .expect("Failed to create HFClient")
    } else {
        if std::env::var(HF_TOKEN).is_err() {
            return None;
        }
        HFClientBuilder::new().build().expect("Failed to create HFClient")
    };
    Some(HFClientSync::from_inner(client).expect("Failed to create HFClientSync"))
}

const TEST_ORG: &str = "huggingface";
const TEST_USER: &str = "julien-c";
const TEST_MODEL_AUTHOR: &str = "openai-community";
const TEST_MODEL_REPO: &str = "hf-internal-testing/tiny-gemma3";
const TEST_DATASET_REPO: &str = "hf-internal-testing/cats_vs_dogs_sample";

/// Split a `"owner/name"` string into an `HFRepositorySync` handle.
fn repo_handle(client: &HFClientSync, repo_id: &str) -> hf_hub::HFRepositorySync {
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    if parts.len() == 2 {
        client.model(parts[0], parts[1])
    } else {
        client.model("", repo_id)
    }
}

fn collect_file_paths(test_repo: &hf_hub::HFRepositorySync) -> std::collections::HashSet<String> {
    test_repo
        .list_tree()
        .recursive(true)
        .send()
        .unwrap()
        .into_iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File { path, .. } => Some(path),
            _ => None,
        })
        .collect()
}

// --- Repo info ---

#[test]
fn test_sync_model_info() {
    let Some(client) = prod_sync_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    let info = client.model_info().repo_id(model_repo).send().unwrap();
    assert!(info.id.contains("tiny-gemma3"));
}

#[test]
fn test_sync_dataset_info() {
    let Some(client) = prod_sync_api() else { return };
    let dataset_repo = TEST_DATASET_REPO;
    let info = client.dataset_info().repo_id(dataset_repo).send().unwrap();
    assert_eq!(info.id, dataset_repo);
}

#[test]
fn test_sync_repo_handle_info_and_file_exists() {
    let Some(client) = prod_sync_api() else { return };
    let model_repo = TEST_MODEL_REPO;
    let repo = repo_handle(&client, model_repo);

    let info = client.model_info().repo_id(model_repo).send().unwrap();
    assert_eq!(info.id, model_repo);

    let exists = repo.file_exists().filename("config.json").send().unwrap();
    assert!(exists);
}

#[test]
fn test_sync_repo_exists() {
    let Some(client) = prod_sync_api() else { return };
    assert!(repo_handle(&client, TEST_MODEL_REPO).exists().send().unwrap());
    assert!(
        !repo_handle(&client, "this-repo-definitely-does-not-exist-12345")
            .exists()
            .send()
            .unwrap()
    );
}

#[test]
fn test_sync_file_exists() {
    let Some(client) = prod_sync_api() else { return };
    let repo = repo_handle(&client, TEST_MODEL_REPO);
    assert!(repo.file_exists().filename("config.json").send().unwrap());
    assert!(!repo.file_exists().filename("nonexistent_file.xyz").send().unwrap());
}

// --- Listing (stream methods collected to Vec) ---

#[test]
fn test_sync_list_models() {
    let Some(client) = prod_sync_api() else { return };
    let author = TEST_MODEL_AUTHOR;
    let models = client.list_models().author(author).limit(3_usize).send().unwrap();
    assert!(!models.is_empty());
    assert!(models[0].id.starts_with(&format!("{}/", author)));
}

#[test]
fn test_sync_list_datasets() {
    let Some(client) = prod_sync_api() else { return };
    let datasets = client.list_datasets().author(TEST_ORG).limit(3_usize).send().unwrap();
    assert!(!datasets.is_empty());
}

#[test]
fn test_sync_list_repo_tree() {
    let Some(client) = prod_sync_api() else { return };
    let entries = repo_handle(&client, TEST_MODEL_REPO).list_tree().send().unwrap();
    let has_config = entries
        .iter()
        .any(|e| matches!(e, RepoTreeEntry::File { path, .. } if path == "config.json"));
    assert!(has_config);
}

#[test]
fn test_sync_list_repo_commits() {
    let Some(client) = prod_sync_api() else { return };
    let commits = repo_handle(&client, TEST_MODEL_REPO).list_commits().send().unwrap();
    assert!(!commits.is_empty());
    assert!(!commits[0].id.is_empty());
    assert!(!commits[0].title.is_empty());
}

// --- Refs ---

#[test]
fn test_sync_list_repo_refs() {
    let Some(client) = prod_sync_api() else { return };
    let refs = repo_handle(&client, TEST_MODEL_REPO).list_refs().send().unwrap();
    assert!(!refs.branches.is_empty());
    assert!(refs.branches.iter().any(|b| b.name == "main"));
}

#[test]
fn test_sync_revision_exists() {
    let Some(client) = prod_sync_api() else { return };
    let repo = repo_handle(&client, TEST_MODEL_REPO);
    assert!(repo.revision_exists().revision("main").send().unwrap());
    assert!(!repo.revision_exists().revision("nonexistent-branch-xyz").send().unwrap());
}

// --- Download ---

#[test]
fn test_sync_download_file() {
    let Some(client) = prod_sync_api() else { return };
    let dir = tempfile::tempdir().unwrap();
    let path = repo_handle(&client, TEST_MODEL_REPO)
        .download_file()
        .filename("config.json")
        .local_dir(dir.path().to_path_buf())
        .send()
        .unwrap();
    assert!(path.exists());
    let content = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("model_type").is_some());
}

// --- Users ---

#[test]
fn test_sync_whoami() {
    let Some(client) = ci_sync_api() else { return };
    let user = client.whoami().send().unwrap();
    assert!(!user.username.is_empty());
}

#[test]
fn test_sync_user_overview() {
    let Some(client) = prod_sync_api() else { return };
    let username = TEST_USER;
    let user = client.user_overview().username(username).send().unwrap();
    assert_eq!(user.username, username);
}

#[test]
fn test_sync_organization_overview() {
    let Some(client) = prod_sync_api() else { return };
    let org_name = TEST_ORG;
    let org = client.organization_overview().organization(org_name).send().unwrap();
    assert_eq!(org.name, org_name);
}

#[test]
fn test_sync_list_organization_members() {
    let Some(client) = prod_sync_api() else { return };
    let members = client.list_organization_members().organization(TEST_ORG).send().unwrap();
    assert!(!members.is_empty());
}

// --- Diffs ---

#[test]
fn test_sync_get_commit_diff() {
    let Some(client) = prod_sync_api() else { return };
    let gpt2 = repo_handle(&client, TEST_MODEL_REPO);
    let commits = gpt2.list_commits().send().unwrap();
    assert!(commits.len() >= 2);

    let diff = gpt2
        .get_commit_diff()
        .compare(format!("{}..{}", commits[1].id, commits[0].id))
        .send()
        .unwrap();
    assert!(!diff.is_empty());
}

#[test]
fn test_sync_get_raw_diff_stream() {
    let Some(client) = prod_sync_api() else { return };
    let gpt2 = repo_handle(&client, TEST_MODEL_REPO);
    let commits = gpt2.list_commits().send().unwrap();
    assert!(commits.len() >= 2);

    let diffs = gpt2
        .get_raw_diff_stream()
        .compare(format!("{}..{}", commits[1].id, commits[0].id))
        .send()
        .unwrap();
    assert!(!diffs.is_empty());
    assert!(!diffs[0].file_path.is_empty());
}

// --- Write operations ---

fn uuid_v4_short() -> String {
    format!("{:016x}", rand::random::<u64>())
}

fn create_test_repo(client: &HFClientSync) -> String {
    let whoami = client.whoami().send().expect("whoami failed");
    let repo_id = format!("{}/hf-hub-sync-test-{}", whoami.username, uuid_v4_short());
    client
        .create_repo()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(false)
        .send()
        .expect("create_repo failed");

    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let test_repo = client.model(parts[0], parts[1]);
    test_repo
        .upload_file()
        .source(AddSource::bytes(b"initial content"))
        .path_in_repo("README.md")
        .commit_message("initial commit")
        .send()
        .expect("seed upload failed");

    repo_id
}

fn delete_test_repo(client: &HFClientSync, repo_id: &str) {
    let _ = client.delete_repo().repo_id(repo_id).send();
}

#[test]
fn test_sync_create_and_delete_repo() {
    let Some(client) = ci_sync_api() else { return };
    if !write_enabled() {
        return;
    }

    let whoami = client.whoami().send().expect("whoami failed");
    let repo_id = format!("{}/hf-hub-sync-test-{}", whoami.username, uuid_v4_short());

    let url = client
        .create_repo()
        .repo_id(&repo_id)
        .private(true)
        .exist_ok(true)
        .send()
        .unwrap();
    assert!(url.url.contains(&repo_id));

    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let test_repo = client.model(parts[0], parts[1]);

    let commit = test_repo
        .upload_file()
        .source(AddSource::bytes(b"hello world"))
        .path_in_repo("test.txt")
        .commit_message("test upload")
        .send()
        .unwrap();
    assert!(commit.commit_oid.is_some());

    assert!(test_repo.file_exists().filename("test.txt").send().unwrap());

    client.delete_repo().repo_id(&repo_id).send().unwrap();
}

#[test]
fn test_sync_create_commit() {
    let Some(client) = ci_sync_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client);
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let test_repo = client.model(parts[0], parts[1]);

    let commit = test_repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("file_a.txt", b"content a".to_vec()),
            CommitOperation::add_bytes("file_b.txt", b"content b".to_vec()),
        ])
        .commit_message("add two files")
        .send()
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = collect_file_paths(&test_repo);
    assert!(files.contains("file_a.txt"));
    assert!(files.contains("file_b.txt"));

    delete_test_repo(&client, &repo_id);
}

#[test]
fn test_sync_upload_folder() {
    let Some(client) = ci_sync_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client);
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let test_repo = client.model(parts[0], parts[1]);

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hello").unwrap();
    std::fs::create_dir_all(dir.path().join("subdir")).unwrap();
    std::fs::write(dir.path().join("subdir/nested.txt"), "nested").unwrap();

    let commit = test_repo
        .upload_folder()
        .folder_path(dir.path().to_path_buf())
        .commit_message("upload folder")
        .send()
        .unwrap();
    assert!(commit.commit_oid.is_some());

    let files = collect_file_paths(&test_repo);
    assert!(files.contains("hello.txt"));
    assert!(files.contains("subdir/nested.txt"));

    delete_test_repo(&client, &repo_id);
}

#[test]
fn test_sync_branch_operations() {
    let Some(client) = ci_sync_api() else { return };
    if !write_enabled() {
        return;
    }
    let repo_id = create_test_repo(&client);
    let parts: Vec<&str> = repo_id.splitn(2, '/').collect();
    let test_repo = client.model(parts[0], parts[1]);

    test_repo.create_branch().branch("test-branch").send().unwrap();

    let refs = test_repo.list_refs().send().unwrap();
    assert!(refs.branches.iter().any(|b| b.name == "test-branch"));

    test_repo.delete_branch().branch("test-branch").send().unwrap();

    let refs = test_repo.list_refs().send().unwrap();
    assert!(!refs.branches.iter().any(|b| b.name == "test-branch"));

    delete_test_repo(&client, &repo_id);
}
