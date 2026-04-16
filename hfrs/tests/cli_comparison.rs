mod helpers;

use std::sync::OnceLock;

use helpers::{CliRunner, require_cli, require_token, require_write};

fn test_model_repo() -> &'static str {
    "hf-internal-testing/tiny-gemma3"
}

fn test_dataset_repo() -> &'static str {
    "hf-internal-testing/cats_vs_dogs_sample"
}

fn test_dataset_download_repo() -> &'static str {
    "hf-internal-testing/cats_vs_dogs_sample"
}

fn test_model_cache_fragment() -> &'static str {
    "hf-internal-testing--tiny-gemma3"
}

fn test_dataset_search() -> &'static str {
    "cats_vs_dogs_sample"
}

fn test_hf_endpoint() -> &'static str {
    "https://huggingface.co"
}

/// Cached whoami username, fetched once and reused across all tests.
fn whoami_username() -> &'static str {
    static USERNAME: OnceLock<String> = OnceLock::new();
    USERNAME.get_or_init(|| {
        let hfrs = CliRunner::hfrs_ci();
        let out = hfrs
            .run_json(&["auth", "whoami"])
            .expect("whoami should succeed for test setup");
        out.get("username")
            .and_then(|v| v.as_str())
            .expect("whoami response should have a username field")
            .to_string()
    })
}

/// Build a full repo ID like "username/repo-name" using the cached whoami username.
fn full_repo(repo_name: &str) -> String {
    format!("{}/{repo_name}", whoami_username())
}

// --- Basic smoke tests (no token needed) ---

#[test]
fn version_runs() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hfrs"))
        .arg("version")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("hfrs "));
}

#[test]
fn env_runs() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hfrs"))
        .arg("env")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("hfrs version:"));
    assert!(stdout.contains("Platform:"));
}

#[test]
fn help_shows_all_commands() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hfrs"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for cmd in &[
        "auth", "cache", "datasets", "download", "models", "repos", "spaces", "upload", "env", "version",
    ] {
        assert!(stdout.contains(cmd), "help output should contain command '{cmd}'");
    }
}

#[test]
fn models_help_shows_subcommands() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hfrs"))
        .args(["models", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("info"));
    assert!(stdout.contains("list"));
}

#[test]
fn repos_help_shows_subcommands() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_hfrs"))
        .args(["repos", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for cmd in &["create", "delete", "move", "settings", "delete-files", "branch", "tag"] {
        assert!(stdout.contains(cmd), "repos help should contain subcommand '{cmd}'");
    }
}

// --- Models comparison tests ---

#[test]
fn models_list_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs.run_json(&["models", "list", "--limit", "3"]).unwrap();
    let hf_out = hf.run_json(&["models", "list", "--limit", "3"]).unwrap();

    assert!(hfrs_out.is_array(), "hfrs output should be an array");
    assert!(hf_out.is_array(), "hf output should be an array");
    assert_eq!(
        hfrs_out.as_array().unwrap().len(),
        hf_out.as_array().unwrap().len(),
        "Should return same number of models"
    );
}

#[test]
fn models_info_returns_valid_json() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["models", "info", test_model_repo()]).unwrap();

    assert!(out.is_object(), "models info should return an object");
    let id = out.get("id").and_then(|v| v.as_str()).unwrap_or("");
    assert!(id.contains("tiny-gemma3"), "model id should contain tiny-gemma3, got: {id}");
    assert!(out.get("author").is_some());
}

#[test]
fn models_list_with_search_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs
        .run_json(&["models", "list", "--search", test_model_repo(), "--limit", "3"])
        .unwrap();
    let hf_out = hf
        .run_json(&["models", "list", "--search", test_model_repo(), "--limit", "3"])
        .unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(hfrs_out.as_array().unwrap().len(), hf_out.as_array().unwrap().len(),);
}

#[test]
fn models_list_with_author_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs
        .run_json(&["models", "list", "--author", "openai", "--limit", "3"])
        .unwrap();
    let hf_out = hf.run_json(&["models", "list", "--author", "openai", "--limit", "3"]).unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(hfrs_out.as_array().unwrap().len(), hf_out.as_array().unwrap().len(),);
}

// --- Datasets comparison tests ---

#[test]
fn datasets_list_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs.run_json(&["datasets", "list", "--limit", "3"]).unwrap();
    let hf_out = hf.run_json(&["datasets", "list", "--limit", "3"]).unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(hfrs_out.as_array().unwrap().len(), hf_out.as_array().unwrap().len(),);
}

#[test]
fn datasets_info_returns_valid_json() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["datasets", "info", test_dataset_repo()]).unwrap();

    assert!(out.is_object(), "datasets info should return an object");
    let id = out.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let expected_dataset = test_dataset_repo();
    assert!(
        id == expected_dataset || id.ends_with(expected_dataset),
        "dataset id should contain {expected_dataset}, got: {id}"
    );
}

// --- Spaces comparison tests ---

#[test]
fn spaces_list_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs.run_json(&["spaces", "list", "--limit", "3"]).unwrap();
    let hf_out = hf.run_json(&["spaces", "list", "--limit", "3"]).unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(hfrs_out.as_array().unwrap().len(), hf_out.as_array().unwrap().len(),);
}

// --- Auth tests ---

#[test]
fn auth_whoami_returns_valid_json() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["auth", "whoami"]).unwrap();

    assert!(out.is_object(), "whoami should return an object");
    assert!(out.get("username").is_some(), "whoami should have username field");
}

// --- Table output tests ---

#[test]
fn models_list_table_output() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_raw(&["models", "list", "--limit", "3", "--format", "table"]).unwrap();

    assert!(out.contains("ID"), "table should have ID header");
    assert!(out.contains("Author"), "table should have Author header");
}

#[test]
fn models_list_quiet_output() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_raw(&["models", "list", "--limit", "3", "--quiet"]).unwrap();

    let lines: Vec<&str> = out.trim().lines().collect();
    assert_eq!(lines.len(), 3, "quiet mode should output 3 IDs");
    for line in &lines {
        assert!(!line.contains(' '), "quiet mode lines should be plain IDs, got: '{line}'");
    }
}

// --- Models field structure tests ---

#[test]
fn models_list_json_has_expected_fields() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["models", "list", "--limit", "5"]).unwrap();

    let items = out.as_array().expect("models list should return an array");
    assert!(!items.is_empty(), "models list should return results");
    for item in items {
        assert!(item.get("id").is_some(), "model item should have 'id' field");
        assert!(item.get("author").is_some(), "model item should have 'author' field");
        assert!(item.get("tags").is_some(), "model item should have 'tags' field");
    }
}

#[test]
fn models_list_sort_by_downloads() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs
        .run_json(&["models", "list", "--sort", "downloads", "--limit", "5"])
        .unwrap();

    let items = out.as_array().expect("models list should return an array");
    assert!(!items.is_empty(), "sorted models list should return results");
    let downloads: Vec<u64> = items
        .iter()
        .filter_map(|item| item.get("downloads").and_then(|d| d.as_u64()))
        .collect();
    for i in 1..downloads.len() {
        assert!(
            downloads[i - 1] >= downloads[i],
            "models should be sorted by downloads descending: {} < {} at index {}",
            downloads[i - 1],
            downloads[i],
            i
        );
    }
}

#[test]
fn models_info_gpt2_has_expected_structure() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["models", "info", test_model_repo()]).unwrap();

    assert!(out.is_object(), "models info should return an object");
    for field in &["id", "author", "tags", "pipeline_tag", "library_name"] {
        assert!(out.get(*field).is_some(), "gpt2 info should have '{field}' field");
    }
}

// --- Datasets field structure tests ---

#[test]
fn datasets_list_json_has_expected_fields() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["datasets", "list", "--limit", "5"]).unwrap();

    let items = out.as_array().expect("datasets list should return an array");
    assert!(!items.is_empty(), "datasets list should return results");
    for item in items {
        assert!(item.get("id").is_some(), "dataset item should have 'id' field");
        assert!(item.get("author").is_some(), "dataset item should have 'author' field");
        assert!(item.get("tags").is_some(), "dataset item should have 'tags' field");
    }
}

#[test]
fn datasets_list_with_search() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs
        .run_json(&["datasets", "list", "--search", test_dataset_search(), "--limit", "3"])
        .unwrap();

    let items = out.as_array().expect("datasets list should return an array");
    assert!(!items.is_empty(), "datasets search should return results");
}

// --- Spaces field structure tests ---

#[test]
fn spaces_list_json_has_expected_fields() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["spaces", "list", "--limit", "5"]).unwrap();

    let items = out.as_array().expect("spaces list should return an array");
    assert!(!items.is_empty(), "spaces list should return results");
    for item in items {
        assert!(item.get("id").is_some(), "space item should have 'id' field");
        assert!(item.get("author").is_some(), "space item should have 'author' field");
        assert!(item.get("sdk").is_some(), "space item should have 'sdk' field");
    }
}

// --- Auth tests ---

#[test]
fn auth_whoami_has_username() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["auth", "whoami"]).unwrap();

    assert!(out.is_object(), "whoami should return an object");
    let username = out
        .get("username")
        .and_then(|v| v.as_str())
        .expect("whoami should have username field");
    assert!(!username.is_empty(), "username should not be empty");
}

// --- Repos tag list tests ---

#[test]
fn repos_tag_list_gpt2() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_json(&["repos", "tag", "list", test_model_repo()]).unwrap();

    assert!(out.is_array(), "repos tag list should return an array");
}

// --- Comparison tests ---

#[test]
fn models_list_sort_comparison() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs
        .run_json(&["models", "list", "--sort", "downloads", "--limit", "3"])
        .unwrap();
    let hf_out = hf.run_json(&["models", "list", "--sort", "downloads", "--limit", "3"]).unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(
        hfrs_out.as_array().unwrap().len(),
        hf_out.as_array().unwrap().len(),
        "both CLIs should return the same number of results when sorting by downloads"
    );
}

#[test]
fn datasets_list_with_search_matches_hf() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let hf = CliRunner::new("hf");
    require_cli(&hf);

    let hfrs_out = hfrs
        .run_json(&["datasets", "list", "--search", test_dataset_search(), "--limit", "3"])
        .unwrap();
    let hf_out = hf
        .run_json(&["datasets", "list", "--search", test_dataset_search(), "--limit", "3"])
        .unwrap();

    assert!(hfrs_out.is_array());
    assert!(hf_out.is_array());
    assert_eq!(
        hfrs_out.as_array().unwrap().len(),
        hf_out.as_array().unwrap().len(),
        "both CLIs should return the same number of results for dataset search"
    );
}

// --- Output format tests ---

#[test]
fn datasets_list_table_has_headers() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs
        .run_raw(&["datasets", "list", "--limit", "3", "--format", "table"])
        .unwrap();

    assert!(out.contains("ID"), "datasets table should have 'ID' header");
    assert!(out.contains("Author"), "datasets table should have 'Author' header");
}

#[test]
fn spaces_list_quiet_output() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let out = hfrs.run_raw(&["spaces", "list", "--limit", "3", "--quiet"]).unwrap();

    let lines: Vec<&str> = out.trim().lines().collect();
    assert_eq!(lines.len(), 3, "quiet mode should output 3 IDs");
    for line in &lines {
        assert!(!line.contains(' '), "quiet mode lines should be plain IDs, got: '{line}'");
    }
}

// --- Error handling tests ---

#[test]
fn models_info_nonexistent_fails() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs
        .run_expecting_failure(&["models", "info", "nonexistent-model-xyz-12345"])
        .unwrap();

    assert_ne!(code, 0, "models info on nonexistent model should exit with non-zero code");
    assert!(stderr.contains("not found"), "error should mention 'not found', got: {stderr}");
    assert!(stderr.contains("authenticated"), "error should suggest authentication, got: {stderr}");
}

// --- Cache dir tests ---

#[test]
fn download_cache_dir_is_respected() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let cache_dir = tmp_dir.path();

    let result = hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "config.json",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
    ]);
    assert!(result.is_ok(), "download with --cache-dir should succeed: {:?}", result.err());

    let output_path = result.unwrap();
    let output_path = output_path.trim();

    // The downloaded file should be under the specified cache dir, not the default
    assert!(
        output_path.starts_with(cache_dir.to_str().unwrap()),
        "downloaded file should be under --cache-dir ({})  but got: {output_path}",
        cache_dir.display()
    );

    // The file should actually exist
    assert!(std::path::Path::new(output_path).exists(), "downloaded file should exist at: {output_path}");
}

#[test]
fn download_default_cache_dir_not_used_when_overridden() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let cache_dir = tmp_dir.path();

    // Download to custom cache dir
    let result = hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "config.json",
        "--cache-dir",
        cache_dir.to_str().unwrap(),
    ]);
    assert!(result.is_ok());

    // Verify the models--gpt2 repo folder was created inside our custom cache dir
    let entries: Vec<_> = std::fs::read_dir(cache_dir)
        .expect("should be able to read cache dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    assert!(
        entries.iter().any(|name| name.contains(test_model_cache_fragment())),
        "cache dir should contain a model repo folder, found: {entries:?}"
    );
}

// --- Write tests ---

#[test]
fn write_repo_create_and_delete() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = format!(
        "hfrs-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let create_result = hfrs.run_raw(&["repos", "create", &repo_name, "--type", "model"]);
    assert!(create_result.is_ok(), "repo creation should succeed: {:?}", create_result.err());

    let full_repo = full_repo(&repo_name);
    let info_result = hfrs.run_json(&["models", "info", &full_repo]);
    let delete_result = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(info_result.is_ok(), "newly created repo should be retrievable via models info");
    assert!(delete_result.is_ok(), "repo deletion should succeed: {:?}", delete_result.err());
}

#[test]
fn write_repo_create_private() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = format!(
        "hfrs-test-private-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let create_result = hfrs.run_raw(&["repos", "create", &repo_name, "--type", "model", "--private"]);
    assert!(create_result.is_ok(), "private repo creation should succeed: {:?}", create_result.err());

    let full_repo = full_repo(&repo_name);
    let info_result = hfrs.run_json(&["models", "info", &full_repo]);
    let delete_result = hfrs.run_raw(&["repos", "delete", &full_repo]);

    let info = info_result.expect("private repo info should be retrievable by owner");
    assert_eq!(info.get("private").and_then(|v| v.as_bool()), Some(true), "repo should be private");
    assert!(delete_result.is_ok(), "private repo deletion should succeed: {:?}", delete_result.err());
}

#[test]
fn write_branch_create_and_delete() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = format!(
        "hfrs-test-branch-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name, "--type", "model"])
        .expect("repo creation should succeed");

    let branch_result = hfrs.run_raw(&["repos", "branch", "create", &full_repo, "test-branch"]);
    assert!(branch_result.is_ok(), "branch creation should succeed: {:?}", branch_result.err());

    let delete_branch_result = hfrs.run_raw(&["repos", "branch", "delete", &full_repo, "test-branch"]);
    assert!(delete_branch_result.is_ok(), "branch deletion should succeed: {:?}", delete_branch_result.err());

    let delete_repo_result = hfrs.run_raw(&["repos", "delete", &full_repo]);
    assert!(delete_repo_result.is_ok(), "repo deletion should succeed: {:?}", delete_repo_result.err());
}

#[test]
fn write_tag_create_and_delete() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = format!(
        "hfrs-test-tag-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name, "--type", "model"])
        .expect("repo creation should succeed");

    let tag_result = hfrs.run_raw(&["repos", "tag", "create", &full_repo, "v0.1"]);
    let list_result = hfrs.run_json(&["repos", "tag", "list", &full_repo]);
    let delete_tag_result = hfrs.run_raw(&["repos", "tag", "delete", &full_repo, "v0.1"]);
    let delete_repo_result = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(tag_result.is_ok(), "tag creation should succeed: {:?}", tag_result.err());

    let tags = list_result.expect("tag list should succeed");
    let empty = vec![];
    let tag_names: Vec<&str> = tags
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
        .collect();
    assert!(tag_names.contains(&"v0.1"), "tag list should contain 'v0.1', got: {tag_names:?}");

    assert!(delete_tag_result.is_ok(), "tag deletion should succeed: {:?}", delete_tag_result.err());
    assert!(delete_repo_result.is_ok(), "repo deletion should succeed: {:?}", delete_repo_result.err());
}

fn unique_repo_name(prefix: &str) -> String {
    format!(
        "{}-{}",
        prefix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    )
}

// =============================================================================
// Auth offline tests (no token/network needed, use isolated HF_HOME)
// =============================================================================

#[test]
fn auth_offline_login_stores_files() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    let result = hfrs.run_raw(&["auth", "login", "--token-value", "hf_test_token_12345678"]);
    assert!(result.is_ok(), "auth login should succeed: {:?}", result.err());

    let stored_tokens_path = tmp.path().join("stored_tokens");
    assert!(stored_tokens_path.exists(), "stored_tokens file should exist");
    let content = std::fs::read_to_string(&stored_tokens_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed["tokens"]["default"]["token"].as_str(), Some("hf_test_token_12345678"));
    assert_eq!(parsed["active"].as_str(), Some("default"));

    let token_path = tmp.path().join("token");
    assert!(token_path.exists(), "token file should exist");
    let token_content = std::fs::read_to_string(&token_path).unwrap();
    assert_eq!(token_content, "hf_test_token_12345678");
}

#[test]
fn auth_offline_login_no_token_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    let (code, stderr) = hfrs.run_expecting_failure(&["auth", "login"]).unwrap();
    assert_ne!(code, 0);
    assert!(stderr.contains("token") || stderr.contains("Generate"), "error should mention token, got: {stderr}");
}

#[test]
fn auth_offline_login_first_active() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_work_token_1234",
        "--token-name",
        "work",
    ])
    .unwrap();
    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_personal_token_5678",
        "--token-name",
        "personal",
    ])
    .unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    assert!(output.contains("work"), "should list work token");
    assert!(output.contains("personal"), "should list personal token");

    let work_line = output.lines().find(|l| l.contains("work")).unwrap();
    let personal_line = output.lines().find(|l| l.contains("personal")).unwrap();
    assert!(work_line.contains("(active)"), "first stored token (work) should be active");
    assert!(!personal_line.contains("(active)"), "second token (personal) should not be active");
}

#[test]
fn auth_offline_logout_default() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&["auth", "login", "--token-value", "hf_token_1234"]).unwrap();
    hfrs.run_raw(&["auth", "logout"]).unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    assert!(output.contains("No stored tokens"), "should show no tokens after logout, got: {output}");
}

#[test]
fn auth_offline_logout_named_preserves_others() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_token_aaa",
        "--token-name",
        "alpha",
    ])
    .unwrap();
    hfrs.run_raw(&["auth", "login", "--token-value", "hf_token_bbb", "--token-name", "beta"])
        .unwrap();
    hfrs.run_raw(&["auth", "logout", "--token-name", "beta"]).unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    assert!(output.contains("alpha"), "alpha token should still exist");
    assert!(!output.contains("beta"), "beta token should be removed");
}

#[test]
fn auth_offline_logout_active_switches() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_first_token_12",
        "--token-name",
        "first",
    ])
    .unwrap();
    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_second_token_34",
        "--token-name",
        "second",
    ])
    .unwrap();
    // "first" is active; log it out
    hfrs.run_raw(&["auth", "logout", "--token-name", "first"]).unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    assert!(output.contains("second"), "second token should remain");
    assert!(output.contains("(active)"), "remaining token should become active");
}

#[test]
fn auth_offline_logout_last_clears_file() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&["auth", "login", "--token-value", "hf_only_token_xx"]).unwrap();
    let token_path = tmp.path().join("token");
    assert!(token_path.exists(), "token file should exist after login");

    hfrs.run_raw(&["auth", "logout"]).unwrap();
    assert!(!token_path.exists(), "token file should be removed after logging out last token");
}

#[test]
fn auth_offline_switch_updates_active() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_token_aaa_1234",
        "--token-name",
        "aaa",
    ])
    .unwrap();
    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_token_bbb_5678",
        "--token-name",
        "bbb",
    ])
    .unwrap();
    // "aaa" is active; switch to "bbb"
    hfrs.run_raw(&["auth", "switch", "--token-name", "bbb"]).unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    let bbb_line = output.lines().find(|l| l.contains("bbb")).unwrap();
    assert!(bbb_line.contains("(active)"), "bbb should be active after switch");

    let token_content = std::fs::read_to_string(tmp.path().join("token")).unwrap();
    assert_eq!(token_content, "hf_token_bbb_5678");
}

#[test]
fn auth_offline_switch_nonexistent_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    hfrs.run_raw(&["auth", "login", "--token-value", "hf_token_x_1234", "--token-name", "x"])
        .unwrap();

    let (code, stderr) = hfrs
        .run_expecting_failure(&["auth", "switch", "--token-name", "nonexistent"])
        .unwrap();
    assert_ne!(code, 0);
    assert!(stderr.contains("not found"), "should mention token not found, got: {stderr}");
}

#[test]
fn auth_offline_list_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();
    assert!(output.contains("No stored tokens"), "should show empty message, got: {output}");
}

#[test]
fn auth_offline_token_masking() {
    let tmp = tempfile::tempdir().unwrap();
    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap());

    // Short token (<=8 chars) masked as all asterisks
    hfrs.run_raw(&["auth", "login", "--token-value", "short", "--token-name", "short"])
        .unwrap();
    // Long token masked as prefix...suffix
    hfrs.run_raw(&[
        "auth",
        "login",
        "--token-value",
        "hf_abcdefghijklmnop",
        "--token-name",
        "long",
    ])
    .unwrap();

    let output = hfrs.run_raw(&["auth", "list"]).unwrap();

    let short_line = output.lines().find(|l| l.contains("short")).unwrap();
    assert!(short_line.contains("*****"), "short token should be masked as asterisks, got: {short_line}");

    let long_line = output.lines().find(|l| l.contains("long")).unwrap();
    assert!(long_line.contains("hf_a...mnop"), "long token should show prefix...suffix, got: {long_line}");
}

#[test]
fn auth_offline_hf_token_path_override() {
    let tmp = tempfile::tempdir().unwrap();
    let custom_token_path = tmp.path().join("custom_token_file");

    let hfrs = CliRunner::hfrs_isolated(tmp.path().to_str().unwrap())
        .with_env("HF_TOKEN_PATH", custom_token_path.to_str().unwrap());

    hfrs.run_raw(&["auth", "login", "--token-value", "hf_custom_path_token_1234"])
        .unwrap();

    assert!(custom_token_path.exists(), "token should be written to HF_TOKEN_PATH");
    let content = std::fs::read_to_string(&custom_token_path).unwrap();
    assert_eq!(content, "hf_custom_path_token_1234");

    let default_path = tmp.path().join("token");
    assert!(!default_path.exists(), "token should NOT be at default path when HF_TOKEN_PATH is set");
}

// =============================================================================
// Auth integration tests (require HF_TOKEN)
// =============================================================================

#[test]
fn auth_hf_token_env_precedence() {
    require_token();
    let tmp = tempfile::tempdir().unwrap();
    let hf_home = tmp.path().to_str().unwrap();

    // Store a fake token in isolated HF_HOME
    let hfrs_setup = CliRunner::hfrs_isolated(hf_home);
    hfrs_setup
        .run_raw(&["auth", "login", "--token-value", "hf_fake_invalid_token_x"])
        .unwrap();

    // Run with real HF_TOKEN env (from parent) — should take precedence over stored fake
    let hfrs = CliRunner::hfrs().with_env("HF_HOME", hf_home);

    let result = hfrs.run_json(&["auth", "whoami"]);
    assert!(
        result.is_ok(),
        "whoami should succeed with real HF_TOKEN over fake stored token: {:?}",
        result.err()
    );
}

#[test]
fn auth_cli_flag_precedence() {
    require_token();
    let hfrs = CliRunner::hfrs();

    // Pass invalid token via --token flag; even though HF_TOKEN env has a valid one,
    // --token should take precedence and cause a 401
    let (code, _stderr) = hfrs
        .run_expecting_failure(&["--token", "invalid_token_xyz_000", "auth", "whoami"])
        .unwrap();
    assert_ne!(code, 0, "--token flag should override HF_TOKEN env var");
}

#[test]
fn auth_whoami_invalid_token() {
    // No require_token() — we provide an explicitly invalid one
    let hfrs = CliRunner::hfrs().without_token();

    let (code, stderr) = hfrs
        .run_expecting_failure(&["--token", "invalid_token_for_testing", "auth", "whoami"])
        .unwrap();
    assert_ne!(code, 0);
    assert!(
        stderr.contains("Invalid")
            || stderr.contains("expired")
            || stderr.contains("authenticated")
            || stderr.contains("token"),
        "should indicate auth failure, got: {stderr}"
    );
}

// =============================================================================
// Download tests (read-only integration, require HF_TOKEN)
// =============================================================================

#[test]
fn download_single_file_basic() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "config.json",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = result.trim();
    assert!(std::path::Path::new(path).exists(), "downloaded file should exist at: {path}");

    let content = std::fs::read_to_string(tmp.path().join("config.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_object(), "config.json should be valid JSON");
}

#[test]
fn download_local_dir() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "config.json",
        "--local-dir",
        tmp.path().to_str().unwrap(),
    ])
    .unwrap();

    let expected = tmp.path().join("config.json");
    assert!(expected.exists(), "file should be at local_dir/config.json");
}

#[test]
fn download_specific_revision() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "config.json",
        "--revision",
        "main",
        "--local-dir",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(result.is_ok(), "download with --revision main should succeed: {:?}", result.err());

    assert!(tmp.path().join("config.json").exists(), "downloaded file should exist");
}

#[test]
fn download_caching_and_force() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();
    let cache_dir = tmp.path().to_str().unwrap();

    // First download with --force-download to populate fresh cache dir
    let path1 = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "config.json",
            "--cache-dir",
            cache_dir,
            "--force-download",
        ])
        .unwrap()
        .trim()
        .to_string();
    assert!(std::path::Path::new(&path1).exists(), "first download should produce a file");

    // Second download should use cache and return same path
    let path2 = hfrs
        .run_raw(&["download", test_model_repo(), "config.json", "--cache-dir", cache_dir])
        .unwrap()
        .trim()
        .to_string();
    assert_eq!(path1, path2, "cached download should return same path");

    // Force re-download should still succeed
    let path3 = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "config.json",
            "--cache-dir",
            cache_dir,
            "--force-download",
        ])
        .unwrap()
        .trim()
        .to_string();
    assert!(std::path::Path::new(&path3).exists(), "force-downloaded file should exist");
}

#[test]
fn download_quiet_mode() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let (code, stdout, _stderr) = hfrs
        .run_full(&[
            "download",
            test_model_repo(),
            "config.json",
            "--local-dir",
            tmp.path().to_str().unwrap(),
            "--quiet",
        ])
        .unwrap();
    assert_eq!(code, 0, "download should succeed");
    assert!(!stdout.trim().is_empty(), "quiet mode should print the local path");
    assert!(tmp.path().join("config.json").exists(), "file should be downloaded");
}

#[test]
fn download_nonexistent_file() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs
        .run_expecting_failure(&["download", test_model_repo(), "nonexistent-file-xyz.txt"])
        .unwrap();
    assert_ne!(code, 0);
    assert!(stderr.to_lowercase().contains("not found"), "should mention file not found, got: {stderr}");
}

#[test]
fn download_nonexistent_repo() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs
        .run_expecting_failure(&["download", "nonexistent-user-xyz/nonexistent-repo-abc", "config.json"])
        .unwrap();
    assert_ne!(code, 0);
    assert!(stderr.to_lowercase().contains("not found"), "should mention repo not found, got: {stderr}");
}

#[test]
fn download_wrong_repo_type() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let (code, _stderr) = hfrs
        .run_expecting_failure(&["download", test_model_repo(), "config.json", "--type", "dataset"])
        .unwrap();
    assert_ne!(code, 0, "downloading model repo as dataset type should fail");
}

#[test]
fn download_snapshot_entire_repo() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    // Use --include to keep download small while testing snapshot path
    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "--include",
            "*.json",
            "--include",
            "*.txt",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(path.exists(), "snapshot path should exist");
    assert!(path.is_dir(), "snapshot download should return a directory, got: {}", path.display());
    assert!(tmp.path().join("config.json").exists(), "snapshot should contain config.json");
}

#[test]
fn download_snapshot_multiple_files() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "config.json",
            "tokenizer.json",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(path.is_dir(), "multiple filenames should trigger snapshot download");
    assert!(tmp.path().join("config.json").exists(), "config.json should be in snapshot");
    assert!(tmp.path().join("tokenizer.json").exists(), "tokenizer.json should be in snapshot");
}

#[test]
fn download_snapshot_include_pattern() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "--include",
            "*.json",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(path.is_dir(), "include pattern should trigger snapshot download");

    let entries: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    assert!(!entries.is_empty(), "should have downloaded some JSON files");
    for entry in &entries {
        assert!(entry.ends_with(".json"), "all files should be .json, got: {entry}");
    }
}

#[test]
fn download_snapshot_exclude_pattern() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    // Include only JSON files, then exclude tokenizer-related ones
    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "--include",
            "*.json",
            "--exclude",
            "tokenizer*",
            "--exclude",
            "special*",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(path.is_dir());

    let entries: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    assert!(!entries.is_empty(), "should have downloaded some files");
    for entry in &entries {
        assert!(!entry.starts_with("tokenizer"), "excluded tokenizer files should not be present, got: {entry}");
    }
}

#[test]
fn download_snapshot_include_exclude() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "--include",
            "*.json",
            "--exclude",
            "tokenizer*",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(path.is_dir());

    let entries: Vec<String> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    assert!(!entries.is_empty(), "should have some JSON files");
    for entry in &entries {
        assert!(entry.ends_with(".json"), "should only have JSON files, got: {entry}");
        assert!(!entry.starts_with("tokenizer"), "tokenizer files should be excluded, got: {entry}");
    }
}

#[test]
fn download_snapshot_cache_dir() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();
    let local_dir = tmp.path().to_str().unwrap();

    // Use --local-dir to verify snapshot download places files correctly
    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "--include",
            "config.json",
            "--local-dir",
            local_dir,
        ])
        .unwrap();

    let path = result.trim();
    assert!(path.starts_with(local_dir), "snapshot should be under --local-dir ({local_dir}), got: {path}");
    assert!(tmp.path().join("config.json").exists(), "config.json should exist in local dir");
}

#[test]
fn download_snapshot_local_dir() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "--include",
        "config.json",
        "--local-dir",
        tmp.path().to_str().unwrap(),
    ])
    .unwrap();

    assert!(tmp.path().join("config.json").exists(), "config.json should be directly in --local-dir");
}

#[test]
fn download_dataset_type() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    let result = hfrs.run_raw(&[
        "download",
        test_dataset_download_repo(),
        "README.md",
        "--type",
        "dataset",
        "--cache-dir",
        tmp.path().to_str().unwrap(),
    ]);
    assert!(result.is_ok(), "downloading from a dataset repo should succeed: {:?}", result.err());

    let path = result.unwrap().trim().to_string();
    assert!(std::path::Path::new(&path).exists(), "downloaded dataset file should exist at: {path}");
}

#[test]
fn download_local_dir_auto_create() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();
    let nested_dir = tmp.path().join("deeply").join("nested").join("dir");

    assert!(!nested_dir.exists(), "nested dir should not exist yet");

    let result = hfrs.run_raw(&[
        "download",
        test_model_repo(),
        "config.json",
        "--local-dir",
        nested_dir.to_str().unwrap(),
    ]);
    assert!(result.is_ok(), "download should create missing --local-dir: {:?}", result.err());
    assert!(nested_dir.join("config.json").exists(), "file should be at nested local-dir/config.json");
}

#[test]
fn download_non_writable_location() {
    require_token();
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs
        .run_expecting_failure(&[
            "download",
            test_model_repo(),
            "config.json",
            "--local-dir",
            "/nonexistent_root_dir/download",
        ])
        .unwrap();
    assert_ne!(code, 0, "download to non-writable location should fail");
    assert!(!stderr.contains("panic"), "should get a clean error, not a panic, got: {stderr}");
}

#[test]
fn download_single_with_include_uses_snapshot() {
    require_token();
    let hfrs = CliRunner::hfrs();
    let tmp = tempfile::tempdir().unwrap();

    // Single filename + --include should trigger snapshot path (returns directory)
    let result = hfrs
        .run_raw(&[
            "download",
            test_model_repo(),
            "config.json",
            "--include",
            "*.json",
            "--local-dir",
            tmp.path().to_str().unwrap(),
        ])
        .unwrap();

    let path = std::path::Path::new(result.trim());
    assert!(
        path.is_dir(),
        "single filename + --include should trigger snapshot (directory), got: {}",
        path.display()
    );
}

#[test]
fn download_no_repo_id_fails() {
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs.run_expecting_failure(&["download"]).unwrap();
    assert_ne!(code, 0);
    assert!(
        stderr.contains("required") || stderr.contains("Usage") || stderr.contains("usage"),
        "should show usage/required arg error, got: {stderr}"
    );
}

// =============================================================================
// Upload tests (write integration, require HF_TOKEN + HF_TEST_WRITE)
// =============================================================================

#[test]
fn write_upload_single_file() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-file");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("test.txt");
    std::fs::write(&file_path, "hello world").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap()]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload should succeed: {:?}", result.err());
    let output = result.unwrap();
    assert!(output.contains("http"), "output should contain a commit URL, got: {output}");
}

#[test]
fn write_upload_auto_create() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-autocreate");
    let full_repo = full_repo(&repo_name);

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("test.txt");
    std::fs::write(&file_path, "auto-created repo content").unwrap();

    // Upload to a repo that doesn't exist — should auto-create
    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap()]);
    let info_result = hfrs.run_json(&["models", "info", &full_repo]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with auto-create should succeed: {:?}", result.err());
    assert!(info_result.is_ok(), "auto-created repo should exist");
}

#[test]
fn write_upload_private_auto_create() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-private");
    let full_repo = full_repo(&repo_name);

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("test.txt");
    std::fs::write(&file_path, "private repo content").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap(), "--private"]);
    let info_result = hfrs.run_json(&["models", "info", &full_repo]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "private auto-create upload should succeed: {:?}", result.err());
    let info = info_result.expect("private repo should be accessible by owner");
    assert_eq!(info.get("private").and_then(|v| v.as_bool()), Some(true), "repo should be private");
}

#[test]
fn write_upload_path_in_repo() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-path");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("data.txt");
    std::fs::write(&file_path, "nested file content").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap(), "subdir/data.txt"]);

    // Verify by downloading from nested path
    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_result = hfrs.run_raw(&[
        "download",
        &full_repo,
        "subdir/data.txt",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with path_in_repo should succeed: {:?}", result.err());
    assert!(dl_result.is_ok(), "should be able to download file at path_in_repo: {:?}", dl_result.err());
}

#[test]
fn write_upload_commit_message_and_description() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-commit");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("test.txt");
    std::fs::write(&file_path, "commit test").unwrap();

    let result = hfrs.run_raw(&[
        "upload",
        &full_repo,
        file_path.to_str().unwrap(),
        "--commit-message",
        "Custom commit message",
        "--commit-description",
        "Detailed description of changes",
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with commit message and description should succeed: {:?}", result.err());
}

#[test]
fn write_upload_create_pr() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-pr");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    // Need an initial commit for PRs to work
    let tmp = tempfile::tempdir().unwrap();
    let readme_path = tmp.path().join("README.md");
    std::fs::write(&readme_path, "# Test\n").unwrap();
    hfrs.run_raw(&["upload", &full_repo, readme_path.to_str().unwrap(), "README.md"])
        .expect("initial upload should succeed");

    let file_path = tmp.path().join("new_file.txt");
    std::fs::write(&file_path, "PR content").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap(), "--create-pr"]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with --create-pr should succeed: {:?}", result.err());
    let output = result.unwrap();
    assert!(output.contains("http"), "output should contain a PR/commit URL, got: {output}");
}

#[test]
fn write_upload_to_branch() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-branch");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");
    hfrs.run_raw(&["repos", "branch", "create", &full_repo, "test-branch"])
        .expect("branch creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("branch_file.txt");
    std::fs::write(&file_path, "branch content").unwrap();

    let result = hfrs.run_raw(&[
        "upload",
        &full_repo,
        file_path.to_str().unwrap(),
        "--revision",
        "test-branch",
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload to specific branch should succeed: {:?}", result.err());
}

#[test]
fn write_upload_quiet() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-quiet");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("quiet.txt");
    std::fs::write(&file_path, "quiet mode content").unwrap();

    let (code, stdout, _stderr) = hfrs
        .run_full(&["upload", &full_repo, file_path.to_str().unwrap(), "--quiet"])
        .unwrap();
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert_eq!(code, 0, "quiet upload should succeed");
    assert!(!stdout.trim().is_empty(), "quiet mode should print the commit URL");
}

#[test]
fn write_upload_nonexistent_path_fails() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-nopath");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let (code, stderr) = hfrs
        .run_expecting_failure(&["upload", &full_repo, "/nonexistent/path/file.txt"])
        .unwrap();
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert_ne!(code, 0);
    assert!(stderr.contains("does not exist"), "should mention path does not exist, got: {stderr}");
}

#[test]
fn write_upload_folder() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-folder");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("upload_dir");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("a.txt"), "file a").unwrap();
    std::fs::write(folder.join("b.txt"), "file b").unwrap();
    std::fs::create_dir(folder.join("sub")).unwrap();
    std::fs::write(folder.join("sub/c.txt"), "file c").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, folder.to_str().unwrap()]);

    // Verify by downloading
    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_result = hfrs.run_raw(&[
        "download",
        &full_repo,
        "--include",
        "*.txt",
        "--local-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "folder upload should succeed: {:?}", result.err());
    assert!(dl_result.is_ok(), "download should succeed after folder upload: {:?}", dl_result.err());
    assert!(dl_tmp.path().join("a.txt").exists(), "a.txt should be downloadable");
    assert!(dl_tmp.path().join("b.txt").exists(), "b.txt should be downloadable");
}

#[test]
fn write_upload_folder_include() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-include");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("mixed");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("keep.md"), "# keep").unwrap();
    std::fs::write(folder.join("skip.txt"), "skip this").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, folder.to_str().unwrap(), "--include", "*.md"]);

    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_md = hfrs.run_raw(&[
        "download",
        &full_repo,
        "keep.md",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let dl_txt = hfrs.run_raw(&[
        "download",
        &full_repo,
        "skip.txt",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "folder upload with --include should succeed: {:?}", result.err());
    assert!(dl_md.is_ok(), ".md file should have been uploaded");
    assert!(dl_txt.is_err(), ".txt file should NOT have been uploaded");
}

#[test]
fn write_upload_folder_exclude() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-exclude");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("content");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("good.txt"), "good").unwrap();
    std::fs::write(folder.join("bad.log"), "bad").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, folder.to_str().unwrap(), "--exclude", "*.log"]);

    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_txt = hfrs.run_raw(&[
        "download",
        &full_repo,
        "good.txt",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let dl_log = hfrs.run_raw(&[
        "download",
        &full_repo,
        "bad.log",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "folder upload with --exclude should succeed: {:?}", result.err());
    assert!(dl_txt.is_ok(), ".txt file should have been uploaded");
    assert!(dl_log.is_err(), ".log file should NOT have been uploaded");
}

#[test]
fn write_upload_folder_delete() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-delete");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();

    // First upload: include a .old file
    let folder1 = tmp.path().join("v1");
    std::fs::create_dir(&folder1).unwrap();
    std::fs::write(folder1.join("current.txt"), "current").unwrap();
    std::fs::write(folder1.join("legacy.old"), "old stuff").unwrap();
    hfrs.run_raw(&["upload", &full_repo, folder1.to_str().unwrap()])
        .expect("first upload");

    // Second upload with --delete "*.old"
    let folder2 = tmp.path().join("v2");
    std::fs::create_dir(&folder2).unwrap();
    std::fs::write(folder2.join("new.txt"), "new content").unwrap();
    let result = hfrs.run_raw(&["upload", &full_repo, folder2.to_str().unwrap(), "--delete", "*.old"]);

    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_old = hfrs.run_raw(&[
        "download",
        &full_repo,
        "legacy.old",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with --delete should succeed: {:?}", result.err());
    assert!(dl_old.is_err(), ".old file should have been deleted from repo");
}

#[test]
fn write_upload_folder_path_in_repo() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-folder-path");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("data");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("x.txt"), "x content").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, folder.to_str().unwrap(), "nested/data"]);

    let dl_tmp = tempfile::tempdir().unwrap();
    let dl_result = hfrs.run_raw(&[
        "download",
        &full_repo,
        "nested/data/x.txt",
        "--cache-dir",
        dl_tmp.path().to_str().unwrap(),
    ]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "folder upload with path_in_repo should succeed: {:?}", result.err());
    assert!(dl_result.is_ok(), "file should be downloadable at nested path: {:?}", dl_result.err());
}

#[test]
fn write_upload_empty_excluded_folder() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-empty");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("empty_dir");
    std::fs::create_dir(&folder).unwrap();
    std::fs::write(folder.join("only.txt"), "only file").unwrap();

    // Exclude everything — should handle gracefully (succeed or fail cleanly)
    let (code, _stdout, stderr) = hfrs
        .run_full(&["upload", &full_repo, folder.to_str().unwrap(), "--exclude", "*.txt"])
        .unwrap();
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(!stderr.contains("panic"), "should not panic on empty upload, got: {stderr}");
    // Accept either success (empty commit) or clean failure
    assert!(code == 0 || code == 1, "should exit cleanly, got code: {code}");
}

#[test]
fn upload_no_repo_id_fails() {
    let hfrs = CliRunner::hfrs();

    let (code, stderr) = hfrs.run_expecting_failure(&["upload"]).unwrap();
    assert_ne!(code, 0);
    assert!(
        stderr.contains("required") || stderr.contains("Usage") || stderr.contains("usage"),
        "should show usage/required arg error, got: {stderr}"
    );
}

// Test 58 (upload no write access) skipped — requires a separate user account
// Test 59 (upload --type space) skipped — Spaces require SDK configuration

#[test]
fn write_upload_dataset_type() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-dataset");
    let full_repo = full_repo(&repo_name);

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("data.csv");
    std::fs::write(&file_path, "col1,col2\n1,2\n3,4").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap(), "--type", "dataset"]);

    let info_result = hfrs.run_json(&["datasets", "info", &full_repo]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo, "--type", "dataset"]);

    assert!(result.is_ok(), "upload as dataset should succeed: {:?}", result.err());
    assert!(info_result.is_ok(), "dataset repo should exist: {:?}", info_result.err());
}

#[test]
fn write_upload_large_file() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-large");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("large.bin");

    // Create 11MB random file (above typical LFS threshold of 10MB)
    let mut data = vec![0u8; 11 * 1024 * 1024];
    rand::Fill::fill(&mut data[..], &mut rand::rng());
    std::fs::write(&file_path, &data).unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap()]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "large file upload should succeed (LFS/xet): {:?}", result.err());
}

#[test]
fn write_upload_special_chars() {
    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-upload-special");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("file with spaces & (parens).txt");
    std::fs::write(&file_path, "special chars content").unwrap();

    let result = hfrs.run_raw(&["upload", &full_repo, file_path.to_str().unwrap()]);
    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);

    assert!(result.is_ok(), "upload with special chars in filename should succeed: {:?}", result.err());
}

// =============================================================================
// Cross-cutting tests (color, logging, endpoints, exit codes)
// =============================================================================

#[test]
fn no_color_flag_suppresses_ansi() {
    let hfrs = CliRunner::hfrs().without_token().with_env("CLICOLOR_FORCE", "1");

    // Force color on, then trigger an error
    let (_, stderr_with_color) = hfrs.run_expecting_failure(&["--token", "invalid", "auth", "whoami"]).unwrap();

    // With --no-color, ANSI codes should be gone
    let (_, stderr_no_color) = hfrs
        .run_expecting_failure(&["--no-color", "--token", "invalid", "auth", "whoami"])
        .unwrap();

    assert!(
        stderr_no_color.find('\x1b').is_none(),
        "--no-color should suppress ANSI codes, got: {stderr_no_color}"
    );
    assert!(
        stderr_with_color.find('\x1b').is_some(),
        "CLICOLOR_FORCE=1 should produce ANSI codes, got: {stderr_with_color}"
    );
}

#[test]
fn no_color_env_suppresses_ansi() {
    let hfrs_color = CliRunner::hfrs().without_token().with_env("CLICOLOR_FORCE", "1");

    let hfrs_no_color = CliRunner::hfrs()
        .without_token()
        .with_env("CLICOLOR_FORCE", "1")
        .with_env("NO_COLOR", "1");

    let (_, stderr_color) = hfrs_color
        .run_expecting_failure(&["--token", "invalid", "auth", "whoami"])
        .unwrap();

    let (_, stderr_no_color) = hfrs_no_color
        .run_expecting_failure(&["--token", "invalid", "auth", "whoami"])
        .unwrap();

    assert!(stderr_color.find('\x1b').is_some(), "CLICOLOR_FORCE should produce ANSI codes");
    assert!(
        stderr_no_color.find('\x1b').is_none(),
        "NO_COLOR should suppress ANSI codes even with CLICOLOR_FORCE"
    );
}

#[test]
fn hf_endpoint_override() {
    require_token();
    let hfrs = CliRunner::hfrs().with_env("HF_ENDPOINT", test_hf_endpoint());

    let result = hfrs.run_json(&["auth", "whoami"]);
    assert!(result.is_ok(), "HF_ENDPOINT override should work: {:?}", result.err());
}

#[test]
fn invalid_endpoint_clean_error() {
    let hfrs = CliRunner::hfrs().without_token().with_env("HF_ENDPOINT", "http://localhost:1");

    let (code, stderr) = hfrs
        .run_expecting_failure(&["--token", "fake_token", "auth", "whoami"])
        .unwrap();
    assert_ne!(code, 0);
    assert!(
        stderr.to_lowercase().contains("connection") || stderr.to_lowercase().contains("error"),
        "should give clean connection error, got: {stderr}"
    );
    assert!(!stderr.contains("panic"), "should not panic");
}

#[test]
fn hf_log_level_debug() {
    require_token();
    let hfrs = CliRunner::hfrs().with_env("HF_LOG_LEVEL", "debug");

    let (code, _stdout, stderr) = hfrs.run_full(&["auth", "whoami", "--format", "json"]).unwrap();
    assert_eq!(code, 0);
    // Debug logs (e.g. "resolved authentication token") go to stderr
    assert!(!stderr.is_empty(), "HF_LOG_LEVEL=debug should produce debug output on stderr");
}

#[test]
fn hf_debug_error_chain() {
    let hfrs_no_debug = CliRunner::hfrs().without_token();

    let hfrs_debug = CliRunner::hfrs().without_token().with_env("HF_DEBUG", "1");

    // Without HF_DEBUG, error should suggest setting it
    let (_, stderr_no_debug) = hfrs_no_debug
        .run_expecting_failure(&["--token", "invalid", "auth", "whoami"])
        .unwrap();
    assert!(
        stderr_no_debug.contains("HF_DEBUG=1"),
        "without HF_DEBUG, should suggest setting it, got: {stderr_no_debug}"
    );

    // With HF_DEBUG, the suggestion should NOT appear
    let (_, stderr_debug) = hfrs_debug
        .run_expecting_failure(&["--token", "invalid", "auth", "whoami"])
        .unwrap();
    assert!(
        !stderr_debug.contains("Set HF_DEBUG=1"),
        "with HF_DEBUG=1, should not show the suggestion, got: {stderr_debug}"
    );
}

#[test]
fn exit_codes() {
    require_token();
    let hfrs = CliRunner::hfrs();

    // Success
    let (code, _, _) = hfrs.run_full(&["version"]).unwrap();
    assert_eq!(code, 0, "successful command should exit 0");

    // Failure
    let (code, _, _) = hfrs.run_full(&["models", "info", "nonexistent-xyz-12345"]).unwrap();
    assert_ne!(code, 0, "failed command should exit non-zero");
}

// =============================================================================
// Signal handling tests (xet upload/download abort via SIGINT)
// =============================================================================

#[cfg(unix)]
#[test]
fn signal_abort_during_xet_upload() {
    use std::time::{Duration, Instant};

    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    let repo_name = unique_repo_name("hfrs-signal-upload");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("large_signal_test.bin");

    // 50MB random file — large enough that upload takes a few seconds
    let mut data = vec![0u8; 50 * 1024 * 1024];
    rand::Fill::fill(&mut data[..], &mut rand::rng());
    std::fs::write(&file_path, &data).unwrap();
    drop(data);

    let mut child = hfrs
        .spawn(&["upload", &full_repo, file_path.to_str().unwrap()])
        .expect("failed to spawn upload");

    let pid = child.id();

    std::thread::sleep(Duration::from_millis(500));

    // Check if process already finished before we could signal it
    if let Ok(Some(_status)) = child.try_wait() {
        eprintln!("upload finished before SIGINT could be sent — skipping abort assertion");
        let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);
        return;
    }

    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGINT);
    }

    let start = Instant::now();
    let timeout = Duration::from_secs(30);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("CLI did not exit within {timeout:?} after SIGINT");
                }
                std::thread::sleep(Duration::from_millis(100));
            },
            Err(e) => panic!("error waiting for child: {e}"),
        }
    };

    assert!(!status.success(), "CLI should exit non-zero after SIGINT, got: {status}");

    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);
}

#[cfg(unix)]
#[test]
fn signal_abort_during_xet_download() {
    use std::time::{Duration, Instant};

    require_token();
    require_write();
    let hfrs = CliRunner::hfrs_ci();

    // First: create a repo with a large xet file
    let repo_name = unique_repo_name("hfrs-signal-download");
    let full_repo = full_repo(&repo_name);

    hfrs.run_raw(&["repos", "create", &repo_name]).expect("repo creation");

    let tmp_upload = tempfile::tempdir().unwrap();
    let upload_path = tmp_upload.path().join("large_for_download.bin");

    let mut data = vec![0u8; 50 * 1024 * 1024];
    rand::Fill::fill(&mut data[..], &mut rand::rng());
    std::fs::write(&upload_path, &data).unwrap();
    drop(data);

    let upload_result = hfrs.run_raw(&["upload", &full_repo, upload_path.to_str().unwrap()]);
    if upload_result.is_err() {
        let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);
        panic!("upload failed, cannot test download abort: {:?}", upload_result.err());
    }
    drop(tmp_upload);

    // Now download and send SIGINT mid-transfer.
    // Use a short delay so the process is still transferring when the signal
    // arrives. If the download completes before we can signal, skip the
    // assertion — we cannot test the abort path on very fast networks.
    let tmp_download = tempfile::tempdir().unwrap();
    let mut child = hfrs
        .spawn(&[
            "download",
            &full_repo,
            "--local-dir",
            tmp_download.path().to_str().unwrap(),
        ])
        .expect("failed to spawn download");

    let pid = child.id();

    std::thread::sleep(Duration::from_millis(500));

    // Check if process already finished before we could signal it
    if let Ok(Some(_status)) = child.try_wait() {
        eprintln!("download finished before SIGINT could be sent — skipping abort assertion");
        let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);
        return;
    }

    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGINT);
    }

    let start = Instant::now();
    let timeout = Duration::from_secs(30);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("CLI did not exit within {timeout:?} after SIGINT");
                }
                std::thread::sleep(Duration::from_millis(100));
            },
            Err(e) => panic!("error waiting for child: {e}"),
        }
    };

    assert!(!status.success(), "CLI should exit non-zero after SIGINT during download, got: {status}");

    let _ = hfrs.run_raw(&["repos", "delete", &full_repo]);
}
