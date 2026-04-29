//! Synchronous write operations using HFClientSync.
//!
//! Creates a temporary repo, uploads files, manages branches, and cleans up.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --features blocking --example blocking_write

use hf_hub::repository::{AddSource, CommitOperation, RepoTreeEntry};
use hf_hub::{HFClientSync, RepoTypeModel};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;
    let user = client.whoami().send()?;
    let unique = std::process::id();
    let repo = client.model(&user.username, format!("sync-example-{unique}"));

    // --- Create repo ---

    let repo_url = client
        .create_repo::<RepoTypeModel>()
        .repo_id(repo.repo_path())
        .private(true)
        .exist_ok(true)
        .send()?;
    println!("Created repo: {}", repo_url.url);

    // --- Upload a single file ---

    let commit = repo
        .upload_file()
        .source(AddSource::bytes(b"Hello from HFClientSync!"))
        .path_in_repo("hello.txt")
        .commit_message("Add hello.txt")
        .send()?;
    println!("Uploaded hello.txt: {:?}", commit.commit_url);

    // --- Create a multi-file commit ---

    let commit = repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("data/file1.txt", b"File 1 content".to_vec()),
            CommitOperation::add_bytes("data/file2.txt", b"File 2 content".to_vec()),
        ])
        .commit_message("Add data files")
        .send()?;
    println!("Created commit: {:?}", commit.commit_oid);

    // --- Upload a folder ---

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    std::fs::write(tmp_dir.path().join("root.txt"), "root file").unwrap();
    std::fs::create_dir_all(tmp_dir.path().join("subdir")).unwrap();
    std::fs::write(tmp_dir.path().join("subdir/nested.txt"), "nested file").unwrap();

    let commit = repo
        .upload_folder()
        .folder_path(tmp_dir.path().to_path_buf())
        .path_in_repo("uploaded")
        .commit_message("Upload folder")
        .send()?;
    println!("Uploaded folder: {:?}", commit.commit_oid);

    // --- List files ---

    let entries = repo.list_tree().recursive(true).send()?;
    println!("\nAll files in repo:");
    for entry in &entries {
        if let RepoTreeEntry::File { path, .. } = entry {
            println!("  - {path}");
        }
    }

    // --- Download a file ---

    let download_dir = tempfile::tempdir().expect("failed to create tempdir");
    let path = repo
        .download_file()
        .filename("hello.txt")
        .local_dir(download_dir.path().to_path_buf())
        .send()?;
    let content = std::fs::read_to_string(&path).unwrap();
    println!("\nDownloaded hello.txt: {content:?}");

    // --- Branch and tag management ---

    repo.create_branch().branch("dev").send()?;
    println!("\nCreated branch 'dev'");

    repo.create_tag().tag("v1.0").message("First release").send()?;
    println!("Created tag 'v1.0'");

    let refs = repo.list_refs().send()?;
    println!("Branches: {:?}", refs.branches.iter().map(|b| &b.name).collect::<Vec<_>>());
    println!("Tags: {:?}", refs.tags.iter().map(|t| &t.name).collect::<Vec<_>>());

    repo.delete_tag().tag("v1.0").send()?;
    repo.delete_branch().branch("dev").send()?;
    println!("Cleaned up branch and tag");

    // --- Delete a file ---

    repo.delete_file().path_in_repo("hello.txt").send()?;
    let gone = !repo.file_exists().filename("hello.txt").send()?;
    println!("\nhello.txt deleted: {gone}");

    // --- Clean up ---

    client
        .delete_repo::<RepoTypeModel>()
        .repo_id(repo.repo_path())
        .missing_ok(true)
        .send()?;
    println!("Deleted repo");

    Ok(())
}
