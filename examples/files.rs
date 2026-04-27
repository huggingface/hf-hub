//! File operations: listing, downloading, uploading, and committing.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example files

use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repository::{AddSource, CommitOperation, RepoTreeEntry};
#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let model = client.model("openai-community", "gpt2");

    // --- Read operations ---

    let files = model.list_files().send().await?;
    println!("Files in gpt2: {}", files.len());
    for f in files.iter().take(5) {
        println!("  - {f}");
    }

    let tree_stream = model.list_tree().recursive(true).send()?;
    futures::pin_mut!(tree_stream);
    println!("\nTree entries in gpt2:");
    let mut count = 0;
    while let Some(Ok(entry)) = tree_stream.next().await {
        match &entry {
            RepoTreeEntry::File { path, size, .. } => println!("  file: {path} ({size} bytes)"),
            RepoTreeEntry::Directory { path, .. } => println!("  dir:  {path}"),
        }
        count += 1;
        if count >= 5 {
            break;
        }
    }

    let paths_info = model
        .get_paths_info()
        .paths(vec!["config.json".to_string(), "README.md".to_string()])
        .send()
        .await?;
    println!("\nPaths info for gpt2:");
    for entry in &paths_info {
        println!("  {entry:?}");
    }

    let metadata = model.get_file_metadata().filepath("config.json").send().await?;
    println!(
        "\nMetadata for gpt2/config.json: commit={}, size={}, etag={}",
        metadata.commit_hash, metadata.file_size, metadata.etag
    );

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let downloaded = model
        .download_file()
        .filename("config.json")
        .local_dir(tmp_dir.path().to_path_buf())
        .send()
        .await?;
    println!("\nDownloaded gpt2/config.json to: {}", downloaded.display());

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send().await?;
    let unique = std::process::id();
    let repo = client.model(&user.username, format!("example-files-{unique}"));

    client
        .create_repo()
        .repo_id(repo.repo_path())
        .private(true)
        .exist_ok(true)
        .send()
        .await?;
    println!("\nCreated test repo: {}", repo.repo_path());

    let commit = repo
        .upload_file()
        .source(AddSource::Bytes(b"Hello from Rust!".to_vec()))
        .path_in_repo("hello.txt")
        .commit_message("Add hello.txt via example")
        .send()
        .await?;
    println!("Uploaded hello.txt: {:?}", commit.commit_url);

    let commit = repo
        .create_commit()
        .operations(vec![
            CommitOperation::add_bytes("data/file1.txt", b"File 1 content".to_vec()),
            CommitOperation::add_bytes("data/file2.txt", b"File 2 content".to_vec()),
        ])
        .commit_message("Add data files via create_commit")
        .send()
        .await?;
    println!("Created commit with 2 files: {:?}", commit.commit_oid);

    let upload_dir = tmp_dir.path().join("upload_folder");
    std::fs::create_dir_all(upload_dir.join("subdir")).expect("failed to create dir");
    std::fs::write(upload_dir.join("root.txt"), "root file").expect("failed to write");
    std::fs::write(upload_dir.join("subdir/nested.txt"), "nested file").expect("failed to write");

    let commit = repo
        .upload_folder()
        .folder_path(upload_dir)
        .path_in_repo("uploaded")
        .commit_message("Upload folder via example")
        .send()
        .await?;
    println!("Uploaded folder: {:?}", commit.commit_oid);

    repo.delete_file().path_in_repo("hello.txt").send().await?;
    println!("Deleted hello.txt");

    repo.delete_folder().path_in_repo("data").send().await?;
    println!("Deleted data/ folder");

    client.delete_repo().repo_id(repo.repo_path()).missing_ok(true).send().await?;
    println!("Cleaned up test repo");

    Ok(())
}
