//! File operations: listing, downloading, uploading, and committing.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example files

use futures::StreamExt;
use hf_hub::types::{AddSource, CommitOperation, RepoTreeEntry};
use hf_hub::{
    CreateRepoParams, DeleteRepoParams, HFClient, RepoCreateCommitParams, RepoDeleteFileParams, RepoDeleteFolderParams,
    RepoDownloadFileParams, RepoGetPathsInfoParams, RepoListFilesParams, RepoListTreeParams, RepoUploadFileParams,
    RepoUploadFolderParams,
};
#[tokio::main]
async fn main() -> hf_hub::Result<()> {
    let api = HFClient::new()?;
    let model = api.model("openai-community", "gpt2");

    // --- Read operations ---

    let files = model.list_files(&RepoListFilesParams::default()).await?;
    println!("Files in gpt2: {}", files.len());
    for f in files.iter().take(5) {
        println!("  - {f}");
    }

    let tree_stream = model.list_tree(&RepoListTreeParams::builder().recursive(true).build())?;
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
        .get_paths_info(
            &RepoGetPathsInfoParams::builder()
                .paths(vec!["config.json".to_string(), "README.md".to_string()])
                .build(),
        )
        .await?;
    println!("\nPaths info for gpt2:");
    for entry in &paths_info {
        println!("  {entry:?}");
    }

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let downloaded = model
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(tmp_dir.path().to_path_buf())
                .build(),
        )
        .await?;
    println!("\nDownloaded gpt2/config.json to: {}", downloaded.display());

    // --- Write operations (creates real resources on the Hub) ---

    let user = api.whoami().await?;
    let unique = std::process::id();
    let repo = api.model(&user.username, format!("example-files-{unique}"));

    api.create_repo(
        &CreateRepoParams::builder()
            .repo_id(repo.repo_path())
            .private(true)
            .exist_ok(true)
            .build(),
    )
    .await?;
    println!("\nCreated test repo: {}", repo.repo_path());

    let commit = repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"Hello from Rust!".to_vec()))
                .path_in_repo("hello.txt")
                .commit_message("Add hello.txt via example")
                .build(),
        )
        .await?;
    println!("Uploaded hello.txt: {:?}", commit.commit_url);

    let commit = repo
        .create_commit(
            &RepoCreateCommitParams::builder()
                .operations(vec![
                    CommitOperation::Add {
                        path_in_repo: "data/file1.txt".to_string(),
                        source: AddSource::Bytes(b"File 1 content".to_vec()),
                    },
                    CommitOperation::Add {
                        path_in_repo: "data/file2.txt".to_string(),
                        source: AddSource::Bytes(b"File 2 content".to_vec()),
                    },
                ])
                .commit_message("Add data files via create_commit")
                .build(),
        )
        .await?;
    println!("Created commit with 2 files: {:?}", commit.commit_oid);

    let upload_dir = tmp_dir.path().join("upload_folder");
    std::fs::create_dir_all(upload_dir.join("subdir")).expect("failed to create dir");
    std::fs::write(upload_dir.join("root.txt"), "root file").expect("failed to write");
    std::fs::write(upload_dir.join("subdir/nested.txt"), "nested file").expect("failed to write");

    let commit = repo
        .upload_folder(
            &RepoUploadFolderParams::builder()
                .folder_path(upload_dir)
                .path_in_repo("uploaded")
                .commit_message("Upload folder via example")
                .build(),
        )
        .await?;
    println!("Uploaded folder: {:?}", commit.commit_oid);

    repo.delete_file(&RepoDeleteFileParams::builder().path_in_repo("hello.txt").build())
        .await?;
    println!("Deleted hello.txt");

    repo.delete_folder(&RepoDeleteFolderParams::builder().path_in_repo("data").build())
        .await?;
    println!("Deleted data/ folder");

    api.delete_repo(&DeleteRepoParams::builder().repo_id(repo.repo_path()).missing_ok(true).build())
        .await?;
    println!("Cleaned up test repo");

    Ok(())
}
