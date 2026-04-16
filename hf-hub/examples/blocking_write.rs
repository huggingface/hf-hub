//! Synchronous write operations using HFClientSync.
//!
//! Creates a temporary repo, uploads files, manages branches, and cleans up.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p hf-hub --features blocking --example blocking_write

use hf_hub::{
    AddSource, CommitOperation, CreateRepoParams, DeleteRepoParams, HFClientSync, RepoCreateBranchParams,
    RepoCreateCommitParams, RepoCreateTagParams, RepoDeleteBranchParams, RepoDeleteFileParams, RepoDeleteTagParams,
    RepoDownloadFileParams, RepoListFilesParams, RepoListRefsParams, RepoUploadFileParams, RepoUploadFolderParams,
};

fn main() -> hf_hub::Result<()> {
    let api = HFClientSync::new()?;
    let user = api.whoami()?;
    let unique = std::process::id();
    let repo = api.model(&user.username, format!("sync-example-{unique}"));

    // --- Create repo ---

    let repo_url = api.create_repo(
        &CreateRepoParams::builder()
            .repo_id(repo.repo_path())
            .private(true)
            .exist_ok(true)
            .build(),
    )?;
    println!("Created repo: {}", repo_url.url);

    // --- Upload a single file ---

    let commit = repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(b"Hello from HFClientSync!".to_vec()))
            .path_in_repo("hello.txt")
            .commit_message("Add hello.txt")
            .build(),
    )?;
    println!("Uploaded hello.txt: {:?}", commit.commit_url);

    // --- Create a multi-file commit ---

    let commit = repo.create_commit(
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
            .commit_message("Add data files")
            .build(),
    )?;
    println!("Created commit: {:?}", commit.commit_oid);

    // --- Upload a folder ---

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    std::fs::write(tmp_dir.path().join("root.txt"), "root file").unwrap();
    std::fs::create_dir_all(tmp_dir.path().join("subdir")).unwrap();
    std::fs::write(tmp_dir.path().join("subdir/nested.txt"), "nested file").unwrap();

    let commit = repo.upload_folder(
        &RepoUploadFolderParams::builder()
            .folder_path(tmp_dir.path().to_path_buf())
            .path_in_repo("uploaded")
            .commit_message("Upload folder")
            .build(),
    )?;
    println!("Uploaded folder: {:?}", commit.commit_oid);

    // --- List files ---

    let files = repo.list_files(&RepoListFilesParams::default())?;
    println!("\nAll files in repo:");
    for f in &files {
        println!("  - {f}");
    }

    // --- Download a file ---

    let download_dir = tempfile::tempdir().expect("failed to create tempdir");
    let path = repo.download_file(
        &RepoDownloadFileParams::builder()
            .filename("hello.txt")
            .local_dir(download_dir.path().to_path_buf())
            .build(),
    )?;
    let content = std::fs::read_to_string(&path).unwrap();
    println!("\nDownloaded hello.txt: {content:?}");

    // --- Branch and tag management ---

    repo.create_branch(&RepoCreateBranchParams::builder().branch("dev").build())?;
    println!("\nCreated branch 'dev'");

    repo.create_tag(&RepoCreateTagParams::builder().tag("v1.0").message("First release").build())?;
    println!("Created tag 'v1.0'");

    let refs = repo.list_refs(&RepoListRefsParams::default())?;
    println!("Branches: {:?}", refs.branches.iter().map(|b| &b.name).collect::<Vec<_>>());
    println!("Tags: {:?}", refs.tags.iter().map(|t| &t.name).collect::<Vec<_>>());

    repo.delete_tag(&RepoDeleteTagParams::builder().tag("v1.0").build())?;
    repo.delete_branch(&RepoDeleteBranchParams::builder().branch("dev").build())?;
    println!("Cleaned up branch and tag");

    // --- Delete a file ---

    repo.delete_file(&RepoDeleteFileParams::builder().path_in_repo("hello.txt").build())?;
    let gone = !repo.file_exists(&hf_hub::RepoFileExistsParams::builder().filename("hello.txt").build())?;
    println!("\nhello.txt deleted: {gone}");

    // --- Clean up ---

    api.delete_repo(&DeleteRepoParams::builder().repo_id(repo.repo_path()).missing_ok(true).build())?;
    println!("Deleted repo");

    Ok(())
}
