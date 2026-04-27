//! Download and upload files from/to the Hugging Face Hub.
//!
//! Demonstrates:
//! - Downloading a file to the HF cache
//! - Downloading a file to a local directory
//! - Downloading a file as a byte stream
//! - Uploading a file from bytes
//! - Uploading a file from a local path
//!
//! Read operations require no auth. Write operations require HF_TOKEN.
//! Run: cargo run -p examples --example download_upload

use std::io::Write;

use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repository::AddSource;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let model = client.model("openai-community", "gpt2");

    // --- Download to HF cache ---

    let cached_path = model.download_file().filename("config.json").send().await?;
    println!("Downloaded to cache: {}", cached_path.display());

    // --- Download a large xet-backed file to local directory ---

    let xet_repo = client.model("Lightricks", "LTX-2.3");
    let xet_path = xet_repo
        .download_file()
        .filename("ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors")
        .local_dir(tmp_dir.path().to_path_buf())
        .send()
        .await?;
    let xet_size = std::fs::metadata(&xet_path).map(|m| m.len()).unwrap_or(0);
    println!("Downloaded xet file to: {} ({xet_size} bytes)", xet_path.display());

    // --- Download to local directory ---

    let local_path = model
        .download_file()
        .filename("config.json")
        .local_dir(tmp_dir.path().to_path_buf())
        .send()
        .await?;
    let size = std::fs::metadata(&local_path).map(|m| m.len()).unwrap_or(0);
    println!("Downloaded to local dir: {} ({size} bytes)", local_path.display());

    // --- Download as stream ---

    let (content_length, mut stream) = model.download_file_stream().filename("config.json").send().await?;
    println!(
        "Streaming config.json (content-length: {})",
        content_length.map_or("unknown".to_string(), |n| format!("{n}"))
    );

    let stream_dest = tmp_dir.path().join("config_streamed.json");
    let mut file = std::fs::File::create(&stream_dest)?;
    let mut total = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        total += chunk.len() as u64;
        file.write_all(&chunk)?;
    }
    file.flush()?;
    println!("Streamed {total} bytes to {}", stream_dest.display());

    // --- Download as stream and process in memory ---

    let (_content_length, mut stream) = model.download_file_stream().filename("config.json").send().await?;

    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
    }

    let config: serde_json::Value = serde_json::from_slice(&buf)?;
    println!("Parsed config in memory: model_type={}, vocab_size={}", config["model_type"], config["vocab_size"]);

    // --- Download a folder (snapshot) ---

    let snapshot_dir = tmp_dir.path().join("snapshot");
    let snapshot_path = model
        .snapshot_download()
        .local_dir(snapshot_dir)
        .allow_patterns(vec!["*.json".to_string()])
        .send()
        .await?;
    println!("Downloaded snapshot to: {}", snapshot_path.display());
    for entry in std::fs::read_dir(&snapshot_path)? {
        let entry = entry?;
        println!("  {}", entry.file_name().to_string_lossy());
    }

    // --- Upload (requires HF_TOKEN and HF_TEST_WRITE=1) ---

    if std::env::var("HF_TOKEN").is_err() || std::env::var("HF_TEST_WRITE").is_err() {
        println!("\nSkipping upload examples (set HF_TOKEN and HF_TEST_WRITE=1 to run)");
        return Ok(());
    }

    let user = client.whoami().send().await?;
    let repo = client.model(&user.username, format!("example-download-upload-{}", std::process::id()));

    client
        .create_repo()
        .repo_id(repo.repo_path())
        .private(true)
        .exist_ok(true)
        .send()
        .await?;
    println!("\nCreated repo: {}", repo.repo_path());

    // Upload from bytes
    let commit = repo
        .upload_file()
        .source(AddSource::Bytes(b"Hello from Rust!".to_vec()))
        .path_in_repo("hello.txt")
        .commit_message("Add hello.txt from bytes")
        .send()
        .await?;
    println!("Uploaded hello.txt: {:?}", commit.commit_url);

    // Upload from local file
    let local_file = tmp_dir.path().join("local_data.txt");
    std::fs::write(&local_file, "Data from a local file").expect("failed to write local file");

    let commit = repo
        .upload_file()
        .source(AddSource::File(local_file))
        .path_in_repo("data/local_data.txt")
        .commit_message("Add local_data.txt from file path")
        .send()
        .await?;
    println!("Uploaded data/local_data.txt: {:?}", commit.commit_url);

    // Upload a folder
    let upload_dir = tmp_dir.path().join("my_folder");
    std::fs::create_dir_all(upload_dir.join("subdir")).expect("failed to create dirs");
    std::fs::write(upload_dir.join("root.txt"), "root file").expect("failed to write");
    std::fs::write(upload_dir.join("subdir/nested.txt"), "nested file").expect("failed to write");

    let commit = repo
        .upload_folder()
        .folder_path(upload_dir)
        .path_in_repo("uploaded")
        .commit_message("Upload folder with nested files")
        .send()
        .await?;
    println!("Uploaded folder: {:?}", commit.commit_url);

    // Cleanup
    client.delete_repo().repo_id(repo.repo_path()).missing_ok(true).send().await?;
    println!("Cleaned up repo: {}", repo.repo_path());

    Ok(())
}
