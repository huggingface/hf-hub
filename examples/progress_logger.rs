//! Log every progress event from a transfer to stderr.
//!
//! Demonstrates the minimal `ProgressHandler` implementation that emits one
//! timestamped line per event, suitable for plumbing into a logging pipeline
//! or for debugging event sequences. Unlike `progress.rs`, which pretty-prints
//! the fields of each event, this example leans on the `Debug` impl of
//! `ProgressEvent` to produce one structured line per event.
//!
//! Run: cargo run -p examples --example progress_logger
//!
//! To exercise the upload code path too:
//!   HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo run -p examples --example progress_logger

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use hf_hub::HFClient;
use hf_hub::files::{AddSource, RepoDownloadFileParams, RepoSnapshotDownloadParams, RepoUploadFileParams};
use hf_hub::progress::{ProgressEvent, ProgressHandler};
use hf_hub::repo::{CreateRepoParams, DeleteRepoParams};

/// Logs each `ProgressEvent` to stderr with a millisecond offset from when
/// the handler was constructed, plus an event counter. Thread-safe — uses
/// an atomic counter so concurrent invocations produce a coherent sequence
/// number.
struct LoggingProgressHandler {
    started: Instant,
    seq: AtomicU64,
}

impl LoggingProgressHandler {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            seq: AtomicU64::new(0),
        }
    }
}

impl ProgressHandler for LoggingProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        let elapsed_ms = self.started.elapsed().as_millis();
        eprintln!("[#{n:04} +{elapsed_ms:>5}ms] {event:?}");
    }
}

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");

    // --- Download a small file: shows Start → Progress* → Complete ---

    eprintln!("=== download_file ===");
    let handler: Arc<LoggingProgressHandler> = Arc::new(LoggingProgressHandler::new());
    let model = client.model("openai-community", "gpt2");
    let path = model
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(tmp_dir.path().to_path_buf())
                .progress(Some(handler))
                .build(),
        )
        .await?;
    println!("Saved to {}", path.display());

    // --- Snapshot download: shows multiple files and AggregateProgress mixing
    //     with per-file Progress deltas ---

    eprintln!("\n=== snapshot_download (*.json) ===");
    let handler: Arc<LoggingProgressHandler> = Arc::new(LoggingProgressHandler::new());
    let snapshot_dir = tmp_dir.path().join("snapshot");
    model
        .snapshot_download(
            &RepoSnapshotDownloadParams::builder()
                .local_dir(snapshot_dir.clone())
                .allow_patterns(vec!["*.json".to_string()])
                .progress(Some(handler))
                .build(),
        )
        .await?;
    println!("Snapshot saved to {}", snapshot_dir.display());

    // --- Upload: shows Start → Progress* → Committing → Complete ---
    //     Requires HF_TOKEN and HF_TEST_WRITE=1.

    if std::env::var("HF_TOKEN").is_err() || std::env::var("HF_TEST_WRITE").is_err() {
        println!("\nSkipping upload example (set HF_TOKEN and HF_TEST_WRITE=1 to run).");
        return Ok(());
    }

    eprintln!("\n=== upload_file ===");
    let user = client.whoami().await?;
    let repo = client.model(&user.username, format!("example-progress-logger-{}", std::process::id()));
    client
        .create_repo(
            &CreateRepoParams::builder()
                .repo_id(repo.repo_path())
                .private(true)
                .exist_ok(true)
                .build(),
        )
        .await?;

    let handler: Arc<LoggingProgressHandler> = Arc::new(LoggingProgressHandler::new());
    let commit = repo
        .upload_file(
            &RepoUploadFileParams::builder()
                .source(AddSource::Bytes(b"hello from progress_logger".to_vec()))
                .path_in_repo("hello.txt")
                .commit_message("example: progress_logger")
                .progress(Some(handler))
                .build(),
        )
        .await?;
    println!("Committed: {:?}", commit.commit_url);

    client
        .delete_repo(&DeleteRepoParams::builder().repo_id(repo.repo_path()).missing_ok(true).build())
        .await?;

    Ok(())
}
