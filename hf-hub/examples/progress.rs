//! Download a file with progress tracking.
//!
//! Demonstrates implementing the `ProgressHandler` trait to receive
//! real-time progress callbacks during file transfers.
//!
//! Run: cargo run -p hf-hub --example progress

use std::sync::Arc;

use hf_hub::{
    DownloadEvent, FileStatus, HFClient, ProgressEvent, ProgressHandler, RepoDownloadFileParams, UploadEvent,
    UploadPhase,
};

struct PrintProgressHandler;

impl ProgressHandler for PrintProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(dl) => match dl {
                DownloadEvent::Start {
                    total_files,
                    total_bytes,
                } => {
                    println!("Starting download: {total_files} file(s), {total_bytes} bytes");
                },
                DownloadEvent::Progress { files } => {
                    for f in files {
                        let pct = if f.total_bytes > 0 {
                            f.bytes_completed * 100 / f.total_bytes
                        } else {
                            0
                        };
                        let status = match f.status {
                            FileStatus::Started => "started",
                            FileStatus::InProgress => "downloading",
                            FileStatus::Complete => "complete",
                        };
                        println!("  {}: {pct}% ({}/{}) [{status}]", f.filename, f.bytes_completed, f.total_bytes);
                    }
                },
                DownloadEvent::AggregateProgress {
                    bytes_completed,
                    total_bytes,
                    ..
                } => {
                    println!("  aggregate: {bytes_completed}/{total_bytes}");
                },
                DownloadEvent::Complete => {
                    println!("Download complete.");
                },
            },
            ProgressEvent::Upload(ul) => match ul {
                UploadEvent::Start {
                    total_files,
                    total_bytes,
                } => {
                    println!("Starting upload: {total_files} file(s), {total_bytes} bytes");
                },
                UploadEvent::Progress {
                    phase,
                    bytes_completed,
                    total_bytes,
                    bytes_per_sec,
                    transfer_bytes_completed,
                    transfer_bytes,
                    transfer_bytes_per_sec,
                    files,
                } => {
                    let phase_str = match phase {
                        UploadPhase::Preparing => "preparing",
                        UploadPhase::CheckingUploadMode => "checking upload mode",
                        UploadPhase::Uploading => "uploading",
                        UploadPhase::Committing => "committing",
                    };
                    let rate = bytes_per_sec.map(|r| format!(" ({:.1} B/s)", r)).unwrap_or_default();
                    println!("  [{phase_str}] {bytes_completed}/{total_bytes}{rate}");
                    if *transfer_bytes > 0 {
                        let transfer_rate =
                            transfer_bytes_per_sec.map(|r| format!(" ({:.1} B/s)", r)).unwrap_or_default();
                        println!("    transfer: {transfer_bytes_completed}/{transfer_bytes}{transfer_rate}");
                    }
                    for f in files {
                        let status = match f.status {
                            FileStatus::Started => "started",
                            FileStatus::InProgress => "uploading",
                            FileStatus::Complete => "complete",
                        };
                        let pct = if f.total_bytes > 0 {
                            f.bytes_completed * 100 / f.total_bytes
                        } else {
                            0
                        };
                        println!("    {}: {pct}% ({}/{}) [{status}]", f.filename, f.bytes_completed, f.total_bytes);
                    }
                },
                UploadEvent::FileComplete { files, phase } => {
                    println!("  Files completed during {phase:?}: {}", files.join(", "));
                },
                UploadEvent::Complete => {
                    println!("Upload complete.");
                },
            },
        }
    }
}

#[tokio::main]
async fn main() -> hf_hub::Result<()> {
    let api = HFClient::new()?;
    let model = api.model("openai-community", "gpt2");

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");

    let path = model
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("config.json")
                .local_dir(tmp_dir.path().to_path_buf())
                .progress(Some(Arc::new(PrintProgressHandler)))
                .build(),
        )
        .await?;

    println!("File saved to: {}", path.display());
    Ok(())
}
