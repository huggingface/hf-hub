//! Download a file with progress tracking.
//!
//! Demonstrates implementing the `ProgressHandler` trait to receive
//! real-time progress callbacks during file transfers.
//!
//! Run: cargo run -p examples --example progress
//! To observe network transfer separately from reconstruction on an Xet file:
//! cargo run -p examples --example progress -- openai-community/gpt2 model.safetensors

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler, UploadEvent};
use hf_hub::{HFClient, HFError, split_id};

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
                        let pct = (f.bytes_completed * 100).checked_div(f.total_bytes).unwrap_or(0);
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
                    transfer_bytes_completed,
                    transfer_bytes,
                    transfer_bytes_per_sec,
                    ..
                } => {
                    let rate = transfer_bytes_per_sec.map(|r| format!(" ({r:.1} B/s)")).unwrap_or_default();
                    println!("  reconstructed: {bytes_completed}/{total_bytes}");
                    println!("  network: {transfer_bytes_completed}/{transfer_bytes}{rate}");
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
                    bytes_completed,
                    total_bytes,
                    bytes_per_sec,
                    transfer_bytes_completed,
                    transfer_bytes,
                    transfer_bytes_per_sec,
                    files,
                } => {
                    let rate = bytes_per_sec.map(|r| format!(" ({:.1} B/s)", r)).unwrap_or_default();
                    println!("  [uploading] {bytes_completed}/{total_bytes}{rate}");
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
                        let pct = (f.bytes_completed * 100).checked_div(f.total_bytes).unwrap_or(0);
                        println!("    {}: {pct}% ({}/{}) [{status}]", f.filename, f.bytes_completed, f.total_bytes);
                    }
                },
                UploadEvent::Committing => {
                    println!("  Phase: committing");
                },
                UploadEvent::Complete => {
                    println!("Upload complete.");
                },
            },
        }
    }
}

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let mut args = std::env::args().skip(1);
    let repo_id = args.next().unwrap_or_else(|| "openai-community/gpt2".to_string());
    let filename = args.next().unwrap_or_else(|| "config.json".to_string());
    let revision = args.next();
    if args.next().is_some() {
        return Err(HFError::InvalidParameter("usage: progress [REPO_ID [FILENAME [REVISION]]]".to_string()));
    }
    let client = HFClient::new()?;
    let (namespace, name) = split_id(&repo_id);
    let model = client.model(namespace, name);

    let tmp_dir = tempfile::tempdir()?;

    let path = model
        .download_file()
        .filename(filename)
        .maybe_revision(revision)
        .local_dir(tmp_dir.path().to_path_buf())
        .progress(PrintProgressHandler)
        .send()
        .await?;

    println!("File saved to: {}", path.display());
    Ok(())
}
