//! Upload a large local folder using the high-level repo handle, with a simple
//! aggregate progress logger. Run with:
//!   HF_TOKEN=hf_xxx cargo run -p examples --example upload_large_folder -- <owner> <name> <folder>

use std::sync::Arc;

use hf_hub::HFClient;
use hf_hub::progress::{ProgressEvent, ProgressHandler, UploadEvent};

struct Logger;

impl ProgressHandler for Logger {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Upload(UploadEvent::LargeFolderStatus {
                committed,
                files_total,
                preuploaded,
                lfs_total,
                dedup_bytes_saved,
                ..
            }) => {
                println!(
                    "committed {committed}/{files_total} | lfs {preuploaded}/{lfs_total} uploaded | dedup saved {dedup_bytes_saved} bytes"
                );
            },
            ProgressEvent::Upload(UploadEvent::Complete) => println!("done"),
            _ => {},
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let owner = args.next().expect("owner");
    let name = args.next().expect("name");
    let folder = args.next().expect("folder path");

    let token = std::env::var("HF_TOKEN").expect("HF_TOKEN");
    let client = HFClient::builder().token(token).build()?;
    let repo = client.model(&owner, &name);

    let report = repo
        .upload_large_folder()
        .folder_path(std::path::PathBuf::from(folder))
        .progress(Arc::new(Logger))
        .send()
        .await?;

    println!(
        "uploaded {} files in {} commits ({} via lfs, {} bytes, {} deduped)",
        report.total_files,
        report.commits.len(),
        report.files_uploaded_lfs,
        report.bytes_uploaded,
        report.dedup_bytes_saved
    );
    Ok(())
}
