//! Download a file with the built-in indicatif progress bar.
//!
//! Demonstrates the batteries-included `IndicatifProgress` handler shipped behind
//! the `indicatif` feature — no need to hand-roll a `ProgressHandler`.
//!
//! Requires the `indicatif` feature.
//! Run: cargo run -p examples --features indicatif --example progress_indicatif

use hf_hub::HFClient;
use hf_hub::progress::IndicatifProgress;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let model = client.model("openai-community", "gpt2");

    let tmp_dir = tempfile::tempdir().expect("failed to create tempdir");

    let path = model
        .download_file()
        .filename("model.safetensors")
        .local_dir(tmp_dir.path().to_path_buf())
        .progress(IndicatifProgress::new())
        .send()
        .await?;

    println!("File saved to: {}", path.display());
    Ok(())
}
