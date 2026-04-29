//! Synchronous repository handle ergonomics using HFClientSync.
//!
//! Demonstrates typed repo constructors, revision pinning, tagged repo info,
//! and converting a generic repo handle into HFSpaceSync.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --features blocking --example blocking_repo_handles

use hf_hub::HFClientSync;

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    let model = client.model("openai-community", "gpt2");
    println!("Model handle: owner={}, name={}", model.owner(), model.name());

    let info = client.model_info().repo_id(model.repo_path()).send()?;
    println!("Model info: {} (sha: {:?})", info.id, info.sha);

    let config_exists = model.file_exists().filename("config.json").send()?;
    println!("config.json exists on {}: {config_exists}", model.repo_path());

    let info = client.dataset_info().repo_id("rajpurkar/squad").send()?;
    println!("Dataset info: {}", info.id);

    let space = client.space("huggingface", "transformers-benchmarks");
    let info = client.space_info().repo_id(space.repo_path()).send()?;
    println!("Space info: {} (sdk: {:?})", info.id, info.sdk);

    Ok(())
}
