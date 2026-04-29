//! Repository handle ergonomics: typed repo constructors, revision pinning,
//! tagged repo info, and converting a generic repo handle into HFSpace.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --example repo_handles

use hf_hub::HFClient;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let model = client.model("openai-community", "gpt2");
    println!("Model handle: owner={}, name={}", model.owner(), model.name());

    let info = client.model_info().repo_id(model.repo_path()).send().await?;
    println!("Model info: {} (sha: {:?})", info.id, info.sha);

    let config_exists = model.file_exists().filename("config.json").send().await?;
    println!("config.json exists on {}: {config_exists}", model.repo_path());

    let info = client.dataset_info().repo_id("rajpurkar/squad").send().await?;
    println!("Dataset info: {}", info.id);

    let space = client.space("huggingface", "transformers-benchmarks");
    let info = client.space_info().repo_id(space.repo_path()).send().await?;
    println!("Space info: {} (sdk: {:?})", info.id, info.sdk);

    Ok(())
}
