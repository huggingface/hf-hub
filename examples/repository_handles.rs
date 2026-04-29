//! Repository handle ergonomics: typed repo constructors, revision pinning,
//! tagged repo info, and converting a generic repo handle into HFSpace.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --example repo_handles

use hf_hub::{HFClient, HFSpace, RepoType};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let model = client.model("openai-community", "gpt2");
    println!("Model handle: owner={}, name={}", model.owner(), model.name());

    let info = model.info().send().await?;
    println!("Model info: {} (sha: {:?})", info.id, info.sha);

    let config_exists = model.file_exists().filename("config.json").send().await?;
    println!("config.json exists on {}: {config_exists}", model.repo_path());

    let dataset = client.dataset("rajpurkar", "squad");
    let info = dataset.info().send().await?;
    println!("Dataset info: {}", info.id);

    let generic_space = client.repo(RepoType::Space, "huggingface", "transformers-benchmarks");
    let space = HFSpace::try_from(generic_space)?;

    let info = space.info().send().await?;
    println!("Space info: {} (sdk: {:?})", info.id, info.sdk);

    let direct_space = client.space("huggingface", "transformers-benchmarks");
    println!("Direct space handle matches converted handle: {}", direct_space.repo_path() == space.repo_path());

    Ok(())
}
