//! Synchronous repository handle ergonomics using HFClientSync.
//!
//! Demonstrates typed repo constructors, revision pinning, tagged repo info,
//! and converting a generic repo handle into HFSpaceSync.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --features blocking --example blocking_repo_handles

use hf_hub::{HFClientSync, HFSpaceSync, RepoType};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    let model = client.model("openai-community", "gpt2");
    println!("Model handle: owner={}, name={}", model.owner(), model.name());

    let info = model.info().send()?;
    println!("Model info: {} (sha: {:?})", info.id, info.sha);

    let config_exists = model.file_exists().filename("config.json").send()?;
    println!("config.json exists on {}: {config_exists}", model.repo_path());

    let dataset = client.dataset("rajpurkar", "squad");
    let info = dataset.info().send()?;
    println!("Dataset info: {}", info.id);

    let generic_space = client.repo(RepoType::Space, "huggingface-projects", "diffusers-gallery");
    let space = HFSpaceSync::try_from(generic_space)?;

    let info = space.info().send()?;
    println!("Space info: {} (sdk: {:?})", info.id, info.sdk);

    let direct_space = client.space("huggingface-projects", "diffusers-gallery");
    println!("Direct space handle matches converted handle: {}", direct_space.repo_path() == space.repo_path());

    Ok(())
}
