//! Synchronous repository handle ergonomics using HFClientSync.
//!
//! Demonstrates typed repo constructors, revision pinning, tagged repo info,
//! and converting a generic repo handle into HFSpaceSync.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --features blocking --example blocking_repo_handles

use hf_hub::types::{RepoFileExistsParams, RepoInfo, RepoInfoParams, RepoType};
use hf_hub::{HFClientSync, HFSpaceSync};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    let model = client.model("openai-community", "gpt2");
    println!("Model handle: owner={}, name={}", model.owner(), model.name());

    match model.info(&RepoInfoParams::default())? {
        RepoInfo::Model(info) => println!("Model info: {} (sha: {:?})", info.id, info.sha),
        _ => unreachable!(),
    }

    let config_exists = model.file_exists(&RepoFileExistsParams::builder().filename("config.json").build())?;
    println!("config.json exists on {}: {config_exists}", model.repo_path());

    let dataset = client.repo(RepoType::Dataset, "rajpurkar", "squad");
    match dataset.info(&RepoInfoParams::default())? {
        RepoInfo::Dataset(info) => println!("Dataset info: {}", info.id),
        _ => unreachable!(),
    }

    let generic_space = client.repo(RepoType::Space, "huggingface", "transformers-benchmarks");
    let space = HFSpaceSync::try_from(generic_space)?;

    match space.info(&RepoInfoParams::default())? {
        RepoInfo::Space(info) => println!("Space info: {} (sdk: {:?})", info.id, info.sdk),
        _ => unreachable!(),
    }

    let direct_space = client.space("huggingface", "transformers-benchmarks");
    println!("Direct space handle matches converted handle: {}", direct_space.repo_path() == space.repo_path());

    Ok(())
}
