//! Build a runtime-tagged repository handle from a `&str` kind via [`RepoTypeAny`].
//!
//! When the repo kind comes from config, a CLI flag, or some upstream enum, the
//! `RepoTypeAny` enum lets you skip the per-kind `match` and carry the choice as a value.
//! `HFRepository<RepoTypeAny>` is a sized, concrete type — no `Box`, no `dyn`. Trait
//! methods dispatch on the enum variant at runtime.
//!
//! `info()` on the runtime-tagged handle returns a [`RepoInfo`] enum: its accessor
//! methods read fields shared across repo kinds without matching, and matching on the
//! variant (or `as_model()` / `as_space()` / ...) yields the same per-kind struct the
//! typed handles (`client.model(..)` / `client.dataset(..)` / etc.) return.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --example repo_type_any

use hf_hub::repository::RepoInfo;
use hf_hub::{HFClient, RepoType, RepoTypeAny};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let inputs = [
        ("model", "openai-community", "gpt2"),
        ("datasets", "rajpurkar", "squad"),
        ("space", "huggingface", "transformers-benchmarks"),
    ];

    for (kind_str, owner, name) in inputs {
        let kind: RepoTypeAny = kind_str.parse()?;
        let repo = client.repository(kind, owner, name);

        println!(
            "{}: {} (api segment={:?})",
            repo.repo_type().singular(),
            repo.repo_path(),
            repo.repo_type().plural()
        );

        let info = repo.info().send().await?;
        println!("  author={:?} likes={:?} last_modified={:?}", info.author(), info.likes(), info.last_modified());

        match &info {
            RepoInfo::Model(model) => println!("  pipeline_tag={:?}", model.pipeline_tag),
            RepoInfo::Dataset(dataset) => println!("  citation set: {}", dataset.citation.is_some()),
            RepoInfo::Space(space) => println!("  sdk={:?}", space.sdk),
            RepoInfo::Kernel(kernel) => println!("  trusted_publisher={:?}", kernel.trusted_publisher),
            _ => {},
        }
    }

    Ok(())
}
