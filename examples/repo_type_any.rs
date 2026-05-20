//! Build a runtime-tagged repository handle from a `&str` kind via [`RepoTypeAny`].
//!
//! When the repo kind comes from config, a CLI flag, or some upstream enum, the
//! `RepoTypeAny` enum lets you skip the per-kind `match` and carry the choice as a value.
//! `HFRepository<RepoTypeAny>` is a sized, concrete type — no `Box`, no `dyn`. Trait
//! methods dispatch on the enum variant at runtime.
//!
//! Kind-specific methods like the per-kind `info()` are not available on
//! `HFRepository<RepoTypeAny>`; use the typed handles (`client.model(..)` /
//! `client.dataset(..)` / etc.) when you need a kind-specific response shape.
//!
//! Read-only operations require no auth.
//! Run: cargo run -p examples --example repo_type_any

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

        let exists = repo.exists().send().await?;
        println!("  exists: {exists}");
    }

    Ok(())
}
