//! Space operations: runtime info, secrets, variables, and lifecycle management.
//!
//! Requires HF_TOKEN.
//! Run: cargo run -p examples --example spaces

use hf_hub::{HFClient, RepoTypeSpace};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // --- Read operations ---

    let reference_space = client.space("huggingface", "transformers-benchmarks");
    let runtime = reference_space.runtime().send().await?;
    println!("Space runtime: {runtime:?}");

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send().await?;
    let unique = std::process::id();
    let space = client.space(&user.username, format!("example-space-{unique}"));

    client
        .create_repo()
        .repo_type(RepoTypeSpace)
        .repo_id(space.repo_path())
        .private(true)
        .space_sdk("static")
        .exist_ok(true)
        .send()
        .await?;
    println!("\nCreated test space: {}", space.repo_path());

    space.add_secret().key("EXAMPLE_SECRET").value("secret-value").send().await?;
    println!("Added secret: EXAMPLE_SECRET");

    space.delete_secret().key("EXAMPLE_SECRET").send().await?;
    println!("Deleted secret: EXAMPLE_SECRET");

    space.add_variable().key("EXAMPLE_VAR").value("var-value").send().await?;
    println!("Added variable: EXAMPLE_VAR");

    space.delete_variable().key("EXAMPLE_VAR").send().await?;
    println!("Deleted variable: EXAMPLE_VAR");

    let paused = space.pause().send().await?;
    println!("Paused space: {paused:?}");

    let restarted = space.restart().send().await?;
    println!("Restarted space: {restarted:?}");

    client
        .delete_repo()
        .repo_type(RepoTypeSpace)
        .repo_id(space.repo_path())
        .missing_ok(true)
        .send()
        .await?;
    println!("Cleaned up test space");

    Ok(())
}
