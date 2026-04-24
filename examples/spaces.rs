//! Space operations: runtime info, secrets, variables, and lifecycle management.
//!
//! Requires HF_TOKEN and the "spaces" feature.
//! Run: cargo run -p examples --example spaces

use hf_hub::repository::{CreateRepoParams, DeleteRepoParams};
use hf_hub::spaces::{SpaceSecretDeleteParams, SpaceSecretParams, SpaceVariableDeleteParams, SpaceVariableParams};
use hf_hub::{HFClient, RepoType};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // --- Read operations ---

    let reference_space = client.space("huggingface", "transformers-benchmarks");
    let runtime = reference_space.runtime().await?;
    println!("Space runtime: {runtime:?}");

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().await?;
    let unique = std::process::id();
    let space = client.space(&user.username, format!("example-space-{unique}"));

    client
        .create_repo(
            CreateRepoParams::builder()
                .repo_id(space.repo_path())
                .repo_type(RepoType::Space)
                .private(true)
                .space_sdk("static")
                .exist_ok(true)
                .build(),
        )
        .await?;
    println!("\nCreated test space: {}", space.repo_path());

    space
        .add_secret(SpaceSecretParams::builder().key("EXAMPLE_SECRET").value("secret-value").build())
        .await?;
    println!("Added secret: EXAMPLE_SECRET");

    space
        .delete_secret(SpaceSecretDeleteParams::builder().key("EXAMPLE_SECRET").build())
        .await?;
    println!("Deleted secret: EXAMPLE_SECRET");

    space
        .add_variable(SpaceVariableParams::builder().key("EXAMPLE_VAR").value("var-value").build())
        .await?;
    println!("Added variable: EXAMPLE_VAR");

    space
        .delete_variable(SpaceVariableDeleteParams::builder().key("EXAMPLE_VAR").build())
        .await?;
    println!("Deleted variable: EXAMPLE_VAR");

    let paused = space.pause().await?;
    println!("Paused space: {paused:?}");

    let restarted = space.restart().await?;
    println!("Restarted space: {restarted:?}");

    client
        .delete_repo(
            DeleteRepoParams::builder()
                .repo_id(space.repo_path())
                .repo_type(RepoType::Space)
                .missing_ok(true)
                .build(),
        )
        .await?;
    println!("Cleaned up test space");

    Ok(())
}
