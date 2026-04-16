//! Synchronous Space operations using HFClientSync and HFSpaceSync.
//!
//! Demonstrates runtime info, secrets, variables, and lifecycle management.
//!
//! Requires HF_TOKEN and the "blocking" + "spaces" features.
//! Run: cargo run -p hf-hub --features "blocking spaces" --example blocking_spaces

use hf_hub::{
    CreateRepoParams, DeleteRepoParams, HFClientSync, RepoType, SpaceSecretDeleteParams, SpaceSecretParams,
    SpaceVariableDeleteParams, SpaceVariableParams,
};

fn main() -> hf_hub::Result<()> {
    let api = HFClientSync::new()?;

    // --- Read operations ---

    let reference_space = api.space("huggingface", "transformers-benchmarks");
    let runtime = reference_space.runtime()?;
    println!("Space runtime: {runtime:?}");

    // --- Write operations (creates real resources on the Hub) ---

    let user = api.whoami()?;
    let unique = std::process::id();
    let space = api.space(&user.username, format!("blocking-example-space-{unique}"));

    api.create_repo(
        &CreateRepoParams::builder()
            .repo_id(space.repo_path())
            .repo_type(RepoType::Space)
            .private(true)
            .space_sdk("static")
            .exist_ok(true)
            .build(),
    )?;
    println!("\nCreated test space: {}", space.repo_path());

    space.add_secret(&SpaceSecretParams::builder().key("EXAMPLE_SECRET").value("secret-value").build())?;
    println!("Added secret: EXAMPLE_SECRET");

    space.delete_secret(&SpaceSecretDeleteParams::builder().key("EXAMPLE_SECRET").build())?;
    println!("Deleted secret: EXAMPLE_SECRET");

    space.add_variable(&SpaceVariableParams::builder().key("EXAMPLE_VAR").value("var-value").build())?;
    println!("Added variable: EXAMPLE_VAR");

    space.delete_variable(&SpaceVariableDeleteParams::builder().key("EXAMPLE_VAR").build())?;
    println!("Deleted variable: EXAMPLE_VAR");

    let paused = space.pause()?;
    println!("Paused space: {paused:?}");

    let restarted = space.restart()?;
    println!("Restarted space: {restarted:?}");

    api.delete_repo(
        &DeleteRepoParams::builder()
            .repo_id(space.repo_path())
            .repo_type(RepoType::Space)
            .missing_ok(true)
            .build(),
    )?;
    println!("Cleaned up test space");

    Ok(())
}
