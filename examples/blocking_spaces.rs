//! Synchronous Space operations using HFClientSync.
//!
//! Demonstrates runtime info, secrets, variables, and lifecycle management on a
//! blocking `HFRepositorySync<RepoTypeSpace>`.
//!
//! Requires HF_TOKEN and the "blocking" feature.
//! Run: cargo run -p examples --features blocking --example blocking_spaces

use hf_hub::{HFClientSync, RepoTypeSpace};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    // --- Read operations ---

    let reference_space = client.space("huggingface", "transformers-benchmarks");
    let runtime = reference_space.runtime().send()?;
    println!("Space runtime: {runtime:?}");

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send()?;
    let unique = std::process::id();
    let space = client.space(&user.username, format!("blocking-example-space-{unique}"));

    client
        .create_repo()
        .repo_type(RepoTypeSpace)
        .repo_id(space.repo_path())
        .private(true)
        .space_sdk("static")
        .exist_ok(true)
        .send()?;
    println!("\nCreated test space: {}", space.repo_path());

    space.add_secret().key("EXAMPLE_SECRET").value("secret-value").send()?;
    println!("Added secret: EXAMPLE_SECRET");

    space.delete_secret().key("EXAMPLE_SECRET").send()?;
    println!("Deleted secret: EXAMPLE_SECRET");

    space.add_variable().key("EXAMPLE_VAR").value("var-value").send()?;
    println!("Added variable: EXAMPLE_VAR");

    space.delete_variable().key("EXAMPLE_VAR").send()?;
    println!("Deleted variable: EXAMPLE_VAR");

    let paused = space.pause().send()?;
    println!("Paused space: {paused:?}");

    let restarted = space.restart().send()?;
    println!("Restarted space: {restarted:?}");

    client
        .delete_repo()
        .repo_type(RepoTypeSpace)
        .repo_id(space.repo_path())
        .missing_ok(true)
        .send()?;
    println!("Cleaned up test space");

    Ok(())
}
