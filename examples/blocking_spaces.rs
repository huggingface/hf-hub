//! Synchronous Space operations using HFClientSync and HFSpaceSync.
//!
//! Demonstrates runtime info, secrets, variables, and lifecycle management.
//!
//! Requires HF_TOKEN and the "blocking" + "spaces" features.
//! Run: cargo run -p examples --features blocking --example blocking_spaces

use hf_hub::{HFClientSync, RepoType};

fn main() -> hf_hub::HFResult<()> {
    let client = HFClientSync::new()?;

    // --- Read operations ---

    let reference_space = client.space("huggingface-projects", "diffusers-gallery");
    let runtime = reference_space.runtime().send()?;
    println!("Space runtime: {runtime:?}");

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send()?;
    let unique = std::process::id();
    let space = client.space(&user.username, format!("blocking-example-space-{unique}"));

    client
        .create_repo()
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
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
        .repo_id(space.repo_path())
        .repo_type(RepoType::Space)
        .missing_ok(true)
        .send()?;
    println!("Cleaned up test space");

    Ok(())
}
