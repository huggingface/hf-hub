//! Commit operations: listing commits, refs, diffs, and branch/tag management.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example commits

use futures::StreamExt;
use hf_hub::{HFClient, RepoTypeModel};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // --- Read operations ---

    let repo = client.model("openai-community", "gpt2");
    let commits_stream = repo.list_commits().send()?;
    futures::pin_mut!(commits_stream);
    println!("Recent commits in gpt2:");
    let mut first_two_ids: Vec<String> = Vec::new();
    let mut count = 0;
    while let Some(Ok(commit)) = commits_stream.next().await {
        println!("  {} - {}", &commit.id[..8], commit.title);
        if first_two_ids.len() < 2 {
            first_two_ids.push(commit.id.clone());
        }
        count += 1;
        if count >= 5 {
            break;
        }
    }

    let refs = repo.list_refs().send().await?;
    println!("\nBranches:");
    for b in &refs.branches {
        println!("  {} -> {}", b.name, &b.target_commit[..8]);
    }
    println!("Tags:");
    for t in &refs.tags {
        println!("  {} -> {}", t.name, &t.target_commit[..8]);
    }

    if first_two_ids.len() == 2 {
        let compare = format!("{}..{}", first_two_ids[1], first_two_ids[0]);
        let diff = repo.get_commit_diff().compare(&compare).send().await?;
        println!("\nDiff ({compare}):");
        println!("  {} chars", diff.len());

        let raw_diff = repo.get_raw_diff().compare(&compare).send().await?;
        println!("Raw diff: {} chars", raw_diff.len());
    }

    // --- Write operations (creates real resources on the Hub) ---

    let user = client.whoami().send().await?;
    let unique = std::process::id();
    let repo = client.model(&user.username, format!("example-commits-{unique}"));

    client
        .create_repository()
        .repo_type(RepoTypeModel)
        .repo_id(repo.repo_path())
        .private(true)
        .exist_ok(true)
        .send()
        .await?;
    println!("\nCreated test repo: {}", repo.repo_path());

    repo.create_branch().branch("feature-branch").send().await?;
    println!("Created branch: feature-branch");

    repo.delete_branch().branch("feature-branch").send().await?;
    println!("Deleted branch: feature-branch");

    repo.create_tag().tag("v0.1.0").message("Initial release").send().await?;
    println!("Created tag: v0.1.0");

    repo.delete_tag().tag("v0.1.0").send().await?;
    println!("Deleted tag: v0.1.0");

    client
        .delete_repository()
        .repo_type(RepoTypeModel)
        .repo_id(repo.repo_path())
        .missing_ok(true)
        .send()
        .await?;
    println!("Cleaned up test repo");

    Ok(())
}
