//! Commit operations: listing commits, refs, diffs, and branch/tag management.
//!
//! Requires HF_TOKEN environment variable.
//! Run: cargo run -p examples --example commits

use futures::StreamExt;
use hf_hub::{
    CreateRepoParams, DeleteRepoParams, HFClient, RepoCreateBranchParams, RepoCreateTagParams, RepoDeleteBranchParams,
    RepoDeleteTagParams, RepoGetCommitDiffParams, RepoGetRawDiffParams, RepoListCommitsParams, RepoListRefsParams,
};

#[tokio::main]
async fn main() -> hf_hub::Result<()> {
    let api = HFClient::new()?;

    // --- Read operations ---

    let repo = api.model("openai-community", "gpt2");
    let commits_stream = repo.list_commits(&RepoListCommitsParams::default())?;
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

    let refs = repo.list_refs(&RepoListRefsParams::default()).await?;
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
        let diff = repo
            .get_commit_diff(&RepoGetCommitDiffParams::builder().compare(&compare).build())
            .await?;
        println!("\nDiff ({compare}):");
        println!("  {} chars", diff.len());

        let raw_diff = repo
            .get_raw_diff(&RepoGetRawDiffParams::builder().compare(&compare).build())
            .await?;
        println!("Raw diff: {} chars", raw_diff.len());
    }

    // --- Write operations (creates real resources on the Hub) ---

    let user = api.whoami().await?;
    let unique = std::process::id();
    let repo = api.model(&user.username, format!("example-commits-{unique}"));

    api.create_repo(
        &CreateRepoParams::builder()
            .repo_id(repo.repo_path())
            .private(true)
            .exist_ok(true)
            .build(),
    )
    .await?;
    println!("\nCreated test repo: {}", repo.repo_path());

    repo.create_branch(&RepoCreateBranchParams::builder().branch("feature-branch").build())
        .await?;
    println!("Created branch: feature-branch");

    repo.delete_branch(&RepoDeleteBranchParams::builder().branch("feature-branch").build())
        .await?;
    println!("Deleted branch: feature-branch");

    repo.create_tag(&RepoCreateTagParams::builder().tag("v0.1.0").message("Initial release").build())
        .await?;
    println!("Created tag: v0.1.0");

    repo.delete_tag(&RepoDeleteTagParams::builder().tag("v0.1.0").build()).await?;
    println!("Deleted tag: v0.1.0");

    api.delete_repo(&DeleteRepoParams::builder().repo_id(repo.repo_path()).missing_ok(true).build())
        .await?;
    println!("Cleaned up test repo");

    Ok(())
}
