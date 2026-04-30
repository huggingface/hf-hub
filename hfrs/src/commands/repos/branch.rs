use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Manage repository branches
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: BranchCommand,
}

/// Branch subcommands
#[derive(Subcommand)]
pub enum BranchCommand {
    /// Create a new branch
    Create(BranchCreateArgs),
    /// Delete a branch
    Delete(BranchDeleteArgs),
}

/// Create a new branch
#[derive(ClapArgs)]
pub struct BranchCreateArgs {
    /// Repository ID
    pub repo_id: String,

    /// Branch name to create
    pub branch: String,

    /// Starting revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,
}

/// Delete a branch
#[derive(ClapArgs)]
pub struct BranchDeleteArgs {
    /// Repository ID
    pub repo_id: String,

    /// Branch name to delete
    pub branch: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        BranchCommand::Create(a) => create(client, a).await,
        BranchCommand::Delete(a) => delete(client, a).await,
    }
}

async fn create(client: &HFClient, args: BranchCreateArgs) -> Result<CommandResult> {
    crate::with_typed_repo!(client, &args.repo_id, args.r#type, |repo| {
        repo.create_branch()
            .branch(args.branch)
            .maybe_revision(args.revision)
            .send()
            .await?
    });
    Ok(CommandResult::Raw("Branch created.".to_string()))
}

async fn delete(client: &HFClient, args: BranchDeleteArgs) -> Result<CommandResult> {
    crate::with_typed_repo!(client, &args.repo_id, args.r#type, |repo| {
        repo.delete_branch().branch(args.branch).send().await?
    });
    Ok(CommandResult::Silent)
}
