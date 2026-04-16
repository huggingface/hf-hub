use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::{HFClient, RepoCreateBranchParams, RepoDeleteBranchParams};

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

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        BranchCommand::Create(a) => create(api, a).await,
        BranchCommand::Delete(a) => delete(api, a).await,
    }
}

async fn create(api: &HFClient, args: BranchCreateArgs) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(api, &args.repo_id, repo_type);
    let params = RepoCreateBranchParams {
        branch: args.branch,
        revision: args.revision,
    };
    repo.create_branch(&params).await?;
    Ok(CommandResult::Raw("Branch created.".to_string()))
}

async fn delete(api: &HFClient, args: BranchDeleteArgs) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(api, &args.repo_id, repo_type);
    let params = RepoDeleteBranchParams { branch: args.branch };
    repo.delete_branch(&params).await?;
    Ok(CommandResult::Silent)
}
