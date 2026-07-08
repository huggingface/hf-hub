use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::{HFClient, RepoTypeAny};

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Delete a repository
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g., username/my-model)
    pub repo_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Do not fail if the repository does not exist
    #[arg(long)]
    pub missing_ok: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    client
        .delete_repository()
        .repo_type(RepoTypeAny::from(args.r#type))
        .repo_id(&args.repo_id)
        .missing_ok(args.missing_ok)
        .send()
        .await?;
    Ok(CommandResult::Silent)
}
