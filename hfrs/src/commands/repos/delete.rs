use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::repo::DeleteRepoParams;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Delete a repository
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Do not fail if the repository does not exist
    #[arg(long)]
    pub missing_ok: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let params = DeleteRepoParams {
        repo_id: args.repo_id,
        repo_type: Some(repo_type),
        missing_ok: args.missing_ok,
    };
    client.delete_repo(&params).await?;
    Ok(CommandResult::Silent)
}
