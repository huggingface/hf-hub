use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::repo::MoveRepoParams;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Move (rename) a repository
#[derive(ClapArgs)]
pub struct Args {
    /// Source repository ID
    pub from_id: String,

    /// Destination repository ID
    pub to_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let params = MoveRepoParams {
        from_id: args.from_id,
        to_id: args.to_id,
        repo_type: Some(repo_type),
    };
    let result = client.move_repo(&params).await?;
    Ok(CommandResult::Raw(result.url))
}
