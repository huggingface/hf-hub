use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Create a new repository
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Make the repository private
    #[arg(long)]
    pub private: bool,

    /// Do not fail if the repository already exists
    #[arg(long)]
    pub exist_ok: bool,

    /// Space SDK (only for Space repositories)
    #[arg(long)]
    pub space_sdk: Option<String>,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let result = client
        .create_repo()
        .repo_id(args.repo_id)
        .repo_type(repo_type)
        .maybe_private(if args.private { Some(true) } else { None })
        .exist_ok(args.exist_ok)
        .maybe_space_sdk(args.space_sdk)
        .send()
        .await?;
    Ok(CommandResult::Raw(result.url))
}
