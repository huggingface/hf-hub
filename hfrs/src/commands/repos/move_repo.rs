use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::{HFClient, RepoTypeDataset, RepoTypeModel, RepoTypeSpace};

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
    let result = match args.r#type {
        RepoTypeArg::Model => {
            client
                .move_repository()
                .repo_type(RepoTypeModel)
                .from_id(args.from_id)
                .to_id(args.to_id)
                .send()
                .await?
        },
        RepoTypeArg::Dataset => {
            client
                .move_repository()
                .repo_type(RepoTypeDataset)
                .from_id(args.from_id)
                .to_id(args.to_id)
                .send()
                .await?
        },
        RepoTypeArg::Space => {
            client
                .move_repository()
                .repo_type(RepoTypeSpace)
                .from_id(args.from_id)
                .to_id(args.to_id)
                .send()
                .await?
        },
    };
    Ok(CommandResult::Raw(result.url))
}
