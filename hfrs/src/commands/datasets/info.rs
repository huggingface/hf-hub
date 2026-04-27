use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::repository::RepoInfo;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// Show detailed information about a dataset
#[derive(ClapArgs)]
pub struct Args {
    /// Dataset ID (e.g. squad or rajpurkar/squad)
    pub dataset_id: String,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (owner, name) = crate::util::split_repo_id(&args.dataset_id);
    let repo = client.dataset(owner, name);
    let repo_info = repo.info().maybe_revision(args.revision).send().await?;
    let info = match repo_info {
        RepoInfo::Dataset(d) => d,
        _ => anyhow::bail!("Expected dataset info"),
    };
    let json_value = json!({
        "id": info.id,
        "author": info.author,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags,
        "created_at": info.created_at,
        "last_modified": info.last_modified,
        "siblings": info.siblings,
        "card_data": info.card_data,
        "trending_score": info.trending_score,
        "description": info.description,
        "used_storage": info.used_storage,
    });
    let output = CommandOutput::single_item(json_value);
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
