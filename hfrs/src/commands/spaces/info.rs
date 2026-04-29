use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// Show detailed information about a Space
#[derive(ClapArgs)]
pub struct Args {
    /// Space ID (e.g. gradio/hello_world)
    pub space_id: String,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (owner, name) = crate::util::split_repo_id(&args.space_id);
    let repo = client.space(owner, name);
    let info = repo
        .info()
        .maybe_revision(args.revision)
        .send()
        .await?
        .into_space()
        .map_err(|_| anyhow::anyhow!("Expected space info"))?;
    let json_value = json!({
        "id": info.id,
        "author": info.author,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
        "sdk": info.sdk,
        "likes": info.likes,
        "tags": info.tags,
        "created_at": info.created_at,
        "last_modified": info.last_modified,
        "siblings": info.siblings,
        "card_data": info.card_data,
        "trending_score": info.trending_score,
        "host": info.host,
        "subdomain": info.subdomain,
        "runtime": info.runtime,
        "used_storage": info.used_storage,
    });
    let output = CommandOutput::single_item(json_value);
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
