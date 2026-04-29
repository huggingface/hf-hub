use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// Show detailed information about a model
#[derive(ClapArgs)]
pub struct Args {
    /// Model ID (e.g. gpt2 or openai-community/gpt2)
    pub model_id: String,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (owner, name) = crate::util::split_repo_id(&args.model_id);
    let repo = client.model(owner, name);
    let info = repo.info().maybe_revision(args.revision).send().await?.into_model_info()?;
    let json_value = json!({
        "id": info.id,
        "author": info.author,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
        "downloads": info.downloads,
        "likes": info.likes,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "tags": info.tags,
        "created_at": info.created_at,
        "last_modified": info.last_modified,
        "siblings": info.siblings,
        "card_data": info.card_data,
        "config": info.config,
        "trending_score": info.trending_score,
        "gguf": info.gguf,
        "spaces": info.spaces,
        "used_storage": info.used_storage,
        "widget_data": info.widget_data,
    });
    let output = CommandOutput::single_item(json_value);
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
