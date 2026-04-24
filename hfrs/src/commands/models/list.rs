use anyhow::Result;
use clap::Args as ClapArgs;
use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repo::ListModelsParams;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// List models on the Hub
#[derive(ClapArgs)]
pub struct Args {
    /// Search query
    #[arg(long)]
    pub search: Option<String>,

    /// Filter by author/organization
    #[arg(long)]
    pub author: Option<String>,

    /// Filter tags (can be specified multiple times)
    #[arg(long)]
    pub filter: Vec<String>,

    /// Sort field
    #[arg(long)]
    pub sort: Option<String>,

    /// Maximum number of results
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: OutputFormat,

    /// Print only model IDs
    #[arg(long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let filter = if args.filter.is_empty() {
        None
    } else {
        Some(args.filter.join(","))
    };

    let params = ListModelsParams {
        search: args.search,
        author: args.author,
        filter,
        sort: args.sort,
        pipeline_tag: None,
        full: None,
        card_data: None,
        fetch_config: None,
        limit: Some(args.limit),
    };

    let stream = client.list_models(&params)?;
    futures::pin_mut!(stream);

    let mut models = Vec::new();
    while let Some(item) = stream.next().await {
        models.push(item?);
    }

    if models.is_empty() && matches!(args.format, OutputFormat::Table) {
        return Ok(CommandResult::Raw("No models found.".to_string()));
    }

    let headers = vec![
        "ID".to_string(),
        "Author".to_string(),
        "Downloads".to_string(),
        "Likes".to_string(),
        "Pipeline".to_string(),
    ];

    let rows = models
        .iter()
        .map(|m| {
            vec![
                m.id.clone(),
                m.author
                    .clone()
                    .unwrap_or_else(|| m.id.split('/').next().unwrap_or_default().to_string()),
                m.downloads.map(|d| d.to_string()).unwrap_or_default(),
                m.likes.map(|l| l.to_string()).unwrap_or_default(),
                m.pipeline_tag.clone().unwrap_or_default(),
            ]
        })
        .collect();

    let quiet_values = models.iter().map(|m| m.id.clone()).collect();

    let json_value: serde_json::Value = models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "author": m.author.clone().unwrap_or_else(|| m.id.split('/').next().unwrap_or_default().to_string()),
                "downloads": m.downloads,
                "likes": m.likes,
                "pipeline_tag": m.pipeline_tag,
                "library_name": m.library_name,
                "tags": m.tags,
                "trending_score": m.trending_score,
            })
        })
        .collect::<Vec<_>>()
        .into();

    let output = CommandOutput {
        headers,
        rows,
        json_value,
        quiet_values,
    };
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: args.quiet,
    })
}
