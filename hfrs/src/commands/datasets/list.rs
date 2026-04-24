use anyhow::Result;
use clap::Args as ClapArgs;
use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repository::ListDatasetsParams;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// List datasets on the Hub
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

    /// Print only dataset IDs
    #[arg(long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let filter = if args.filter.is_empty() {
        None
    } else {
        Some(args.filter.join(","))
    };

    let params = ListDatasetsParams {
        search: args.search,
        author: args.author,
        filter,
        sort: args.sort,
        full: None,
        limit: Some(args.limit),
    };

    let stream = client.list_datasets(params)?;
    futures::pin_mut!(stream);

    let mut datasets = Vec::new();
    while let Some(item) = stream.next().await {
        datasets.push(item?);
    }

    if datasets.is_empty() && matches!(args.format, OutputFormat::Table) {
        return Ok(CommandResult::Raw("No datasets found.".to_string()));
    }

    let headers = vec![
        "ID".to_string(),
        "Author".to_string(),
        "Downloads".to_string(),
        "Likes".to_string(),
    ];

    let rows = datasets
        .iter()
        .map(|d| {
            vec![
                d.id.clone(),
                d.author.clone().unwrap_or_default(),
                d.downloads.map(|v| v.to_string()).unwrap_or_default(),
                d.likes.map(|v| v.to_string()).unwrap_or_default(),
            ]
        })
        .collect();

    let quiet_values = datasets.iter().map(|d| d.id.clone()).collect();

    let json_value: serde_json::Value = datasets
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                "author": d.author,
                "downloads": d.downloads,
                "likes": d.likes,
                "tags": d.tags,
                "trending_score": d.trending_score,
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
