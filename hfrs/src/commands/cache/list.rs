use std::time::SystemTime;

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;
    const KB: u64 = 1_024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn format_system_time(t: SystemTime) -> String {
    let secs = t.duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if secs == 0 {
        return String::new();
    }
    format!("{secs}")
}

/// List cached repositories and files
#[derive(ClapArgs)]
pub struct Args {
    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let cache_info = client.scan_cache().await?;

    if cache_info.repos.is_empty() && matches!(args.format, OutputFormat::Table) {
        return Ok(CommandResult::Raw("No cached repos found.".to_string()));
    }

    let headers = vec![
        "Repo".to_string(),
        "Type".to_string(),
        "Size".to_string(),
        "Revisions".to_string(),
        "Last Accessed".to_string(),
    ];

    let rows = cache_info
        .repos
        .iter()
        .map(|r| {
            vec![
                r.repo_id.clone(),
                r.repo_type.to_string(),
                format_bytes(r.size_on_disk),
                r.revisions.len().to_string(),
                format_system_time(r.last_accessed),
            ]
        })
        .collect();

    let json_value: serde_json::Value = cache_info
        .repos
        .iter()
        .map(|r| {
            json!({
                "repo_id": r.repo_id,
                "repo_type": r.repo_type.to_string(),
                "size_on_disk": r.size_on_disk,
                "revision_count": r.revisions.len(),
                "last_accessed": format_system_time(r.last_accessed),
            })
        })
        .collect::<Vec<_>>()
        .into();

    let output = CommandOutput {
        headers,
        rows,
        json_value,
        quiet_values: vec![],
    };
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
