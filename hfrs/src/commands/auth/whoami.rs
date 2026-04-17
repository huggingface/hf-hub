use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use serde_json::json;

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

/// Show the currently authenticated user
#[derive(ClapArgs)]
pub struct Args {
    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let user = client.whoami().await?;
    let orgs: Vec<String> = user
        .orgs
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|o| o.name.clone())
        .collect();
    let json_value = json!({
        "username": user.username,
        "fullname": user.fullname,
        "email": user.email,
        "orgs": orgs,
    });
    let output = CommandOutput::single_item(json_value);
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
