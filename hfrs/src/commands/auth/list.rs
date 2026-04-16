use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use crate::output::CommandResult;
use crate::util::token;

/// List stored accounts
#[derive(ClapArgs)]
pub struct Args {}

pub async fn execute(_client: &HFClient, _args: Args) -> Result<CommandResult> {
    let entries = token::list_tokens();
    if entries.is_empty() {
        return Ok(CommandResult::Raw("No stored tokens. Use `hfrs auth login` to add one.".to_string()));
    }
    let lines: Vec<String> = entries
        .iter()
        .map(|e| {
            let active_marker = if e.is_active { " (active)" } else { "" };
            format!("  {} — {}{}", e.name, e.token_masked, active_marker)
        })
        .collect();
    Ok(CommandResult::Raw(lines.join("\n")))
}
