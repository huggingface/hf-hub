use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use crate::output::CommandResult;
use crate::util::token;

/// Log out and remove stored credentials
#[derive(ClapArgs)]
pub struct Args {
    /// Name of the token to remove
    #[arg(long)]
    pub token_name: Option<String>,
}

pub async fn execute(_client: &HFClient, args: Args) -> Result<CommandResult> {
    let name = args.token_name.unwrap_or_else(|| "default".to_string());
    token::delete_token(&name)?;
    Ok(CommandResult::Raw(format!("Token '{name}' removed.")))
}
