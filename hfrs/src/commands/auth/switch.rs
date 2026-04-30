use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use crate::output::CommandResult;
use crate::util::token;

/// Switch to a different stored account
#[derive(ClapArgs)]
pub struct Args {
    /// Name of the token to switch to
    #[arg(long)]
    pub token_name: String,
}

pub async fn execute(_client: &HFClient, args: Args) -> Result<CommandResult> {
    let name = args.token_name;
    token::switch_token(&name)?;
    Ok(CommandResult::Raw(format!("Switched to token '{name}'.")))
}
