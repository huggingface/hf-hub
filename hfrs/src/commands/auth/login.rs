use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use crate::output::CommandResult;
use crate::util::token;

/// Log in with a Hugging Face token
#[derive(ClapArgs)]
pub struct Args {
    /// The token value to store
    #[arg(long)]
    pub token_value: Option<String>,

    /// Name to store the token under
    #[arg(long, default_value = "default")]
    pub token_name: String,
}

pub async fn execute(_client: &HFClient, args: Args) -> Result<CommandResult> {
    let value = match args.token_value {
        Some(v) => v,
        None => anyhow::bail!(
            "No token provided. Generate a token at https://huggingface.co/settings/tokens and pass it via --token-value."
        ),
    };
    let name = args.token_name;
    token::save_token(&name, &value)?;
    Ok(CommandResult::Raw(format!("Token saved as '{name}'.")))
}
