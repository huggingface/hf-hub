pub mod list;
pub mod login;
pub mod logout;
pub mod switch;
pub mod whoami;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage authentication credentials
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: AuthCommand,
}

/// Authentication subcommands
#[derive(Subcommand)]
pub enum AuthCommand {
    /// Log in with a Hugging Face token
    Login(login::Args),
    /// Log out and remove stored credentials
    Logout(logout::Args),
    /// Switch to a different stored account
    Switch(switch::Args),
    /// List stored accounts
    List(list::Args),
    /// Show the currently authenticated user
    Whoami(whoami::Args),
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        AuthCommand::Login(a) => login::execute(api, a).await,
        AuthCommand::Logout(a) => logout::execute(api, a).await,
        AuthCommand::Switch(a) => switch::execute(api, a).await,
        AuthCommand::List(a) => list::execute(api, a).await,
        AuthCommand::Whoami(a) => whoami::execute(api, a).await,
    }
}
