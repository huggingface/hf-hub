pub mod info;
pub mod list;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage Spaces
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: SpacesCommand,
}

/// Spaces subcommands
#[derive(Subcommand)]
pub enum SpacesCommand {
    /// Show detailed information about a Space
    Info(info::Args),
    /// List Spaces on the Hub
    #[command(alias = "ls")]
    List(list::Args),
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        SpacesCommand::Info(a) => info::execute(api, a).await,
        SpacesCommand::List(a) => list::execute(api, a).await,
    }
}
