pub mod info;
pub mod list;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage models
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

/// Models subcommands
#[derive(Subcommand)]
pub enum ModelsCommand {
    /// Show detailed information about a model
    Info(info::Args),
    /// List models on the Hub
    #[command(alias = "ls")]
    List(list::Args),
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        ModelsCommand::Info(a) => info::execute(client, a).await,
        ModelsCommand::List(a) => list::execute(client, a).await,
    }
}
