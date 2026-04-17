pub mod list;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage the local model cache
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: CacheCommand,
}

/// Cache subcommands
#[derive(Subcommand)]
pub enum CacheCommand {
    /// List cached repositories and files
    List(list::Args),
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        CacheCommand::List(a) => list::execute(client, a).await,
    }
}
