pub mod info;
pub mod list;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage datasets
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: DatasetsCommand,
}

/// Datasets subcommands
#[derive(Subcommand)]
pub enum DatasetsCommand {
    /// Show detailed information about a dataset
    Info(info::Args),
    /// List datasets on the Hub
    #[command(alias = "ls")]
    List(list::Args),
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        DatasetsCommand::Info(a) => info::execute(api, a).await,
        DatasetsCommand::List(a) => list::execute(api, a).await,
    }
}
