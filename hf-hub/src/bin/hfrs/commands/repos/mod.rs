pub mod branch;
pub mod create;
pub mod delete;
pub mod delete_files;
pub mod move_repo;
pub mod settings;
pub mod tag;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

/// Manage repositories
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: ReposCommand,
}

/// Repos subcommands
#[derive(Subcommand)]
pub enum ReposCommand {
    /// Create a new repository
    Create(create::Args),
    /// Delete a repository
    Delete(delete::Args),
    /// Move (rename) a repository
    Move(move_repo::Args),
    /// Update repository settings
    Settings(settings::Args),
    /// Delete files from a repository
    DeleteFiles(delete_files::Args),
    /// Manage branches
    Branch(branch::Args),
    /// Manage tags
    Tag(tag::Args),
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        ReposCommand::Create(a) => create::execute(api, a).await,
        ReposCommand::Delete(a) => delete::execute(api, a).await,
        ReposCommand::Move(a) => move_repo::execute(api, a).await,
        ReposCommand::Settings(a) => settings::execute(api, a).await,
        ReposCommand::DeleteFiles(a) => delete_files::execute(api, a).await,
        ReposCommand::Branch(a) => branch::execute(api, a).await,
        ReposCommand::Tag(a) => tag::execute(api, a).await,
    }
}
