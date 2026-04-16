pub mod cp;
pub mod create;
pub mod delete;
pub mod info;
pub mod list;
pub mod move_bucket;
pub mod remove;
pub mod sync;

use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;

use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: BucketsCommand,
}

#[derive(Subcommand)]
pub enum BucketsCommand {
    /// Copy files to/from a bucket
    Cp(cp::Args),
    /// Create a new bucket
    Create(create::Args),
    /// Delete a bucket
    Delete(delete::Args),
    /// Show detailed information about a bucket
    Info(info::Args),
    /// List buckets or files in a bucket
    #[command(alias = "ls")]
    List(list::Args),
    /// Move (rename) a bucket
    Move(move_bucket::Args),
    /// Remove files from a bucket
    #[command(alias = "rm")]
    Remove(remove::Args),
    /// Sync files between a local directory and a bucket
    Sync(sync::Args),
}

pub async fn execute(api: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    match args.command {
        BucketsCommand::Cp(a) => cp::execute(api, a, multi).await,
        BucketsCommand::Create(a) => create::execute(api, a).await,
        BucketsCommand::Delete(a) => delete::execute(api, a).await,
        BucketsCommand::Info(a) => info::execute(api, a).await,
        BucketsCommand::List(a) => list::execute(api, a).await,
        BucketsCommand::Move(a) => move_bucket::execute(api, a).await,
        BucketsCommand::Remove(a) => remove::execute(api, a).await,
        BucketsCommand::Sync(a) => sync::execute(api, a, multi).await,
    }
}

/// Parse a bucket ID from CLI input.
/// Accepts `namespace/name` or `hf://buckets/namespace/name`.
/// Returns `(namespace, name)` or an error.
pub(crate) fn parse_bucket_id(input: &str) -> Result<(String, String)> {
    let id = input.strip_prefix("hf://buckets/").unwrap_or(input);

    match id.split_once('/') {
        Some((ns, name)) if !ns.is_empty() && !name.is_empty() && !name.contains('/') => {
            Ok((ns.to_string(), name.to_string()))
        },
        _ => anyhow::bail!(
            "Invalid bucket ID '{input}'. Expected format: 'namespace/bucket_name' or 'hf://buckets/namespace/bucket_name'"
        ),
    }
}
