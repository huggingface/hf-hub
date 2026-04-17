use std::io::{self, Write};

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use super::parse_bucket_id;
use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    /// Bucket ID (namespace/name or hf://buckets/namespace/name)
    pub bucket_id: String,

    /// Skip confirmation prompt
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Do not fail if the bucket does not exist
    #[arg(long)]
    pub missing_ok: bool,

    /// Print only the bucket ID
    #[arg(short, long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (namespace, name) = parse_bucket_id(&args.bucket_id)?;
    let bucket_id = format!("{}/{}", namespace, name);

    if !args.yes {
        eprint!("Are you sure you want to delete bucket '{bucket_id}'? [y/N] ");
        io::stderr().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            return Ok(CommandResult::Raw("Aborted.".to_string()));
        }
    }

    client.delete_bucket(&bucket_id, args.missing_ok).await?;

    if args.quiet {
        Ok(CommandResult::Raw(bucket_id))
    } else {
        Ok(CommandResult::Raw(format!("Bucket deleted: {bucket_id}")))
    }
}
