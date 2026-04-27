use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use super::parse_bucket_id;
use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    /// Bucket ID (namespace/name or hf://buckets/namespace/name)
    pub bucket_id: String,

    /// Make the bucket private
    #[arg(long)]
    pub private: bool,

    /// Do not fail if the bucket already exists
    #[arg(long)]
    pub exist_ok: bool,

    /// Print only the bucket handle
    #[arg(short, long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (namespace, name) = parse_bucket_id(&args.bucket_id)?;

    let result = client
        .create_bucket()
        .namespace(&namespace)
        .name(&name)
        .private(args.private)
        .exist_ok(args.exist_ok)
        .send()
        .await?;
    let handle = format!("hf://buckets/{}/{}", namespace, name);

    if args.quiet {
        Ok(CommandResult::Raw(handle))
    } else {
        Ok(CommandResult::Raw(format!("Bucket created: {} (handle: {})", result.url, handle)))
    }
}
