use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use super::parse_bucket_id;
use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    /// Bucket ID (namespace/name or hf://buckets/namespace/name)
    pub bucket_id: String,

    /// Print only the bucket ID
    #[arg(short, long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (namespace, name) = parse_bucket_id(&args.bucket_id)?;
    let bucket = client.bucket(&namespace, &name);
    let info = bucket.info().send().await?;

    if args.quiet {
        Ok(CommandResult::Raw(info.id.clone()))
    } else {
        let json = serde_json::to_string_pretty(&info)?;
        Ok(CommandResult::Raw(json))
    }
}
