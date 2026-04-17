use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;

use super::parse_bucket_id;
use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    /// Source bucket ID
    pub from_id: String,

    /// Destination bucket ID
    pub to_id: String,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let (from_ns, from_name) = parse_bucket_id(&args.from_id)?;
    let (to_ns, to_name) = parse_bucket_id(&args.to_id)?;
    let from = format!("{}/{}", from_ns, from_name);
    let to = format!("{}/{}", to_ns, to_name);

    client.move_bucket(&from, &to).await?;

    Ok(CommandResult::Raw(format!("Bucket moved: {from} -> {to}")))
}
