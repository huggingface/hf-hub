use anyhow::Result;
use clap::Args as ClapArgs;

use crate::output::CommandResult;

#[derive(ClapArgs)]
#[command(about = "Print the hfrs version")]
pub struct Args {}

pub async fn execute(_args: Args) -> Result<CommandResult> {
    Ok(CommandResult::Raw(format!("hfrs {}", env!("CARGO_PKG_VERSION"))))
}
