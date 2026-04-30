use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use serde_json::json;

use crate::cli::{OutputFormat, RepoTypeArg};
use crate::output::{CommandOutput, CommandResult};

/// Show HEAD metadata for a file in a repository (commit, ETag, size, Xet hash)
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. openai-community/gpt2)
    pub repo_id: String,

    /// File path within the repository
    pub filepath: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let metadata = crate::with_typed_repo!(client, &args.repo_id, args.r#type, |repo| {
        repo.get_file_metadata()
            .filepath(args.filepath)
            .maybe_revision(args.revision)
            .send()
            .await?
    });

    let json_value = json!({
        "filename": metadata.filename,
        "commit_hash": metadata.commit_hash,
        "etag": metadata.etag,
        "file_size": metadata.file_size,
        "xet_hash": metadata.xet_hash,
    });

    Ok(CommandResult::Formatted {
        output: CommandOutput::single_item(json_value),
        format: args.format,
        quiet: false,
    })
}
