use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::{HFClient, RepoGetFileMetadataParams};
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

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(api, &args.repo_id, repo_type);

    let params = RepoGetFileMetadataParams {
        filepath: args.filepath,
        revision: args.revision,
    };
    let metadata = repo.get_file_metadata(&params).await?;

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
