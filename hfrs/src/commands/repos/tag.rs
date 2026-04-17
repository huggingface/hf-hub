use anyhow::Result;
use clap::{Args as ClapArgs, Subcommand};
use hf_hub::HFClient;
use hf_hub::types::{RepoCreateTagParams, RepoDeleteTagParams, RepoListRefsParams};
use serde_json::json;

use crate::cli::{OutputFormat, RepoTypeArg};
use crate::output::{CommandOutput, CommandResult};

/// Manage repository tags
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    pub command: TagCommand,
}

/// Tag subcommands
#[derive(Subcommand)]
pub enum TagCommand {
    /// Create a new tag
    Create(TagCreateArgs),
    /// Delete a tag
    Delete(TagDeleteArgs),
    /// List tags
    List(TagListArgs),
}

/// Create a new tag
#[derive(ClapArgs)]
pub struct TagCreateArgs {
    /// Repository ID
    pub repo_id: String,

    /// Tag name to create
    pub tag: String,

    /// Tag message (creates an annotated tag)
    #[arg(short = 'm', long)]
    pub message: Option<String>,

    /// Starting revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,
}

/// Delete a tag
#[derive(ClapArgs)]
pub struct TagDeleteArgs {
    /// Repository ID
    pub repo_id: String,

    /// Tag name to delete
    pub tag: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,
}

/// List tags
#[derive(ClapArgs)]
pub struct TagListArgs {
    /// Repository ID
    pub repo_id: String,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    match args.command {
        TagCommand::Create(a) => create(client, a).await,
        TagCommand::Delete(a) => delete(client, a).await,
        TagCommand::List(a) => list(client, a).await,
    }
}

async fn create(client: &HFClient, args: TagCreateArgs) -> Result<CommandResult> {
    let repo_type: hf_hub::types::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);
    let params = RepoCreateTagParams {
        tag: args.tag,
        revision: args.revision,
        message: args.message,
    };
    repo.create_tag(&params).await?;
    Ok(CommandResult::Raw("Tag created.".to_string()))
}

async fn delete(client: &HFClient, args: TagDeleteArgs) -> Result<CommandResult> {
    let repo_type: hf_hub::types::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);
    let params = RepoDeleteTagParams { tag: args.tag };
    repo.delete_tag(&params).await?;
    Ok(CommandResult::Silent)
}

async fn list(client: &HFClient, args: TagListArgs) -> Result<CommandResult> {
    let repo_type: hf_hub::types::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);
    let params = RepoListRefsParams {
        include_pull_requests: false,
    };
    let refs = repo.list_refs(&params).await?;

    if refs.tags.is_empty() && matches!(args.format, OutputFormat::Table) {
        return Ok(CommandResult::Raw("No tags found.".to_string()));
    }

    let headers = vec!["Name".to_string(), "Ref".to_string(), "Commit".to_string()];
    let rows = refs
        .tags
        .iter()
        .map(|t| vec![t.name.clone(), t.git_ref.clone(), t.target_commit.clone()])
        .collect();
    let quiet_values = refs.tags.iter().map(|t| t.name.clone()).collect();
    let json_value = refs
        .tags
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "ref": t.git_ref,
                "target_commit": t.target_commit,
            })
        })
        .collect::<Vec<_>>()
        .into();

    let output = CommandOutput {
        headers,
        rows,
        json_value,
        quiet_values,
    };
    Ok(CommandResult::Formatted {
        output,
        format: args.format,
        quiet: false,
    })
}
