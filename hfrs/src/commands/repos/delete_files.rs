use anyhow::Result;
use clap::Args as ClapArgs;
use globset::Glob;
use hf_hub::HFClient;
use hf_hub::types::{CommitOperation, RepoCreateCommitParams, RepoListFilesParams};

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;

/// Delete files from a repository
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// File patterns to delete (glob patterns or exact paths relative to the repository root)
    #[arg(required = true)]
    pub patterns: Vec<String>,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Git revision to target
    #[arg(long)]
    pub revision: Option<String>,

    /// Commit message
    #[arg(long, default_value = "Delete files using hfrs")]
    pub commit_message: String,

    /// Commit description
    #[arg(long)]
    pub commit_description: Option<String>,

    /// Create a pull request instead of committing directly
    #[arg(long)]
    pub create_pr: bool,
}

pub async fn execute(client: &HFClient, args: Args) -> Result<CommandResult> {
    let repo_type: hf_hub::types::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);

    let list_params = RepoListFilesParams {
        revision: args.revision.clone(),
    };
    let all_files = repo.list_files(&list_params).await?;

    let matchers: Vec<_> = args
        .patterns
        .iter()
        .filter_map(|p| Glob::new(p).ok().map(|g| g.compile_matcher()))
        .collect();

    let matched_files: Vec<String> = all_files
        .into_iter()
        .filter(|f| matchers.iter().any(|m| m.is_match(f)))
        .collect();

    if matched_files.is_empty() {
        anyhow::bail!("No files matched the given patterns");
    }

    let operations = matched_files
        .into_iter()
        .map(|path| CommitOperation::Delete { path_in_repo: path })
        .collect();
    let params = RepoCreateCommitParams {
        operations,
        commit_message: args.commit_message,
        commit_description: args.commit_description,
        revision: args.revision,
        create_pr: if args.create_pr { Some(true) } else { None },
        parent_commit: None,
        progress: None,
    };
    let result = repo.create_commit(&params).await?;
    let url = result.commit_url.or(result.pr_url).unwrap_or_default();
    Ok(CommandResult::Raw(url))
}
