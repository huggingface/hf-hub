use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Args as ClapArgs;
use hf_hub::progress::Progress;
use hf_hub::repository::AddSource;
use hf_hub::{HFClient, HFRepository, RepoType};
use tracing::info;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;
use crate::progress::CliProgressHandler;

/// Upload files to the Hub
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// Local file or folder to upload (defaults to current directory)
    pub local_path: Option<PathBuf>,

    /// Path in the repository to upload to
    pub path_in_repo: Option<String>,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Create the repo as private if it does not exist yet
    #[arg(long)]
    pub private: bool,

    /// Include patterns for folder upload (can be specified multiple times)
    #[arg(long)]
    pub include: Vec<String>,

    /// Exclude patterns for folder upload (can be specified multiple times)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Delete patterns for folder upload (can be specified multiple times)
    #[arg(long)]
    pub delete: Vec<String>,

    /// Commit message
    #[arg(long)]
    pub commit_message: Option<String>,

    /// Commit description
    #[arg(long)]
    pub commit_description: Option<String>,

    /// Create a pull request instead of committing directly
    #[arg(long)]
    pub create_pr: bool,

    /// Print only the commit URL, suppress progress
    #[arg(long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    let local_path = args.local_path.clone().unwrap_or_else(|| PathBuf::from("."));

    let handler: Option<Progress> = if args.quiet {
        None
    } else {
        multi.map(|multi| CliProgressHandler::new(multi).into())
    };

    let url = crate::with_typed_repo!(client, &args.repo_id, args.r#type, |repo| {
        do_upload(client, repo, args, local_path, handler).await?
    });

    Ok(CommandResult::Raw(url))
}

async fn do_upload<T: RepoType>(
    client: &HFClient,
    repo: HFRepository<T>,
    args: Args,
    local_path: PathBuf,
    handler: Option<Progress>,
) -> Result<String> {
    if !repo.exists().send().await? {
        info!(repo_id = args.repo_id.as_str(), private = args.private, "creating repository");
        client
            .create_repo::<T>()
            .repo_id(args.repo_id.clone())
            .maybe_private(if args.private { Some(true) } else { None })
            .exist_ok(true)
            .send()
            .await?;
    }

    let commit_info = if local_path.is_file() {
        let path_in_repo = args.path_in_repo.unwrap_or_else(|| {
            local_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        repo.upload_file()
            .source(AddSource::file(local_path))
            .path_in_repo(path_in_repo)
            .maybe_revision(args.revision)
            .maybe_commit_message(args.commit_message)
            .maybe_commit_description(args.commit_description)
            .create_pr(args.create_pr)
            .maybe_progress(handler.clone())
            .send()
            .await?
    } else if local_path.is_dir() {
        let allow_patterns = (!args.include.is_empty()).then_some(args.include);
        let ignore_patterns = (!args.exclude.is_empty()).then_some(args.exclude);
        let delete_patterns = (!args.delete.is_empty()).then_some(args.delete);
        repo.upload_folder()
            .folder_path(local_path)
            .maybe_path_in_repo(args.path_in_repo)
            .maybe_revision(args.revision)
            .maybe_commit_message(args.commit_message)
            .maybe_commit_description(args.commit_description)
            .create_pr(args.create_pr)
            .maybe_allow_patterns(allow_patterns)
            .maybe_ignore_patterns(ignore_patterns)
            .maybe_delete_patterns(delete_patterns)
            .maybe_progress(handler.clone())
            .send()
            .await?
    } else {
        bail!("local path does not exist: {}", local_path.display());
    };

    Ok(commit_info.commit_url.or(commit_info.pr_url).unwrap_or_default())
}
