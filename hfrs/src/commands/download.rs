use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::files::{RepoDownloadFileParams, RepoSnapshotDownloadParams};
use hf_hub::progress::Progress;

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;
use crate::progress::CliProgressHandler;

/// Download files from the Hub
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g. username/my-model)
    pub repo_id: String,

    /// Specific filenames to download
    pub filenames: Vec<String>,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Include patterns for snapshot download (can be specified multiple times)
    #[arg(long)]
    pub include: Vec<String>,

    /// Exclude patterns for snapshot download (can be specified multiple times)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Local cache directory
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,

    /// Local directory to save files into (bypasses cache)
    #[arg(long)]
    pub local_dir: Option<PathBuf>,

    /// Force re-download even if cached
    #[arg(long)]
    pub force_download: bool,

    /// Print only the local path, suppress progress
    #[arg(long)]
    pub quiet: bool,
}

pub async fn execute(client: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    let repo_type: hf_hub::RepoType = args.r#type.into();
    let repo = crate::util::make_repo(client, &args.repo_id, repo_type);

    let handler: Option<Progress> = if args.quiet {
        None
    } else if let Some(multi) = multi {
        Some(Arc::new(CliProgressHandler::new(multi)))
    } else {
        None
    };

    let path = if args.filenames.len() == 1 && args.include.is_empty() && args.exclude.is_empty() {
        let params = RepoDownloadFileParams {
            filename: args.filenames.into_iter().next().unwrap(),
            local_dir: args.local_dir,
            revision: args.revision,
            force_download: if args.force_download { Some(true) } else { None },
            local_files_only: None,
            progress: handler.clone(),
        };
        repo.download_file(&params).await?
    } else {
        let allow_patterns = if !args.filenames.is_empty() {
            Some(args.filenames)
        } else if !args.include.is_empty() {
            Some(args.include)
        } else {
            None
        };
        let ignore_patterns = if !args.exclude.is_empty() {
            Some(args.exclude)
        } else {
            None
        };
        let params = RepoSnapshotDownloadParams {
            revision: args.revision,
            allow_patterns,
            ignore_patterns,
            local_dir: args.local_dir,
            force_download: if args.force_download { Some(true) } else { None },
            local_files_only: None,
            max_workers: None,
            progress: handler.clone(),
        };
        repo.snapshot_download(&params).await?
    };

    Ok(CommandResult::Raw(path.display().to_string()))
}
