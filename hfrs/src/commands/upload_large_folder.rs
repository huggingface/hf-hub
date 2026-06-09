use std::path::PathBuf;

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::progress::Progress;
use hf_hub::{HFClient, RepoTypeAny};

use crate::cli::RepoTypeArg;
use crate::output::CommandResult;
use crate::progress::LargeFolderProgressHandler;

/// Upload a large folder to the Hub as resumable, batched commits
#[derive(ClapArgs)]
pub struct Args {
    /// Repository ID (e.g., username/my-model)
    pub repo_id: String,

    /// Local folder to upload
    pub local_path: PathBuf,

    /// Repository type
    #[arg(long, visible_alias = "repo-type", value_enum, default_value = "model")]
    pub r#type: RepoTypeArg,

    /// Git revision (branch, tag, or commit SHA)
    #[arg(long)]
    pub revision: Option<String>,

    /// Create the repo as private if it does not exist yet
    #[arg(long)]
    pub private: bool,

    /// Include only files matching these glob patterns (repeatable)
    #[arg(long)]
    pub include: Vec<String>,

    /// Exclude files matching these glob patterns (repeatable)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Number of concurrent workers (classify + xet upload). Defaults to the library's heuristic.
    #[arg(long)]
    pub num_workers: Option<usize>,

    /// Disable the periodic status report
    #[arg(long)]
    pub no_report: bool,
}

pub async fn execute(client: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    let (owner, name) = match args.repo_id.split_once('/') {
        Some(parts) => parts,
        None => ("", args.repo_id.as_str()),
    };
    let repo = client.repository::<RepoTypeAny>(args.r#type.into(), owner, name);

    let handler: Option<Progress> = match multi {
        Some(multi) if !args.no_report => Some(LargeFolderProgressHandler::new(multi).into()),
        _ => None,
    };

    let allow_patterns = (!args.include.is_empty()).then_some(args.include);
    let ignore_patterns = (!args.exclude.is_empty()).then_some(args.exclude);

    let report = repo
        .upload_large_folder()
        .folder_path(args.local_path)
        .maybe_revision(args.revision)
        .maybe_private(if args.private { Some(true) } else { None })
        .maybe_allow_patterns(allow_patterns)
        .maybe_ignore_patterns(ignore_patterns)
        .maybe_num_workers(args.num_workers)
        .maybe_progress(handler)
        .send()
        .await?;

    let url = report
        .commits
        .last()
        .and_then(|c| c.commit_url.clone().or_else(|| c.pr_url.clone()))
        .unwrap_or_default();
    Ok(CommandResult::Raw(url))
}
