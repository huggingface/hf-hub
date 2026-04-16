use std::io::{self, Write};

use anyhow::Result;
use clap::Args as ClapArgs;
use futures::StreamExt;
use hf_hub::{BucketTreeEntry, HFClient, ListBucketTreeParams};

use crate::output::CommandResult;

#[derive(ClapArgs)]
pub struct Args {
    /// Bucket path (namespace/bucket_name/path or hf://buckets/namespace/bucket_name/path)
    pub argument: String,

    /// Remove files recursively under the given prefix
    #[arg(short = 'R', long)]
    pub recursive: bool,

    /// Skip confirmation prompt
    #[arg(short = 'y', long)]
    pub yes: bool,

    /// Preview deletions without actually deleting
    #[arg(long)]
    pub dry_run: bool,

    /// Include only files matching pattern(s) (requires --recursive)
    #[arg(long)]
    pub include: Vec<String>,

    /// Exclude files matching pattern(s) (requires --recursive)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Print only file paths
    #[arg(short, long)]
    pub quiet: bool,
}

fn parse_bucket_path(input: &str) -> Result<(String, String, Option<String>)> {
    let id = input.strip_prefix("hf://buckets/").unwrap_or(input);
    let parts: Vec<&str> = id.splitn(3, '/').collect();
    match parts.len() {
        2 => Ok((parts[0].to_string(), parts[1].to_string(), None)),
        3 => Ok((
            parts[0].to_string(),
            parts[1].to_string(),
            if parts[2].is_empty() {
                None
            } else {
                Some(parts[2].to_string())
            },
        )),
        _ => anyhow::bail!("Invalid bucket path '{input}'. Expected: namespace/bucket_name/path"),
    }
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    if (!args.include.is_empty() || !args.exclude.is_empty()) && !args.recursive {
        anyhow::bail!("--include and --exclude require --recursive");
    }

    let (namespace, bucket_name, path_prefix) = parse_bucket_path(&args.argument)?;

    if path_prefix.is_none() && !args.recursive {
        anyhow::bail!("Must specify a file path, or use --recursive to delete all files under a prefix");
    }

    let bucket = api.bucket(&namespace, &bucket_name);
    let bucket_id = format!("{namespace}/{bucket_name}");

    let paths_to_delete = if args.recursive {
        let params = ListBucketTreeParams {
            prefix: path_prefix.clone(),
            recursive: Some(true),
        };
        let stream = bucket.list_tree(&params)?;
        futures::pin_mut!(stream);

        let mut paths = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let BucketTreeEntry::File { ref path, .. } = entry {
                let include_match = args.include.is_empty()
                    || args.include.iter().any(|pat| {
                        globset::Glob::new(pat)
                            .ok()
                            .and_then(|g| g.compile_matcher().is_match(path).then_some(()))
                            .is_some()
                    });
                let exclude_match = args.exclude.iter().any(|pat| {
                    globset::Glob::new(pat)
                        .ok()
                        .and_then(|g| g.compile_matcher().is_match(path).then_some(()))
                        .is_some()
                });
                if include_match && !exclude_match {
                    paths.push(path.clone());
                }
            }
        }
        paths
    } else {
        vec![path_prefix.unwrap()]
    };

    if paths_to_delete.is_empty() {
        return Ok(CommandResult::Raw("No files to delete.".to_string()));
    }

    for path in &paths_to_delete {
        if args.quiet {
            println!("{path}");
        } else {
            println!("delete: hf://buckets/{bucket_id}/{path}");
        }
    }

    if args.dry_run {
        let count = paths_to_delete.len();
        return Ok(CommandResult::Raw(format!("(dry run) {count} file(s) would be removed.")));
    }

    if !args.yes {
        let count = paths_to_delete.len();
        eprint!("Delete {count} file(s)? [y/N] ");
        io::stderr().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            return Ok(CommandResult::Raw("Aborted.".to_string()));
        }
    }

    bucket.delete_files(&paths_to_delete).await?;
    let count = paths_to_delete.len();
    Ok(CommandResult::Raw(format!("{count} file(s) removed.")))
}
