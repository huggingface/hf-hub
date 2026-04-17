use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::types::{BucketSyncAction, BucketSyncDirection, BucketSyncParams, Progress};

use crate::output::CommandResult;
use crate::progress::CliProgressHandler;

#[derive(ClapArgs)]
pub struct Args {
    /// Source: local directory or hf://buckets/ns/name(/prefix)
    pub source: String,

    /// Destination: local directory or hf://buckets/ns/name(/prefix)
    pub dest: String,

    /// Delete destination files not present in source
    #[arg(long)]
    pub delete: bool,

    /// Only compare sizes, ignore modification times
    #[arg(long)]
    pub ignore_times: bool,

    /// Only compare modification times, ignore sizes
    #[arg(long)]
    pub ignore_sizes: bool,

    /// Only sync files that already exist at destination
    #[arg(long)]
    pub existing: bool,

    /// Skip files that already exist at destination
    #[arg(long)]
    pub ignore_existing: bool,

    /// Include files matching pattern (can be repeated)
    #[arg(long)]
    pub include: Vec<String>,

    /// Exclude files matching pattern (can be repeated)
    #[arg(long)]
    pub exclude: Vec<String>,

    /// Show per-file operations
    #[arg(short, long)]
    pub verbose: bool,

    /// Suppress output
    #[arg(short, long)]
    pub quiet: bool,
}

struct BucketRef {
    namespace: String,
    bucket_name: String,
    prefix: Option<String>,
}

fn parse_bucket_path(input: &str) -> Option<BucketRef> {
    let rest = input.strip_prefix("hf://buckets/")?;
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() < 2 || parts[0].is_empty() || parts[1].is_empty() {
        return None;
    }
    let prefix = if parts.len() == 3 && !parts[2].is_empty() {
        Some(parts[2].to_string())
    } else {
        None
    };
    Some(BucketRef {
        namespace: parts[0].to_string(),
        bucket_name: parts[1].to_string(),
        prefix,
    })
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

pub async fn execute(client: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    let src_is_bucket = args.source.starts_with("hf://buckets/");
    let dst_is_bucket = args.dest.starts_with("hf://buckets/");

    if src_is_bucket && dst_is_bucket {
        anyhow::bail!("Remote-to-remote sync is not supported.");
    }
    if !src_is_bucket && !dst_is_bucket {
        anyhow::bail!("One of source or dest must be a bucket path (hf://buckets/...).");
    }

    let (bucket_ref, local_path, direction) = if dst_is_bucket {
        let b = parse_bucket_path(&args.dest).ok_or_else(|| anyhow::anyhow!("Invalid bucket path: {}", args.dest))?;
        (b, PathBuf::from(&args.source), BucketSyncDirection::Upload)
    } else {
        let b =
            parse_bucket_path(&args.source).ok_or_else(|| anyhow::anyhow!("Invalid bucket path: {}", args.source))?;
        (b, PathBuf::from(&args.dest), BucketSyncDirection::Download)
    };

    let handler: Progress = if args.quiet {
        None
    } else if let Some(multi) = multi {
        Some(Arc::new(CliProgressHandler::new(multi)))
    } else {
        None
    };

    let bucket = client.bucket(&bucket_ref.namespace, &bucket_ref.bucket_name);

    let params = {
        let b = BucketSyncParams::builder()
            .local_path(local_path)
            .direction(direction)
            .delete(args.delete)
            .ignore_times(args.ignore_times)
            .ignore_sizes(args.ignore_sizes)
            .existing(args.existing)
            .ignore_existing(args.ignore_existing)
            .include(args.include)
            .exclude(args.exclude)
            .verbose(args.verbose)
            .progress(handler);
        if let Some(prefix) = bucket_ref.prefix {
            b.prefix(prefix).build()
        } else {
            b.build()
        }
    };

    let plan = bucket.sync(&params).await?;

    if args.quiet {
        return Ok(CommandResult::Silent);
    }

    if args.verbose {
        for op in &plan.operations {
            let action_str = match op.action {
                BucketSyncAction::Upload => "upload",
                BucketSyncAction::Download => "download",
                BucketSyncAction::Delete => "delete",
                BucketSyncAction::Skip => "skip",
            };
            println!("  {}: {} ({})", action_str, op.path, op.reason);
        }
    }

    let summary = format!(
        "Synced: {} uploaded, {} downloaded, {} deleted, {} skipped ({})",
        plan.uploads(),
        plan.downloads(),
        plan.deletes(),
        plan.skips(),
        format_bytes(plan.transfer_bytes()),
    );

    Ok(CommandResult::Raw(summary))
}
