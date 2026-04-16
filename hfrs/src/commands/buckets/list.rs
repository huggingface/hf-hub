use anyhow::Result;
use clap::Args as ClapArgs;
use futures::StreamExt;
use hf_hub::{BucketTreeEntry, HFClient, ListBucketTreeParams};

use crate::cli::OutputFormat;
use crate::output::{CommandOutput, CommandResult};

#[derive(ClapArgs)]
pub struct Args {
    /// Namespace (to list buckets) or bucket_id[/prefix] (to list files)
    pub argument: String,

    /// Show sizes in human-readable format
    #[arg(short = 'h', long)]
    pub human_readable: bool,

    /// Display files in tree format
    #[arg(long)]
    pub tree: bool,

    /// List files recursively
    #[arg(short = 'R', long)]
    pub recursive: bool,

    /// Output format
    #[arg(long, value_enum, default_value = "table")]
    pub format: OutputFormat,

    /// Print only paths, one per line
    #[arg(short, long)]
    pub quiet: bool,
}

enum ListTarget {
    Namespace(String),
    Files {
        namespace: String,
        bucket_name: String,
        prefix: Option<String>,
    },
}

fn parse_list_argument(input: &str) -> ListTarget {
    let id = input.strip_prefix("hf://buckets/").unwrap_or(input);

    let Some(first_slash) = id.find('/') else {
        return ListTarget::Namespace(id.to_string());
    };

    let namespace = &id[..first_slash];
    let rest = &id[first_slash + 1..];

    if let Some(slash_pos) = rest.find('/') {
        let bucket_name = &rest[..slash_pos];
        let prefix = &rest[slash_pos + 1..];
        ListTarget::Files {
            namespace: namespace.to_string(),
            bucket_name: bucket_name.to_string(),
            prefix: if prefix.is_empty() {
                None
            } else {
                Some(prefix.to_string())
            },
        }
    } else {
        ListTarget::Files {
            namespace: namespace.to_string(),
            bucket_name: rest.to_string(),
            prefix: None,
        }
    }
}

pub async fn execute(api: &HFClient, args: Args) -> Result<CommandResult> {
    match parse_list_argument(&args.argument) {
        ListTarget::Namespace(namespace) => {
            if args.tree || args.recursive {
                anyhow::bail!("--tree and --recursive are only valid when listing files in a bucket");
            }
            list_buckets(api, &namespace, args.format, args.quiet, args.human_readable).await
        },
        ListTarget::Files {
            namespace,
            bucket_name,
            prefix,
        } => {
            if args.tree && matches!(args.format, OutputFormat::Json) {
                anyhow::bail!("--tree cannot be used with --format json");
            }
            list_files(api, &namespace, &bucket_name, prefix, &args).await
        },
    }
}

async fn list_buckets(
    api: &HFClient,
    namespace: &str,
    format: OutputFormat,
    quiet: bool,
    human_readable: bool,
) -> Result<CommandResult> {
    let stream = api.list_buckets(namespace)?;
    futures::pin_mut!(stream);

    let mut buckets = Vec::new();
    while let Some(bucket) = stream.next().await {
        buckets.push(bucket?);
    }

    let json_value = serde_json::to_value(&buckets)?;
    let headers = vec![
        "id".to_string(),
        "private".to_string(),
        "size".to_string(),
        "total_files".to_string(),
        "created_at".to_string(),
    ];
    let rows: Vec<Vec<String>> = buckets
        .iter()
        .map(|b| {
            vec![
                b.id.clone(),
                b.private.to_string(),
                if human_readable {
                    format_size_human(b.size)
                } else {
                    b.size.to_string()
                },
                b.total_files.to_string(),
                b.created_at.clone(),
            ]
        })
        .collect();
    let quiet_values = buckets.iter().map(|b| b.id.clone()).collect();

    Ok(CommandResult::Formatted {
        output: CommandOutput {
            headers,
            rows,
            json_value,
            quiet_values,
        },
        format,
        quiet,
    })
}

async fn list_files(
    api: &HFClient,
    namespace: &str,
    bucket_name: &str,
    prefix: Option<String>,
    args: &Args,
) -> Result<CommandResult> {
    let bucket = api.bucket(namespace, bucket_name);
    let params = ListBucketTreeParams {
        prefix,
        recursive: if args.recursive { Some(true) } else { None },
    };

    let stream = bucket.list_tree(&params)?;
    futures::pin_mut!(stream);

    let mut entries = Vec::new();
    while let Some(entry) = stream.next().await {
        entries.push(entry?);
    }

    if args.tree {
        let tree_output = format_tree(&entries);
        return Ok(CommandResult::Raw(tree_output));
    }

    if args.quiet {
        let lines: Vec<String> = entries
            .iter()
            .map(|e| match e {
                BucketTreeEntry::File { path, .. } => path.clone(),
                BucketTreeEntry::Directory { path, .. } => format!("{path}/"),
            })
            .collect();
        return Ok(CommandResult::Raw(lines.join("\n")));
    }

    let json_value = serde_json::to_value(&entries)?;
    if matches!(args.format, OutputFormat::Json) {
        return Ok(CommandResult::Raw(serde_json::to_string_pretty(&json_value)?));
    }

    let lines: Vec<String> = entries
        .iter()
        .map(|e| match e {
            BucketTreeEntry::File { path, size, mtime, .. } => {
                let size_str = if args.human_readable {
                    format_size_human(*size)
                } else {
                    size.to_string()
                };
                let date_str = mtime.as_deref().unwrap_or("-");
                format!("{size_str:>10}  {date_str}  {path}")
            },
            BucketTreeEntry::Directory { path, .. } => {
                format!("{:>10}  {}  {path}/", "-", "-")
            },
        })
        .collect();

    Ok(CommandResult::Raw(lines.join("\n")))
}

fn format_tree(entries: &[BucketTreeEntry]) -> String {
    let mut lines = Vec::new();
    let total = entries.len();
    for (i, entry) in entries.iter().enumerate() {
        let is_last = i == total - 1;
        let connector = if is_last {
            "\u{2514}\u{2500}\u{2500}"
        } else {
            "\u{251c}\u{2500}\u{2500}"
        };
        let name = match entry {
            BucketTreeEntry::File { path, .. } => path.clone(),
            BucketTreeEntry::Directory { path, .. } => format!("{path}/"),
        };
        lines.push(format!("{connector} {name}"));
    }
    lines.join("\n")
}

fn format_size_human(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut s = size as f64;
    for unit in UNITS {
        if s < 1024.0 {
            return if s.fract() == 0.0 {
                format!("{:.0}{unit}", s)
            } else {
                format!("{:.1}{unit}", s)
            };
        }
        s /= 1024.0;
    }
    format!("{:.1}PB", s)
}
