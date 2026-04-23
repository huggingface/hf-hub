use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Args as ClapArgs;
use hf_hub::HFClient;
use hf_hub::types::{BucketDownloadFilesParams, Progress};

use crate::output::CommandResult;
use crate::progress::CliProgressHandler;

#[derive(ClapArgs)]
pub struct Args {
    /// Source: local path, hf://buckets/ns/name/path, or - for stdin
    pub src: String,

    /// Destination: local path, hf://buckets/ns/name/path, or - for stdout
    pub dst: Option<String>,

    /// Suppress output
    #[arg(short, long)]
    pub quiet: bool,
}

struct BucketPath {
    namespace: String,
    bucket_name: String,
    path: String,
}

fn parse_bucket_path(input: &str) -> Option<BucketPath> {
    let rest = input.strip_prefix("hf://buckets/")?;
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() < 3 || parts[2].is_empty() {
        return None;
    }
    Some(BucketPath {
        namespace: parts[0].to_string(),
        bucket_name: parts[1].to_string(),
        path: parts[2].to_string(),
    })
}

fn filename_from_path(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub async fn execute(client: &HFClient, args: Args, multi: Option<indicatif::MultiProgress>) -> Result<CommandResult> {
    let handler: Option<Progress> = if args.quiet {
        None
    } else if let Some(multi) = multi {
        Some(Arc::new(CliProgressHandler::new(multi)))
    } else {
        None
    };

    let src_is_stdin = args.src == "-";
    let src_is_bucket = args.src.starts_with("hf://buckets/");
    let dst_str = args.dst.clone().unwrap_or_else(|| ".".to_string());
    let dst_is_stdout = dst_str == "-";
    let dst_is_bucket = dst_str.starts_with("hf://buckets/");

    if !src_is_bucket && !dst_is_bucket && !src_is_stdin && !dst_is_stdout {
        anyhow::bail!("At least one of source or destination must be a bucket path (hf://buckets/...)");
    }

    if !src_is_bucket && !src_is_stdin && dst_is_bucket {
        return upload_local(client, &args.src, &dst_str, args.quiet, &handler).await;
    }

    if src_is_stdin && dst_is_bucket {
        return upload_stdin(client, &dst_str, args.quiet, &handler).await;
    }

    if src_is_bucket && !dst_is_bucket && !dst_is_stdout {
        return download_to_local(client, &args.src, &dst_str, args.quiet, &handler).await;
    }

    if src_is_bucket && dst_is_stdout {
        return download_to_stdout(client, &args.src).await;
    }

    if src_is_bucket && dst_is_bucket {
        return server_side_copy(client, &args.src, &dst_str, args.quiet).await;
    }

    anyhow::bail!("Unsupported copy operation: {} -> {}", args.src, dst_str)
}

async fn upload_local(
    client: &HFClient,
    src: &str,
    dst: &str,
    quiet: bool,
    progress: &Option<Progress>,
) -> Result<CommandResult> {
    let dst = parse_bucket_path(dst).ok_or_else(|| anyhow::anyhow!("Invalid bucket destination: {dst}"))?;
    let local_path = PathBuf::from(src);
    if !local_path.exists() {
        anyhow::bail!("Source file not found: {src}");
    }
    let bucket = client.bucket(&dst.namespace, &dst.bucket_name);
    bucket.upload_files(&[(local_path, dst.path.clone())], progress).await?;
    if !quiet {
        return Ok(CommandResult::Raw(format!(
            "Uploaded: {} -> hf://buckets/{}/{}/{}",
            src, dst.namespace, dst.bucket_name, dst.path
        )));
    }
    Ok(CommandResult::Silent)
}

async fn upload_stdin(client: &HFClient, dst: &str, quiet: bool, progress: &Option<Progress>) -> Result<CommandResult> {
    let dst = parse_bucket_path(dst).ok_or_else(|| anyhow::anyhow!("Invalid bucket destination: {dst}"))?;
    let mut data = Vec::new();
    io::stdin().read_to_end(&mut data)?;
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), &data)?;
    let bucket = client.bucket(&dst.namespace, &dst.bucket_name);
    bucket
        .upload_files(&[(tmp.path().to_path_buf(), dst.path.clone())], progress)
        .await?;
    if !quiet {
        return Ok(CommandResult::Raw(format!(
            "Uploaded: (stdin) -> hf://buckets/{}/{}/{}",
            dst.namespace, dst.bucket_name, dst.path
        )));
    }
    Ok(CommandResult::Silent)
}

async fn download_to_local(
    client: &HFClient,
    src: &str,
    dst: &str,
    quiet: bool,
    progress: &Option<Progress>,
) -> Result<CommandResult> {
    let src = parse_bucket_path(src).ok_or_else(|| anyhow::anyhow!("Invalid bucket source: {src}"))?;
    let mut local_path = PathBuf::from(dst);
    if local_path.is_dir() {
        local_path = local_path.join(filename_from_path(&src.path));
    }
    if let Some(parent) = local_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bucket = client.bucket(&src.namespace, &src.bucket_name);
    let params = BucketDownloadFilesParams::builder()
        .files(vec![(src.path.clone(), local_path.clone())])
        .build();
    bucket.download_files(&params, progress).await?;
    if !quiet {
        return Ok(CommandResult::Raw(format!(
            "Downloaded: hf://buckets/{}/{}/{} -> {}",
            src.namespace,
            src.bucket_name,
            src.path,
            local_path.display()
        )));
    }
    Ok(CommandResult::Silent)
}

async fn download_to_stdout(client: &HFClient, src: &str) -> Result<CommandResult> {
    let src = parse_bucket_path(src).ok_or_else(|| anyhow::anyhow!("Invalid bucket source: {src}"))?;
    let tmp = tempfile::NamedTempFile::new()?;
    let bucket = client.bucket(&src.namespace, &src.bucket_name);
    let params = BucketDownloadFilesParams::builder()
        .files(vec![(src.path.clone(), tmp.path().to_path_buf())])
        .build();
    let no_progress: Option<Progress> = None;
    bucket.download_files(&params, &no_progress).await?;
    let data = std::fs::read(tmp.path())?;
    io::stdout().write_all(&data)?;
    Ok(CommandResult::Silent)
}

async fn server_side_copy(client: &HFClient, src: &str, dst: &str, quiet: bool) -> Result<CommandResult> {
    let src = parse_bucket_path(src).ok_or_else(|| anyhow::anyhow!("Invalid bucket source: {src}"))?;
    let dst = parse_bucket_path(dst).ok_or_else(|| anyhow::anyhow!("Invalid bucket destination: {dst}"))?;
    let src_bucket = client.bucket(&src.namespace, &src.bucket_name);
    let metadata = src_bucket.get_file_metadata(&src.path).await?;
    let dst_bucket = client.bucket(&dst.namespace, &dst.bucket_name);
    let copy_params = hf_hub::types::BatchBucketFilesParams {
        copy: vec![hf_hub::types::BucketCopyFile {
            path: dst.path.clone(),
            xet_hash: metadata.xet_hash,
            source_repo_type: "bucket".to_string(),
            source_repo_id: format!("{}/{}", src.namespace, src.bucket_name),
        }],
        ..Default::default()
    };
    dst_bucket.batch(&copy_params).await?;
    if !quiet {
        return Ok(CommandResult::Raw(format!(
            "Copied: hf://buckets/{}/{}/{} -> hf://buckets/{}/{}/{}",
            src.namespace, src.bucket_name, src.path, dst.namespace, dst.bucket_name, dst.path
        )));
    }
    Ok(CommandResult::Silent)
}
