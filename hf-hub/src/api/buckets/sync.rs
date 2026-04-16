use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use futures::StreamExt;
use globset::{Glob, GlobMatcher};

use crate::bucket::HFBucket;
use crate::error::{HFError, Result};
use crate::types::progress::{self, DownloadEvent, ProgressEvent};
use crate::types::{
    BucketSyncParams, BucketTreeEntry, ListBucketTreeParams, SyncAction, SyncDirection, SyncOperation, SyncPlan,
};

const SYNC_TIME_WINDOW_MS: f64 = 1000.0;

fn validate_params(params: &BucketSyncParams) -> Result<()> {
    if params.ignore_times && params.ignore_sizes {
        return Err(HFError::InvalidParameter("cannot use both --ignore-times and --ignore-sizes".to_string()));
    }
    if params.existing && params.ignore_existing {
        return Err(HFError::InvalidParameter("cannot use both --existing and --ignore-existing".to_string()));
    }
    if params.direction == SyncDirection::Upload && !params.local_path.is_dir() {
        return Err(HFError::InvalidParameter(format!(
            "local path must be an existing directory for upload: {}",
            params.local_path.display()
        )));
    }
    Ok(())
}

fn compile_patterns(patterns: &[String]) -> Result<Vec<GlobMatcher>> {
    patterns
        .iter()
        .map(|p| {
            Glob::new(p)
                .map(|g| g.compile_matcher())
                .map_err(|e| HFError::InvalidParameter(format!("invalid glob pattern '{p}': {e}")))
        })
        .collect()
}

fn matches_filters(path: &str, include: &[GlobMatcher], exclude: &[GlobMatcher]) -> bool {
    for pat in exclude {
        if pat.is_match(path) {
            return false;
        }
    }
    if include.is_empty() {
        return true;
    }
    for pat in include {
        if pat.is_match(path) {
            return true;
        }
    }
    false
}

fn list_local_files(root: &Path) -> Result<HashMap<String, (u64, f64)>> {
    let mut result = HashMap::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)?;
        for entry in entries {
            let entry = entry?;
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                let path = entry.path();
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| HFError::Other(format!("failed to strip prefix: {e}")))?;
                let rel_str = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                let metadata = std::fs::metadata(&path)?;
                let size = metadata.len();
                let mtime_ms = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as f64)
                    .unwrap_or(0.0);
                result.insert(rel_str, (size, mtime_ms));
            }
        }
    }

    Ok(result)
}

fn strip_prefix<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        return Some(path);
    }
    let with_slash = if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    };
    path.strip_prefix(&with_slash)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let m = if m <= 2 { m + 9 } else { m - 3 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

fn parse_iso_mtime(s: &str) -> f64 {
    // Expected format: "2024-01-15T10:30:00.123Z" or "2024-01-15T10:30:00Z"
    // or "2024-01-15T10:30:00+00:00" etc.
    let s = s.trim();
    if s.len() < 19 {
        return 0.0;
    }

    let parse = || -> Option<f64> {
        let year: i64 = s[0..4].parse().ok()?;
        if s.as_bytes().get(4)? != &b'-' {
            return None;
        }
        let month: i64 = s[5..7].parse().ok()?;
        if s.as_bytes().get(7)? != &b'-' {
            return None;
        }
        let day: i64 = s[8..10].parse().ok()?;
        if s.as_bytes().get(10)? != &b'T' {
            return None;
        }
        let hour: i64 = s[11..13].parse().ok()?;
        if s.as_bytes().get(13)? != &b':' {
            return None;
        }
        let minute: i64 = s[14..16].parse().ok()?;
        if s.as_bytes().get(16)? != &b':' {
            return None;
        }
        let second: i64 = s[17..19].parse().ok()?;

        let mut frac_ms: f64 = 0.0;
        let mut rest_start = 19;
        if s.as_bytes().get(19) == Some(&b'.') {
            let frac_start = 20;
            let mut frac_end = frac_start;
            while frac_end < s.len() && s.as_bytes()[frac_end].is_ascii_digit() {
                frac_end += 1;
            }
            if frac_end > frac_start {
                let frac_str = &s[frac_start..frac_end];
                let frac_val: f64 = frac_str.parse().ok()?;
                let divisor = 10f64.powi(frac_str.len() as i32);
                frac_ms = (frac_val / divisor) * 1000.0;
            }
            rest_start = frac_end;
        }

        // We ignore timezone offset and assume UTC
        let _ = rest_start;

        let days = days_from_civil(year, month, day);
        let total_ms = days as f64 * 86_400_000.0
            + hour as f64 * 3_600_000.0
            + minute as f64 * 60_000.0
            + second as f64 * 1000.0
            + frac_ms;

        Some(total_ms)
    };

    parse().unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy)]
enum CompareRole {
    Upload,
    Download,
}

fn compare_files(
    path: String,
    role: CompareRole,
    source_size: u64,
    source_mtime: f64,
    dest_size: u64,
    dest_mtime: f64,
    params: &BucketSyncParams,
) -> Option<SyncOperation> {
    let action = match role {
        CompareRole::Upload => SyncAction::Upload,
        CompareRole::Download => SyncAction::Download,
    };

    let source_newer = (source_mtime - dest_mtime) > SYNC_TIME_WINDOW_MS;
    let dest_newer = (dest_mtime - source_mtime) > SYNC_TIME_WINDOW_MS;
    let size_differs = source_size != dest_size;

    let source_label = match role {
        CompareRole::Upload => "local newer",
        CompareRole::Download => "remote newer",
    };
    let dest_newer_label = match role {
        CompareRole::Upload => "remote newer",
        CompareRole::Download => "local newer",
    };

    if params.ignore_existing {
        if params.verbose {
            return Some(SyncOperation {
                action: SyncAction::Skip,
                path,
                size: Some(source_size),
                reason: "exists on receiver (--ignore-existing)".to_string(),
            });
        }
        return None;
    }

    if params.ignore_sizes {
        if source_newer {
            return Some(SyncOperation {
                action,
                path,
                size: Some(source_size),
                reason: source_label.to_string(),
            });
        }
        let reason = if dest_newer {
            dest_newer_label.to_string()
        } else {
            "same mtime".to_string()
        };
        if params.verbose {
            return Some(SyncOperation {
                action: SyncAction::Skip,
                path,
                size: Some(source_size),
                reason,
            });
        }
        return None;
    }

    if params.ignore_times {
        if size_differs {
            return Some(SyncOperation {
                action,
                path,
                size: Some(source_size),
                reason: "size differs".to_string(),
            });
        }
        if params.verbose {
            return Some(SyncOperation {
                action: SyncAction::Skip,
                path,
                size: Some(source_size),
                reason: "same size".to_string(),
            });
        }
        return None;
    }

    if size_differs || source_newer {
        let reason = if size_differs {
            "size differs".to_string()
        } else {
            source_label.to_string()
        };
        return Some(SyncOperation {
            action,
            path,
            size: Some(source_size),
            reason,
        });
    }

    if params.verbose {
        return Some(SyncOperation {
            action: SyncAction::Skip,
            path,
            size: Some(source_size),
            reason: "identical".to_string(),
        });
    }
    None
}

impl HFBucket {
    async fn list_remote_files(
        &self,
        prefix: &Option<String>,
        include: &[GlobMatcher],
        exclude: &[GlobMatcher],
    ) -> Result<(HashMap<String, (u64, f64)>, HashMap<String, BucketTreeEntry>)> {
        let tree_params = ListBucketTreeParams {
            prefix: prefix.clone(),
            recursive: Some(true),
        };

        let stream = self.list_tree(&tree_params)?;
        futures::pin_mut!(stream);

        let mut files: HashMap<String, (u64, f64)> = HashMap::new();
        let mut entries: HashMap<String, BucketTreeEntry> = HashMap::new();

        let prefix_str = prefix.as_deref().unwrap_or("");

        while let Some(item) = stream.next().await {
            let entry = item?;
            match &entry {
                BucketTreeEntry::File { path, size, mtime, .. } => {
                    let rel = match strip_prefix(path, prefix_str) {
                        Some(r) => r.to_string(),
                        None => continue,
                    };
                    if !matches_filters(&rel, include, exclude) {
                        continue;
                    }
                    let mtime_ms = mtime.as_deref().map(parse_iso_mtime).unwrap_or(0.0);
                    files.insert(rel.clone(), (*size, mtime_ms));
                    entries.insert(rel, entry);
                },
                BucketTreeEntry::Directory { .. } => continue,
            }
        }

        Ok((files, entries))
    }

    fn compute_upload_plan(
        &self,
        local_files: &HashMap<String, (u64, f64)>,
        remote_files: &HashMap<String, (u64, f64)>,
        params: &BucketSyncParams,
    ) -> SyncPlan {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(local_files.keys().cloned());
        all_keys.extend(remote_files.keys().cloned());

        let mut operations = Vec::new();

        for key in &all_keys {
            let in_local = local_files.contains_key(key);
            let in_remote = remote_files.contains_key(key);

            if in_local && !in_remote {
                if params.existing {
                    if params.verbose {
                        operations.push(SyncOperation {
                            action: SyncAction::Skip,
                            path: key.clone(),
                            size: local_files.get(key).map(|(s, _)| *s),
                            reason: "not on receiver (--existing)".to_string(),
                        });
                    }
                } else {
                    let (size, _) = local_files[key];
                    operations.push(SyncOperation {
                        action: SyncAction::Upload,
                        path: key.clone(),
                        size: Some(size),
                        reason: "new file".to_string(),
                    });
                }
            } else if in_local && in_remote {
                let (local_size, local_mtime) = local_files[key];
                let (remote_size, remote_mtime) = remote_files[key];
                if let Some(op) = compare_files(
                    key.clone(),
                    CompareRole::Upload,
                    local_size,
                    local_mtime,
                    remote_size,
                    remote_mtime,
                    params,
                ) {
                    operations.push(op);
                }
            } else if !in_local && in_remote && params.delete {
                let (size, _) = remote_files[key];
                operations.push(SyncOperation {
                    action: SyncAction::Delete,
                    path: key.clone(),
                    size: Some(size),
                    reason: "not in source (--delete)".to_string(),
                });
            }
        }

        SyncPlan {
            direction: SyncDirection::Upload,
            operations,
            download_entries: HashMap::new(),
        }
    }

    fn compute_download_plan(
        &self,
        local_files: &HashMap<String, (u64, f64)>,
        remote_files: &HashMap<String, (u64, f64)>,
        remote_entries: &HashMap<String, BucketTreeEntry>,
        params: &BucketSyncParams,
    ) -> SyncPlan {
        let mut all_keys = BTreeSet::new();
        all_keys.extend(local_files.keys().cloned());
        all_keys.extend(remote_files.keys().cloned());

        let mut operations = Vec::new();
        let mut download_entries = HashMap::new();

        for key in &all_keys {
            let in_local = local_files.contains_key(key);
            let in_remote = remote_files.contains_key(key);

            if in_remote && !in_local {
                if params.existing {
                    if params.verbose {
                        operations.push(SyncOperation {
                            action: SyncAction::Skip,
                            path: key.clone(),
                            size: remote_files.get(key).map(|(s, _)| *s),
                            reason: "not on receiver (--existing)".to_string(),
                        });
                    }
                } else {
                    let (size, _) = remote_files[key];
                    operations.push(SyncOperation {
                        action: SyncAction::Download,
                        path: key.clone(),
                        size: Some(size),
                        reason: "new file".to_string(),
                    });
                    if let Some(entry) = remote_entries.get(key) {
                        download_entries.insert(key.clone(), entry.clone());
                    }
                }
            } else if in_local && in_remote {
                let (remote_size, remote_mtime) = remote_files[key];
                let (local_size, local_mtime) = local_files[key];
                if let Some(op) = compare_files(
                    key.clone(),
                    CompareRole::Download,
                    remote_size,
                    remote_mtime,
                    local_size,
                    local_mtime,
                    params,
                ) {
                    if op.action == SyncAction::Download
                        && let Some(entry) = remote_entries.get(key)
                    {
                        download_entries.insert(key.clone(), entry.clone());
                    }
                    operations.push(op);
                }
            } else if in_local && !in_remote && params.delete {
                let (size, _) = local_files[key];
                operations.push(SyncOperation {
                    action: SyncAction::Delete,
                    path: key.clone(),
                    size: Some(size),
                    reason: "not in source (--delete)".to_string(),
                });
            }
        }

        SyncPlan {
            direction: SyncDirection::Download,
            operations,
            download_entries,
        }
    }

    async fn execute_upload_plan(&self, plan: &SyncPlan, params: &BucketSyncParams) -> Result<()> {
        let upload_files: Vec<(PathBuf, String)> = plan
            .operations
            .iter()
            .filter(|op| op.action == SyncAction::Upload)
            .map(|op| {
                let local_path = params.local_path.join(&op.path);
                let remote_path = match &params.prefix {
                    Some(prefix) => format!("{prefix}/{}", op.path),
                    None => op.path.clone(),
                };
                (local_path, remote_path)
            })
            .collect();

        let delete_paths: Vec<String> = plan
            .operations
            .iter()
            .filter(|op| op.action == SyncAction::Delete)
            .map(|op| match &params.prefix {
                Some(prefix) => format!("{prefix}/{}", op.path),
                None => op.path.clone(),
            })
            .collect();

        if !upload_files.is_empty() {
            self.upload_files(&upload_files, &params.progress).await?;
        }
        if !delete_paths.is_empty() {
            self.delete_files(&delete_paths).await?;
        }

        Ok(())
    }

    async fn execute_download_plan(&self, plan: &SyncPlan, params: &BucketSyncParams) -> Result<()> {
        let mut xet_batch_files = Vec::new();
        let mut total_bytes: u64 = 0;

        for op in plan.operations.iter().filter(|op| op.action == SyncAction::Download) {
            let entry = plan.download_entries.get(&op.path).ok_or_else(|| HFError::EntryNotFound {
                path: op.path.clone(),
                repo_id: self.bucket_id(),
            })?;

            let local_path = params.local_path.join(&op.path);
            if let Some(parent) = local_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            match entry {
                BucketTreeEntry::File { xet_hash, size, .. } => {
                    let remote_path = match &params.prefix {
                        Some(prefix) => format!("{prefix}/{}", op.path),
                        None => op.path.clone(),
                    };
                    total_bytes += size;
                    xet_batch_files.push(crate::xet::XetBatchFile {
                        hash: xet_hash.clone(),
                        file_size: *size,
                        path: local_path,
                        filename: remote_path,
                    });
                },
                BucketTreeEntry::Directory { path, .. } => {
                    return Err(HFError::InvalidParameter(format!("Cannot download directory entry: {path}")));
                },
            }
        }

        if !xet_batch_files.is_empty() {
            progress::emit(
                &params.progress,
                ProgressEvent::Download(DownloadEvent::Start {
                    total_files: xet_batch_files.len(),
                    total_bytes,
                }),
            );
            self.xet_download_batch(&xet_batch_files, &params.progress).await?;
            progress::emit(&params.progress, ProgressEvent::Download(DownloadEvent::Complete));
        }

        let delete_paths: Vec<&str> = plan
            .operations
            .iter()
            .filter(|op| op.action == SyncAction::Delete)
            .map(|op| op.path.as_str())
            .collect();

        for rel_path in &delete_paths {
            let local_path = params.local_path.join(rel_path);
            if local_path.is_file() {
                std::fs::remove_file(&local_path)?;
            }
        }

        // Clean up empty parent directories (leaf-first)
        let mut dirs_to_check: BTreeSet<PathBuf> = BTreeSet::new();
        for rel_path in &delete_paths {
            let local_path = params.local_path.join(rel_path);
            let mut parent = local_path.parent().map(Path::to_path_buf);
            while let Some(p) = parent {
                if p == params.local_path || !p.starts_with(&params.local_path) {
                    break;
                }
                dirs_to_check.insert(p.clone());
                parent = p.parent().map(Path::to_path_buf);
            }
        }

        let mut dirs_sorted: Vec<PathBuf> = dirs_to_check.into_iter().collect();
        dirs_sorted.sort_by_key(|b| std::cmp::Reverse(b.components().count()));
        for dir in dirs_sorted {
            if dir.is_dir() {
                let is_empty = std::fs::read_dir(&dir).map(|mut d| d.next().is_none()).unwrap_or(false);
                if is_empty {
                    let _ = std::fs::remove_dir(&dir);
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "xet")]
    pub async fn sync(&self, params: &BucketSyncParams) -> Result<SyncPlan> {
        validate_params(params)?;

        let include = compile_patterns(&params.include)?;
        let exclude = compile_patterns(&params.exclude)?;

        let (remote_files, remote_entries) = self.list_remote_files(&params.prefix, &include, &exclude).await?;

        match params.direction {
            SyncDirection::Upload => {
                let all_local = list_local_files(&params.local_path)?;
                let local_files: HashMap<String, (u64, f64)> = all_local
                    .into_iter()
                    .filter(|(k, _)| matches_filters(k, &include, &exclude))
                    .collect();

                let plan = self.compute_upload_plan(&local_files, &remote_files, params);
                self.execute_upload_plan(&plan, params).await?;
                Ok(plan)
            },
            SyncDirection::Download => {
                std::fs::create_dir_all(&params.local_path)?;

                let all_local = list_local_files(&params.local_path)?;
                let local_files: HashMap<String, (u64, f64)> = all_local
                    .into_iter()
                    .filter(|(k, _)| matches_filters(k, &include, &exclude))
                    .collect();

                let plan = self.compute_download_plan(&local_files, &remote_files, &remote_entries, params);
                self.execute_download_plan(&plan, params).await?;
                Ok(plan)
            },
        }
    }

    #[cfg(not(feature = "xet"))]
    pub async fn sync(&self, _params: &BucketSyncParams) -> Result<SyncPlan> {
        Err(HFError::XetNotEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso_mtime_basic() {
        let ms = parse_iso_mtime("2024-01-15T10:30:00Z");
        assert!(ms > 0.0);

        let expected_days = days_from_civil(2024, 1, 15);
        let expected_ms = expected_days as f64 * 86_400_000.0 + 10.0 * 3_600_000.0 + 30.0 * 60_000.0;
        assert!((ms - expected_ms).abs() < 1.0);
    }

    #[test]
    fn test_parse_iso_mtime_with_frac() {
        let ms = parse_iso_mtime("2024-01-15T10:30:00.500Z");
        let expected_days = days_from_civil(2024, 1, 15);
        let expected_ms = expected_days as f64 * 86_400_000.0 + 10.0 * 3_600_000.0 + 30.0 * 60_000.0 + 500.0;
        assert!((ms - expected_ms).abs() < 1.0);
    }

    #[test]
    fn test_parse_iso_mtime_invalid() {
        assert_eq!(parse_iso_mtime(""), 0.0);
        assert_eq!(parse_iso_mtime("not a date"), 0.0);
        assert_eq!(parse_iso_mtime("short"), 0.0);
    }

    #[test]
    fn test_days_from_civil_epoch() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn test_days_from_civil_known_date() {
        // 2024-01-01 is 19723 days after epoch
        assert_eq!(days_from_civil(2024, 1, 1), 19723);
    }

    #[test]
    fn test_strip_prefix_empty() {
        assert_eq!(strip_prefix("foo/bar.txt", ""), Some("foo/bar.txt"));
    }

    #[test]
    fn test_strip_prefix_match() {
        assert_eq!(strip_prefix("data/foo/bar.txt", "data"), Some("foo/bar.txt"));
        assert_eq!(strip_prefix("data/foo/bar.txt", "data/"), Some("foo/bar.txt"));
    }

    #[test]
    fn test_strip_prefix_no_match() {
        assert_eq!(strip_prefix("other/bar.txt", "data"), None);
    }

    #[test]
    fn test_strip_prefix_partial_no_match() {
        assert_eq!(strip_prefix("submarine.txt", "sub"), None);
    }

    #[test]
    fn test_strip_prefix_exact_file() {
        assert_eq!(strip_prefix("data/file.txt", "data"), Some("file.txt"));
    }

    #[test]
    fn test_matches_filters_no_patterns() {
        assert!(matches_filters("foo.txt", &[], &[]));
    }

    #[test]
    fn test_matches_filters_include() {
        let include = compile_patterns(&["*.txt".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &include, &[]));
        assert!(!matches_filters("foo.bin", &include, &[]));
    }

    #[test]
    fn test_matches_filters_exclude() {
        let exclude = compile_patterns(&["*.log".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &[], &exclude));
        assert!(!matches_filters("foo.log", &[], &exclude));
    }

    #[test]
    fn test_matches_filters_exclude_takes_priority() {
        let include = compile_patterns(&["*.txt".to_string()]).unwrap();
        let exclude = compile_patterns(&["secret*".to_string()]).unwrap();
        assert!(matches_filters("foo.txt", &include, &exclude));
        assert!(!matches_filters("secret.txt", &include, &exclude));
    }

    #[test]
    fn test_validate_params_conflicting_times_sizes() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Download)
            .ignore_times(true)
            .ignore_sizes(true)
            .build();
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_validate_params_conflicting_existing() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Download)
            .existing(true)
            .ignore_existing(true)
            .build();
        assert!(validate_params(&params).is_err());
    }

    #[test]
    fn test_compare_files_identical() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 100, 5000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "identical");
    }

    #[test]
    fn test_compare_files_identical_not_verbose() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 100, 5000.0, 100, 5000.0, &params);
        assert!(op.is_none());
    }

    #[test]
    fn test_compare_files_size_differs() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 200, 5000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Upload);
        assert_eq!(op.reason, "size differs");
    }

    #[test]
    fn test_compare_files_source_newer() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 100, 7000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Upload);
        assert_eq!(op.reason, "local newer");
    }

    #[test]
    fn test_compare_files_download_source_newer() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Download)
            .build();

        let op = compare_files(String::new(), CompareRole::Download, 100, 7000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Download);
        assert_eq!(op.reason, "remote newer");
    }

    #[test]
    fn test_compare_files_within_safety_window() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 100, 5500.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "identical");
    }

    #[test]
    fn test_compare_files_ignore_times() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .ignore_times(true)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 100, 9000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "same size");

        let op = compare_files(String::new(), CompareRole::Upload, 200, 5000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Upload);
        assert_eq!(op.reason, "size differs");
    }

    #[test]
    fn test_compare_files_ignore_sizes() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .ignore_sizes(true)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 200, 5000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "same mtime");

        let op = compare_files(String::new(), CompareRole::Upload, 100, 3000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "remote newer");
    }

    #[test]
    fn test_compare_files_ignore_sizes_download() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Download)
            .ignore_sizes(true)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Download, 100, 3000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "local newer");
    }

    #[test]
    fn test_compare_files_ignore_existing() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .ignore_existing(true)
            .verbose(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 200, 9000.0, 100, 5000.0, &params).unwrap();
        assert_eq!(op.action, SyncAction::Skip);
        assert_eq!(op.reason, "exists on receiver (--ignore-existing)");
    }

    #[test]
    fn test_compare_files_ignore_existing_not_verbose() {
        let params = BucketSyncParams::builder()
            .local_path(PathBuf::from("/tmp"))
            .direction(SyncDirection::Upload)
            .ignore_existing(true)
            .build();

        let op = compare_files(String::new(), CompareRole::Upload, 200, 9000.0, 100, 5000.0, &params);
        assert!(op.is_none());
    }
}
