use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use fs4::fs_std::FileExt;

use crate::types::RepoType;
use crate::types::cache::{CachedFileInfo, CachedRepoInfo, CachedRevisionInfo, HFCacheInfo};

pub(crate) struct CacheLock {
    _file: File,
}

pub(crate) async fn acquire_lock(cache_dir: &Path, repo_folder: &str, etag: &str) -> crate::error::Result<CacheLock> {
    let path = lock_path(cache_dir, repo_folder, etag);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let lock_path_clone = path.clone();
    let lock = tokio::time::timeout(
        std::time::Duration::from_secs(crate::constants::CACHE_LOCK_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let file = File::create(&lock_path_clone)?;
            file.lock_exclusive()?;
            Ok::<_, std::io::Error>(file)
        }),
    )
    .await
    .map_err(|_| crate::error::HFError::CacheLockTimeout { path: path.clone() })?
    .map_err(|e| crate::error::HFError::Other(format!("Lock task failed: {e}")))?
    .map_err(crate::error::HFError::Io)?;
    Ok(CacheLock { _file: lock })
}

pub(crate) async fn write_ref(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
    commit_hash: &str,
) -> crate::error::Result<()> {
    let path = ref_path(cache_dir, repo_folder, revision);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, commit_hash).await?;
    Ok(())
}

pub(crate) async fn read_ref(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
) -> crate::error::Result<Option<String>> {
    let path = ref_path(cache_dir, repo_folder, revision);
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Ok(Some(content.trim().to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// On Windows, copies the blob instead of creating a symlink because symlinks
/// require elevated privileges. This means `find_cached_etag` (which uses
/// `read_link`) cannot determine the cached etag on Windows, effectively
/// disabling conditional-request (If-None-Match) optimization.
pub(crate) async fn create_pointer_symlink(
    cache_dir: &Path,
    repo_folder: &str,
    commit_hash: &str,
    filename: &str,
    etag: &str,
) -> crate::error::Result<()> {
    let pointer = snapshot_path(cache_dir, repo_folder, commit_hash, filename);
    if let Some(parent) = pointer.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let blob = blob_path(cache_dir, repo_folder, etag);
    let pointer_parent = pointer.parent().unwrap();
    let relative = pathdiff::diff_paths(&blob, pointer_parent).unwrap_or(blob);
    let _ = tokio::fs::remove_file(&pointer).await;

    #[cfg(not(windows))]
    {
        match tokio::fs::symlink(&relative, &pointer).await {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(e) => return Err(e.into()),
        }
    }
    #[cfg(windows)]
    {
        match tokio::fs::copy(&blob_path(cache_dir, repo_folder, etag), &pointer).await {
            Ok(_) => {},
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

pub(crate) fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())
}

pub(crate) fn repo_folder_name(repo_id: &str, repo_type: Option<RepoType>) -> String {
    let repo_type = repo_type.unwrap_or(RepoType::Model);
    let type_plural = format!("{}s", repo_type);
    std::iter::once(type_plural.as_str())
        .chain(repo_id.split('/'))
        .collect::<Vec<_>>()
        .join("--")
}

pub(crate) fn blob_path(cache_dir: &Path, repo_folder: &str, etag: &str) -> PathBuf {
    cache_dir.join(repo_folder).join("blobs").join(etag)
}

pub(crate) fn snapshot_path(cache_dir: &Path, repo_folder: &str, commit_hash: &str, filename: &str) -> PathBuf {
    cache_dir.join(repo_folder).join("snapshots").join(commit_hash).join(filename)
}

pub(crate) fn ref_path(cache_dir: &Path, repo_folder: &str, revision: &str) -> PathBuf {
    cache_dir.join(repo_folder).join("refs").join(revision)
}

pub(crate) fn lock_path(cache_dir: &Path, repo_folder: &str, etag: &str) -> PathBuf {
    cache_dir.join(".locks").join(repo_folder).join(format!("{etag}.lock"))
}

pub(crate) fn no_exist_path(cache_dir: &Path, repo_folder: &str, commit_hash: &str, filename: &str) -> PathBuf {
    cache_dir.join(repo_folder).join(".no_exist").join(commit_hash).join(filename)
}

fn parse_repo_folder_name(name: &str) -> Option<(RepoType, String)> {
    let (type_plural, rest) = name.split_once("--")?;
    let type_singular = type_plural.strip_suffix('s')?;
    let repo_type: RepoType = type_singular.parse().ok()?;
    let repo_id = rest.replace("--", "/");
    Some((repo_type, repo_id))
}

async fn read_commit_refs(repo_path: &Path) -> HashMap<String, Vec<String>> {
    let mut commit_refs: HashMap<String, Vec<String>> = HashMap::new();
    let refs_dir = repo_path.join("refs");
    let mut stack = vec![refs_dir.clone()];
    while let Some(dir) = stack.pop() {
        if let Ok(mut ref_entries) = tokio::fs::read_dir(&dir).await {
            while let Ok(Some(ref_entry)) = ref_entries.next_entry().await {
                let entry_path = ref_entry.path();
                if entry_path.is_dir() {
                    stack.push(entry_path);
                } else if entry_path.is_file()
                    && let Ok(content) = tokio::fs::read_to_string(&entry_path).await
                {
                    let commit = content.trim().to_string();
                    let name = entry_path
                        .strip_prefix(&refs_dir)
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| ref_entry.file_name().to_string_lossy().to_string());
                    commit_refs.entry(commit).or_default().push(name);
                }
            }
        }
    }
    commit_refs
}

struct BlobInfo {
    blob_path: PathBuf,
    size: u64,
    accessed: SystemTime,
    modified: SystemTime,
}

async fn resolve_blob_info(file_path: &Path) -> std::result::Result<BlobInfo, String> {
    let resolved = tokio::fs::canonicalize(file_path)
        .await
        .map_err(|e| format!("Cannot resolve {}: {}", file_path.display(), e))?;
    let meta = tokio::fs::metadata(&resolved)
        .await
        .map_err(|e| format!("Cannot read blob for {}: {}", file_path.display(), e))?;
    Ok(BlobInfo {
        blob_path: resolved,
        size: meta.len(),
        accessed: meta.accessed().unwrap_or(SystemTime::UNIX_EPOCH),
        modified: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
    })
}

async fn scan_snapshot(snap_path: &Path, warnings: &mut Vec<String>) -> Vec<CachedFileInfo> {
    let mut files = Vec::new();
    let mut stack = vec![snap_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(mut dir_entries) = tokio::fs::read_dir(&dir).await {
            while let Ok(Some(file_entry)) = dir_entries.next_entry().await {
                let file_path = file_entry.path();
                if file_path.is_dir() {
                    stack.push(file_path);
                    continue;
                }

                let file_name = file_path
                    .strip_prefix(snap_path)
                    .unwrap_or(&file_path)
                    .to_string_lossy()
                    .to_string();

                let blob = match resolve_blob_info(&file_path).await {
                    Ok(b) => b,
                    Err(msg) => {
                        warnings.push(msg);
                        continue;
                    },
                };

                files.push(CachedFileInfo {
                    file_name,
                    file_path: file_path.clone(),
                    blob_path: blob.blob_path,
                    size_on_disk: blob.size,
                    blob_last_accessed: blob.accessed,
                    blob_last_modified: blob.modified,
                });
            }
        }
    }
    files
}

pub async fn scan_cache_dir(cache_dir: &Path) -> crate::error::Result<HFCacheInfo> {
    let mut repos = Vec::new();
    let mut warnings = Vec::new();
    let mut total_size: u64 = 0;

    let mut entries = match tokio::fs::read_dir(cache_dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(HFCacheInfo {
                cache_dir: cache_dir.to_path_buf(),
                repos: vec![],
                size_on_disk: 0,
                warnings: vec![],
            });
        },
        Err(e) => return Err(e.into()),
    };

    while let Some(entry) = entries.next_entry().await? {
        let folder_name = entry.file_name().to_string_lossy().to_string();
        let (repo_type, repo_id) = match parse_repo_folder_name(&folder_name) {
            Some(v) => v,
            None => continue,
        };

        let repo_path = entry.path();
        let commit_refs = read_commit_refs(&repo_path).await;

        let mut revisions = Vec::new();
        let mut repo_last_accessed = SystemTime::UNIX_EPOCH;
        let mut repo_last_modified = SystemTime::UNIX_EPOCH;

        let snapshots_dir = repo_path.join("snapshots");
        if let Ok(mut snap_entries) = tokio::fs::read_dir(&snapshots_dir).await {
            while let Ok(Some(snap_entry)) = snap_entries.next_entry().await {
                let snap_path = snap_entry.path();
                if !snap_path.is_dir() {
                    continue;
                }

                let commit_hash = snap_entry.file_name().to_string_lossy().to_string();
                let files = scan_snapshot(&snap_path, &mut warnings).await;

                let rev_size: u64 = files.iter().map(|f| f.size_on_disk).sum();
                let rev_last_modified = files
                    .iter()
                    .map(|f| f.blob_last_modified)
                    .max()
                    .unwrap_or(SystemTime::UNIX_EPOCH);

                for f in &files {
                    if f.blob_last_accessed > repo_last_accessed {
                        repo_last_accessed = f.blob_last_accessed;
                    }
                    if f.blob_last_modified > repo_last_modified {
                        repo_last_modified = f.blob_last_modified;
                    }
                }

                let refs = commit_refs.get(&commit_hash).cloned().unwrap_or_default();

                revisions.push(CachedRevisionInfo {
                    commit_hash,
                    snapshot_path: snap_path,
                    files,
                    size_on_disk: rev_size,
                    refs,
                    last_modified: rev_last_modified,
                });
            }
        }

        let mut unique_blobs: std::collections::HashMap<std::path::PathBuf, u64> = std::collections::HashMap::new();
        for rev in &revisions {
            for f in &rev.files {
                unique_blobs.entry(f.blob_path.clone()).or_insert(f.size_on_disk);
            }
        }
        let repo_size: u64 = unique_blobs.values().sum();

        total_size += repo_size;
        repos.push(CachedRepoInfo {
            repo_id,
            repo_type,
            repo_path,
            revisions,
            nb_files: unique_blobs.len(),
            size_on_disk: repo_size,
            last_accessed: repo_last_accessed,
            last_modified: repo_last_modified,
        });
    }

    Ok(HFCacheInfo {
        cache_dir: cache_dir.to_path_buf(),
        repos,
        size_on_disk: total_size,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        acquire_lock, blob_path, create_pointer_symlink, is_commit_hash, lock_path, no_exist_path,
        parse_repo_folder_name, read_commit_refs, read_ref, ref_path, repo_folder_name, scan_cache_dir, snapshot_path,
        write_ref,
    };
    use crate::types::RepoType;

    #[test]
    fn test_repo_folder_name_model_with_org() {
        assert_eq!(
            repo_folder_name("google/bert-base-uncased", Some(RepoType::Model)),
            "models--google--bert-base-uncased"
        );
    }

    #[test]
    fn test_repo_folder_name_model_no_org() {
        assert_eq!(repo_folder_name("gpt2", Some(RepoType::Model)), "models--gpt2");
    }

    #[test]
    fn test_repo_folder_name_model_none_type() {
        assert_eq!(repo_folder_name("gpt2", None), "models--gpt2");
    }

    #[test]
    fn test_repo_folder_name_dataset() {
        assert_eq!(repo_folder_name("rajpurkar/squad", Some(RepoType::Dataset)), "datasets--rajpurkar--squad");
    }

    #[test]
    fn test_repo_folder_name_space() {
        assert_eq!(repo_folder_name("dalle-mini/dalle-mini", Some(RepoType::Space)), "spaces--dalle-mini--dalle-mini");
    }

    #[test]
    fn test_blob_path() {
        let cache = Path::new("/home/user/.cache/huggingface/hub");
        assert_eq!(
            blob_path(cache, "models--gpt2", "abc123"),
            PathBuf::from("/home/user/.cache/huggingface/hub/models--gpt2/blobs/abc123")
        );
    }

    #[test]
    fn test_snapshot_path() {
        assert_eq!(
            snapshot_path(Path::new("/cache"), "models--gpt2", "aaa111", "config.json"),
            PathBuf::from("/cache/models--gpt2/snapshots/aaa111/config.json")
        );
    }

    #[test]
    fn test_snapshot_path_nested_file() {
        assert_eq!(
            snapshot_path(Path::new("/cache"), "models--gpt2", "aaa111", "subdir/model.bin"),
            PathBuf::from("/cache/models--gpt2/snapshots/aaa111/subdir/model.bin")
        );
    }

    #[test]
    fn test_ref_path() {
        assert_eq!(
            ref_path(Path::new("/cache"), "models--gpt2", "main"),
            PathBuf::from("/cache/models--gpt2/refs/main")
        );
    }

    #[test]
    fn test_ref_path_pr() {
        assert_eq!(
            ref_path(Path::new("/cache"), "models--gpt2", "refs/pr/1"),
            PathBuf::from("/cache/models--gpt2/refs/refs/pr/1")
        );
    }

    #[test]
    fn test_lock_path() {
        assert_eq!(
            lock_path(Path::new("/cache"), "models--gpt2", "abc123"),
            PathBuf::from("/cache/.locks/models--gpt2/abc123.lock")
        );
    }

    #[test]
    fn test_no_exist_path() {
        assert_eq!(
            no_exist_path(Path::new("/cache"), "models--gpt2", "aaa111", "missing.json"),
            PathBuf::from("/cache/models--gpt2/.no_exist/aaa111/missing.json")
        );
    }

    #[tokio::test]
    async fn test_write_and_read_ref() {
        let dir = tempfile::tempdir().unwrap();
        write_ref(dir.path(), "models--gpt2", "main", "abc123def456abc123def456abc123def456abcd")
            .await
            .unwrap();
        let hash = read_ref(dir.path(), "models--gpt2", "main").await.unwrap();
        assert_eq!(hash, Some("abc123def456abc123def456abc123def456abcd".to_string()));
    }

    #[tokio::test]
    async fn test_read_ref_missing() {
        let dir = tempfile::tempdir().unwrap();
        let hash = read_ref(dir.path(), "models--gpt2", "main").await.unwrap();
        assert_eq!(hash, None);
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_create_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        let blob = blob_path(cache, "models--gpt2", "abc123");
        tokio::fs::create_dir_all(blob.parent().unwrap()).await.unwrap();
        tokio::fs::write(&blob, b"file content").await.unwrap();
        create_pointer_symlink(cache, "models--gpt2", "def456", "config.json", "abc123")
            .await
            .unwrap();
        let pointer = snapshot_path(cache, "models--gpt2", "def456", "config.json");
        assert!(pointer.exists());
        assert!(pointer.symlink_metadata().unwrap().file_type().is_symlink());
        let content = tokio::fs::read_to_string(&pointer).await.unwrap();
        assert_eq!(content, "file content");
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_create_symlink_nested_file() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        let blob = blob_path(cache, "models--gpt2", "abc123");
        tokio::fs::create_dir_all(blob.parent().unwrap()).await.unwrap();
        tokio::fs::write(&blob, b"nested content").await.unwrap();
        create_pointer_symlink(cache, "models--gpt2", "def456", "subdir/model.bin", "abc123")
            .await
            .unwrap();
        let pointer = snapshot_path(cache, "models--gpt2", "def456", "subdir/model.bin");
        assert!(pointer.exists());
        assert!(pointer.symlink_metadata().unwrap().file_type().is_symlink());
        let target = std::fs::read_link(&pointer).unwrap();
        assert!(target.to_string_lossy().contains("blobs"));
        let content = tokio::fs::read_to_string(&pointer).await.unwrap();
        assert_eq!(content, "nested content");
    }

    #[test]
    fn test_is_commit_hash() {
        assert!(is_commit_hash("abc123def456abc123def456abc123def456abcd"));
        assert!(!is_commit_hash("main"));
        assert!(!is_commit_hash("abc123"));
        assert!(!is_commit_hash("xyz123def456abc123def456abc123def456abcd"));
    }

    #[tokio::test]
    async fn test_acquire_and_release_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = acquire_lock(dir.path(), "models--gpt2", "abc123").await.unwrap();
        let lock_file_path = lock_path(dir.path(), "models--gpt2", "abc123");
        assert!(lock_file_path.exists());
        drop(lock);
    }

    #[test]
    fn test_parse_repo_folder_name_model() {
        assert_eq!(parse_repo_folder_name("models--gpt2"), Some((RepoType::Model, "gpt2".to_string())));
    }

    #[test]
    fn test_parse_repo_folder_name_model_with_org() {
        assert_eq!(parse_repo_folder_name("models--google--bert"), Some((RepoType::Model, "google/bert".to_string())));
    }

    #[test]
    fn test_parse_repo_folder_name_dataset() {
        assert_eq!(parse_repo_folder_name("datasets--squad"), Some((RepoType::Dataset, "squad".to_string())));
    }

    #[test]
    fn test_parse_repo_folder_name_invalid() {
        assert_eq!(parse_repo_folder_name(".locks"), None);
    }

    #[tokio::test]
    async fn test_scan_cache_empty() {
        let dir = tempfile::tempdir().unwrap();
        let result = scan_cache_dir(dir.path()).await.unwrap();
        assert_eq!(result.repos.len(), 0);
        assert_eq!(result.size_on_disk, 0);
    }

    #[tokio::test]
    async fn test_scan_cache_nonexistent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does_not_exist");
        let result = scan_cache_dir(&nonexistent).await.unwrap();
        assert_eq!(result.repos.len(), 0);
        assert_eq!(result.size_on_disk, 0);
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_scan_cache_with_repo() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        let repo_folder = "models--gpt2";
        let blob_dir = cache.join(repo_folder).join("blobs");
        tokio::fs::create_dir_all(&blob_dir).await.unwrap();
        tokio::fs::write(blob_dir.join("abc123"), b"hello world").await.unwrap();

        let snap_dir = cache.join(repo_folder).join("snapshots").join("commit1");
        tokio::fs::create_dir_all(&snap_dir).await.unwrap();
        tokio::fs::symlink("../../blobs/abc123", snap_dir.join("file.txt"))
            .await
            .unwrap();

        let refs_dir = cache.join(repo_folder).join("refs");
        tokio::fs::create_dir_all(&refs_dir).await.unwrap();
        tokio::fs::write(refs_dir.join("main"), "commit1").await.unwrap();

        let result = scan_cache_dir(cache).await.unwrap();
        assert_eq!(result.repos.len(), 1);
        assert_eq!(result.repos[0].repo_id, "gpt2");
        assert_eq!(result.repos[0].repo_type, RepoType::Model);
        assert_eq!(result.repos[0].revisions.len(), 1);
        assert_eq!(result.repos[0].revisions[0].refs, vec!["main"]);
        assert_eq!(result.repos[0].revisions[0].files.len(), 1);
        assert_eq!(result.repos[0].revisions[0].size_on_disk, 11);
        assert_eq!(result.repos[0].size_on_disk, 11);
        assert_eq!(result.size_on_disk, 11);
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_read_commit_refs_nested() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("models--gpt2");
        let refs_dir = repo_path.join("refs");

        // Create a regular ref
        tokio::fs::create_dir_all(&refs_dir).await.unwrap();
        tokio::fs::write(refs_dir.join("main"), "commit1").await.unwrap();

        // Create a nested PR ref
        let pr_dir = refs_dir.join("refs").join("pr");
        tokio::fs::create_dir_all(&pr_dir).await.unwrap();
        tokio::fs::write(pr_dir.join("1"), "commit2").await.unwrap();

        let refs = read_commit_refs(&repo_path).await;
        assert_eq!(refs.get("commit1").unwrap(), &vec!["main".to_string()]);
        assert_eq!(refs.get("commit2").unwrap(), &vec!["refs/pr/1".to_string()]);
    }

    #[cfg(not(windows))]
    #[tokio::test]
    async fn test_scan_cache_deduplicates_blob_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        let repo_folder = "models--gpt2";

        let blob_dir = cache.join(repo_folder).join("blobs");
        tokio::fs::create_dir_all(&blob_dir).await.unwrap();
        let blob_content = vec![0u8; 100];
        tokio::fs::write(blob_dir.join("shared_blob"), &blob_content).await.unwrap();

        let snap1 = cache.join(repo_folder).join("snapshots").join("commit1");
        let snap2 = cache.join(repo_folder).join("snapshots").join("commit2");
        tokio::fs::create_dir_all(&snap1).await.unwrap();
        tokio::fs::create_dir_all(&snap2).await.unwrap();
        tokio::fs::symlink("../../blobs/shared_blob", snap1.join("file.txt"))
            .await
            .unwrap();
        tokio::fs::symlink("../../blobs/shared_blob", snap2.join("file.txt"))
            .await
            .unwrap();

        let result = scan_cache_dir(cache).await.unwrap();
        assert_eq!(result.repos.len(), 1);
        let rev_sizes: Vec<u64> = result.repos[0].revisions.iter().map(|r| r.size_on_disk).collect();
        assert!(rev_sizes.iter().all(|&s| s == 100));
        assert_eq!(result.repos[0].size_on_disk, 100);
        assert_eq!(result.size_on_disk, 100);
    }
}
