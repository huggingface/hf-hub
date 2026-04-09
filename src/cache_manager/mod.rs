//! Cache Manager
//!
//! Based on <https://github.com/huggingface/huggingface_hub/blob/c5c9cd2624fd3f051b113cc2e020f221c79eef05/src/huggingface_hub/utils/_cache_manager.py>

#![allow(missing_docs)]

use std::{
    collections::{hash_map::Entry, BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use thiserror::Error;
use walkdir::WalkDir;

#[cfg(feature = "cache-manager-display")]
use comfy_table::Table;

use crate::{
    paths::{get_hub_dir, get_ref_path, is_locks_dir, refs_dir, snapshots_dir},
    Cache, Repo,
};

pub mod fs_utils;
// TODO: use https://github.com/BurntSushi/ripgrep/tree/master/crates/ignore ?
use fs_utils::is_ignored;

pub mod formatting_parsing;
use formatting_parsing::{
    format_size, format_timesince, parse_repo_folder_name, RepoFolderParseError,
};

#[cfg(feature = "cache-manager-display")]
const HF_CACHE_INFO_EXPORT_TABLE_HEADERS: [&str; 8] = [
    "REPO ID",
    "REPO TYPE",
    "SIZE ON DISK",
    "NB FILES",
    "LAST_ACCESSED",
    "LAST_MODIFIED",
    "REFS",
    "LOCAL PATH",
];

/// Represents a specific corruption issue found within a repository.
/// This corresponds to Python's `CorruptedCacheException`.
#[derive(Debug, Error)]
pub enum CorruptedCacheError {
    /// The cache directory is missing.
    #[error("Cache directory not found: {path:?}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.")]
    MissingCacheDir { path: PathBuf },

    /// The snapshots directory cannot be a file.
    #[error("Scan cache expects a directory but found a file: {path:?}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.")]
    CacheDirCantBeFile { path: PathBuf },

    /// The repo folder isn't a dir.
    #[error("Path not a directory at {path:?}")]
    RepoNotDir { path: PathBuf },

    /// The repo folder name doesn't match the `type--user--id` pattern.
    #[error("Invalid repository name: {path:?}")]
    InvalidRepoName { path: PathBuf },

    /// Invalid Repo type.
    #[error("Invalid repo type '{invalid_repo_type}' in path {path:?}")]
    InvalidRepoType {
        path: PathBuf,
        invalid_repo_type: String,
    },

    /// The snapshots directory is missing.
    #[error("Missing snapshots directory at {path:?}")]
    MissingSnapshotsDir { path: PathBuf },

    /// The snapshots directory cannot be a file.
    #[error("Snapshots directory corrupted. Cannot be a file: {path:?}")]
    SnapshotDirCantBeFile { path: PathBuf },

    /// The refs directory cannot be a file.
    #[error("Refs directory cannot be a file: {path:?}")]
    RefsDirCantBeFile { path: PathBuf },

    /// A symlink points to a blob that does not exist.
    #[error("Blob missing (broken symlink): {blob_path:?}")]
    BlobMissing { blob_path: PathBuf },

    /// Invalid Repo type.
    #[error("Reference(s) refer to missing commit hashes: {refs_by_hash:?} ({repo_path:?})")]
    ReferenceMissingCommitHash {
        refs_by_hash: HashMap<String, BTreeSet<String>>,
        repo_path: PathBuf,
    },

    #[error("Invalid or missing file name at {path:?}")]
    InvalidFileName { path: PathBuf },

    /// An unexpected IO error occurred while reading this specific repo
    /// (e.g., permission denied on one folder).
    #[error("IO Error at {path:?}: {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl From<walkdir::Error> for CorruptedCacheError {
    fn from(e: walkdir::Error) -> Self {
        let path = e.path().unwrap_or_else(|| Path::new("")).to_path_buf();

        let source = e
            .into_io_error()
            .unwrap_or_else(|| std::io::Error::from(std::io::ErrorKind::Other));

        CorruptedCacheError::IoError { path, source }
    }
}

/// Holds information about a single cached file.
///
/// > [!WARNING] TODO: confirm if this is true for rust
/// > `blob_last_accessed` and `blob_last_modified` reliability can depend on the OS you
/// > are using. See [python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
/// > for more details.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct CachedFileInfo {
    /// Name of the file. Example: `config.json`.
    pub file_name: String,
    /// Path of the file in the `snapshots` directory. The file path is a symlink
    /// referring to a blob in the `blobs` folder.
    pub file_path: PathBuf,
    /// Path of the blob file. This is equivalent to `file_path.resolve()` from Python.
    pub blob_path: PathBuf,
    /// Size of the blob file in bytes.
    pub size_on_disk: u64,
    ///  Timestamp of the last time the blob file has been accessed (from any revision).
    pub blob_last_accessed: Option<SystemTime>,
    /// Timestamp of the last time the blob file has been modified/created.
    pub blob_last_modified: Option<SystemTime>,
}

impl CachedFileInfo {
    /// Timestamp of the last time the blob file has been accessed (from any
    /// revision), returned as a human-readable string.
    ///
    /// Example: "2 weeks ago".
    pub fn blob_last_accessed_str(&self) -> String {
        match self.blob_last_accessed {
            Some(time) => format_timesince(time),
            None => "Unknown".to_string(),
        }
    }

    /// Timestamp of the last time the blob file has been modified, returned
    /// as a human-readable string.
    ///
    /// Example: "2 weeks ago".
    pub fn blob_last_modified_str(&self) -> String {
        match self.blob_last_modified {
            Some(time) => format_timesince(time),
            None => "Unknown".to_string(),
        }
    }

    /// Size of the blob file as a human-readable string.
    ///
    /// Example: "42.2K".
    pub fn size_on_disk_str(&self) -> String {
        format_size(self.size_on_disk)
    }
}

/// Holds information about a revision.
///
/// A revision correspond to a folder in the `snapshots` folder and is populated with
/// the exact tree structure as the repo on the Hub but contains only symlinks. A
/// revision can be either referenced by 1 or more `refs` or be "detached" (no refs).
///
/// > [!WARNING]
/// > `last_accessed` cannot be determined correctly on a single revision as blob files
/// > are shared across revisions.
///
/// > [!WARNING]
/// > `size_on_disk` is not necessarily the sum of all file sizes because of possible
/// > duplicated files. Besides, only blobs are taken into account, not the (negligible)
/// > size of folders and symlinks.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct CachedRevisionInfo {
    /// Hash of the revision (unique).
    /// Example: `"9338f7b671827df886678df2bdd7cc7b4f36dffd"`.
    pub commit_hash: String,
    /// Path to the revision directory in the `snapshots` folder. It contains the
    /// exact tree structure as the repo on the Hub.
    pub snapshot_path: PathBuf,
    /// "Set" of [`~CachedFileInfo`] describing all files contained in the snapshot.
    pub files: BTreeSet<CachedFileInfo>,
    /// "Set" of `refs` pointing to this revision. If the revision has no `refs`, it
    /// is considered detached.
    /// Example: `{"main", "2.4.0"}` or `{"refs/pr/1"}`.
    pub refs: BTreeSet<String>,
    /// Sum of the blob file sizes that are symlink-ed by the revision.
    pub size_on_disk: u64,
    /// Timestamp of the last time the revision has been created/modified.
    pub last_modified: Option<SystemTime>,
}

impl CachedRevisionInfo {
    /// Timestamp of the last time the blob file has been modified, returned
    /// as a human-readable string.
    ///
    /// Example: "2 weeks ago".
    pub fn last_modified_str(&self) -> String {
        match self.last_modified {
            Some(time) => format_timesince(time),
            None => "Unknown".to_string(),
        }
    }

    /// Size of the blob file as a human-readable string.
    ///
    /// Example: "42.2K".
    pub fn size_on_disk_str(&self) -> String {
        format_size(self.size_on_disk)
    }

    /// Total number of files in the revision.
    pub fn nb_files(&self) -> usize {
        self.files.len()
    }
}

/// Holds information about a cached repository.
///
/// > [!WARNING]
/// > `size_on_disk` is not necessarily the sum of all revisions sizes because of
/// > duplicated files. Besides, only blobs are taken into account, not the (negligible)
/// > size of folders and symlinks
///
/// > [!WARNING]
/// > `last_accessed` and `last_modified` reliability can depend on the OS you are using.
/// > See [python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
/// > for more details.
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct CachedRepoInfo {
    /// Repo struct that holds information like id, type and path
    pub repo: Repo,
    /// Local path to the cached repo.
    pub repo_path: PathBuf,
    /// Sum of the blob file sizes in the cached repo.
    pub size_on_disk: u64,
    /// Total number of blob files in the cached repo.
    pub nb_files: usize,
    /// "Set" of [`~CachedRevisionInfo`] describing all revisions cached in the repo.
    pub revisions: BTreeSet<CachedRevisionInfo>,
    /// Timestamp of the last time a blob file of the repo has been accessed.
    pub last_accessed: Option<SystemTime>,
    /// Timestamp of the last time a blob file of the repo has been modified/created.
    pub last_modified: Option<SystemTime>,
}

impl CachedRepoInfo {
    /// Timestamp of the last time the blob file has been accessed (from any
    /// revision), returned as a human-readable string.
    ///
    /// Example: "2 weeks ago".
    pub fn last_accessed_str(&self) -> String {
        match self.last_accessed {
            Some(time) => format_timesince(time),
            None => "Unknown".to_string(),
        }
    }

    /// Timestamp of the last time the blob file has been modified, returned
    /// as a human-readable string.
    ///
    /// Example: "2 weeks ago".
    pub fn last_modified_str(&self) -> String {
        match self.last_modified {
            Some(time) => format_timesince(time),
            None => "Unknown".to_string(),
        }
    }

    /// Size of the blob file as a human-readable string.
    ///
    /// Example: "42.2K".
    pub fn size_on_disk_str(&self) -> String {
        format_size(self.size_on_disk)
    }

    /// Canonical `type/id` identifier used across cache tooling.
    pub fn cache_id(&self) -> String {
        format!("{}/{}", self.repo.repo_type, self.repo.repo_id)
    }

    /// Mapping between `refs` and revision data structures.
    pub fn refs(&self) -> BTreeMap<&str, &CachedRevisionInfo> {
        let mut refs = BTreeMap::new();

        for revision in self.revisions.iter() {
            for revision_ref in revision.refs.iter() {
                refs.insert(revision_ref.as_str(), revision);
            }
        }

        refs
    }

    pub(crate) fn scan_cached_repo(repo_path: &Path) -> Result<Self, CorruptedCacheError> {
        if !repo_path.is_dir() {
            return Err(CorruptedCacheError::RepoNotDir {
                path: repo_path.to_path_buf(),
            });
        }

        let repo_folder_name = repo_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| CorruptedCacheError::InvalidRepoName {
                path: repo_path.to_path_buf(),
            })?;

        let (repo_type, repo_id) =
            parse_repo_folder_name(repo_folder_name).map_err(|e| match e {
                RepoFolderParseError::MissingSeparator => CorruptedCacheError::InvalidRepoName {
                    path: repo_path.to_path_buf(),
                },
                RepoFolderParseError::InvalidType(invalid_type) => {
                    CorruptedCacheError::InvalidRepoType {
                        path: repo_path.to_path_buf(),
                        invalid_repo_type: invalid_type,
                    }
                }
            })?;

        let repo = Repo::new(repo_id, repo_type);

        let mut blob_stats = HashMap::new();
        let snapshots_path = snapshots_dir(repo_path);

        if !snapshots_path.is_dir() {
            return Err(CorruptedCacheError::MissingSnapshotsDir {
                path: snapshots_path.to_path_buf(),
            });
        }

        let refs_path = refs_dir(repo_path);
        let mut refs_by_hash: HashMap<String, BTreeSet<String>> = HashMap::new();

        if refs_path.exists() {
            if !refs_path.is_dir() {
                return Err(CorruptedCacheError::RefsDirCantBeFile {
                    path: refs_path.to_path_buf(),
                });
            }

            let refs_walker = WalkDir::new(&refs_path).min_depth(1);

            for refs_entry_result in refs_walker
                .into_iter()
                .filter_entry(|entry| !is_ignored(entry))
            {
                let refs_entry = refs_entry_result?;

                if refs_entry.file_type().is_dir() {
                    continue;
                }

                let ref_path = refs_entry.path();

                // TODO: windows handling?
                // python: Path(file_path).resolve()
                // rust dunce::canonicalize ?
                let ref_name = ref_path
                    .strip_prefix(&refs_path)
                    .expect(
                        "strip_prefix should always work since ref_path was found inside refs_path",
                    )
                    .to_string_lossy()
                    .replace("\\", "/");

                let commit_hash =
                    fs::read_to_string(ref_path).map_err(|e| CorruptedCacheError::IoError {
                        path: ref_path.to_path_buf(),
                        source: e,
                    })?;

                refs_by_hash
                    .entry(commit_hash.trim().to_string())
                    .or_default()
                    .insert(ref_name);
            }
        }

        let mut cached_revisions = BTreeSet::new();

        let cached_revisions_walker = WalkDir::new(&snapshots_path).min_depth(1).max_depth(1);

        for cached_revision_entry_result in cached_revisions_walker
            .into_iter()
            .filter_entry(|entry| !is_ignored(entry))
        {
            let cached_revision_entry = cached_revision_entry_result?;
            let cached_revision_dir_path = cached_revision_entry.path();

            if cached_revision_entry.file_type().is_file() {
                return Err(CorruptedCacheError::SnapshotDirCantBeFile {
                    path: cached_revision_dir_path.to_path_buf(),
                });
            }

            let mut cached_files = BTreeSet::new();

            let revision_file_walker = WalkDir::new(cached_revision_dir_path).min_depth(1);

            for revision_file_entry_result in revision_file_walker
                .into_iter()
                .filter_entry(|entry| !is_ignored(entry))
            {
                let revision_file_entry = revision_file_entry_result?;

                if revision_file_entry.file_type().is_dir() {
                    continue;
                }

                let revision_file_entry_path = revision_file_entry.path();

                let blob_path = fs::canonicalize(revision_file_entry_path).map_err(|_| {
                    CorruptedCacheError::BlobMissing {
                        blob_path: revision_file_entry_path.to_path_buf(),
                    }
                })?;

                if !blob_path.exists() {
                    return Err(CorruptedCacheError::BlobMissing {
                        blob_path: blob_path.to_path_buf(),
                    });
                }

                let blob_path = blob_path.to_path_buf();

                // Using hash map Entry so we can insert the blob meta if it doesn't exist
                // BUT handle error if the metadata() call somehow fails (and don't just use .expect)
                let stat = match blob_stats.entry(blob_path.clone()) {
                    Entry::Occupied(blob_stats_entry) => blob_stats_entry.into_mut(),
                    Entry::Vacant(blob_stats_entry) => {
                        let meta =
                            blob_path
                                .metadata()
                                .map_err(|e| CorruptedCacheError::IoError {
                                    path: blob_path.clone(),
                                    source: e,
                                })?;

                        blob_stats_entry.insert(meta)
                    }
                };

                // Handle possible weird paths
                let blob_file_name = revision_file_entry_path
                    .file_name()
                    .ok_or_else(|| CorruptedCacheError::InvalidFileName {
                        path: revision_file_entry_path.to_path_buf(),
                    })?
                    .to_string_lossy()
                    .into_owned();

                cached_files.insert(CachedFileInfo {
                    file_name: blob_file_name,
                    file_path: revision_file_entry_path.to_path_buf(),
                    blob_path,
                    size_on_disk: stat.len(),
                    blob_last_accessed: stat.accessed().ok(),
                    blob_last_modified: stat.modified().ok(),
                });
            }

            // Last modified is either the last modified blob file or the revision folder
            // itself if it is empty
            let revision_last_modified = if !cached_files.is_empty() {
                cached_files
                    .iter()
                    .filter_map(|file| file.blob_last_modified)
                    .max()
            } else {
                cached_revision_dir_path
                    .metadata()
                    .ok()
                    .and_then(|metadata| metadata.modified().ok())
            };

            // Dedupe based on blob path, since different files / copies could symlink
            // and thus not be "unique" in btree
            let unique_blobs: BTreeSet<&PathBuf> =
                cached_files.iter().map(|file| &file.blob_path).collect();

            let revision_size_on_disk = unique_blobs
                .iter()
                .filter_map(|blob_stat| blob_stats.get(*blob_stat).map(|stat| stat.len()))
                .sum();

            let commit_hash = cached_revision_entry
                .file_name()
                .to_string_lossy()
                .into_owned();

            let refs = refs_by_hash
                .remove(&commit_hash)
                .unwrap_or_default()
                .into_iter()
                .collect();

            cached_revisions.insert(CachedRevisionInfo {
                commit_hash,
                snapshot_path: cached_revision_dir_path.to_path_buf(),
                size_on_disk: revision_size_on_disk,
                files: cached_files,
                refs,
                last_modified: revision_last_modified,
            });
        }

        if !refs_by_hash.is_empty() {
            // Check that all refs referred to an existing revision
            return Err(CorruptedCacheError::ReferenceMissingCommitHash {
                refs_by_hash,
                repo_path: repo_path.to_path_buf(),
            });
        }

        let (repo_last_accessed, repo_last_modified) = if !blob_stats.is_empty() {
            let accessed = blob_stats
                .values()
                .filter_map(|metadata| metadata.accessed().ok())
                .max();

            let modified = blob_stats
                .values()
                .filter_map(|metadata| metadata.modified().ok())
                .max();

            (accessed, modified)
        } else {
            let repo_stats = repo_path.metadata().ok();

            let accessed = repo_stats
                .as_ref()
                .and_then(|metadata| metadata.accessed().ok());

            let modified = repo_stats
                .as_ref()
                .and_then(|metadata| metadata.modified().ok());

            (accessed, modified)
        };

        let size_on_disk = blob_stats.values().map(|metadata| metadata.len()).sum();

        Ok(Self {
            repo,
            repo_path: repo_path.to_owned(),
            size_on_disk,
            nb_files: blob_stats.len(),
            revisions: cached_revisions,
            last_accessed: repo_last_accessed,
            last_modified: repo_last_modified,
        })
    }
}

/// Holds the strategy to delete cached revisions.
///
/// This object is not meant to be instantiated programmatically but to be returned by
/// [`HFCacheInfo.delete_revisions`]. See documentation for usage example.
#[derive(Debug)]
pub struct DeleteCacheStrategy {
    /// Expected freed size once strategy is executed.
    pub expected_freed_size: u64,
    /// Set of blob file paths to be deleted.
    pub blobs: BTreeSet<PathBuf>,
    /// Set of reference file paths to be deleted.
    pub refs: BTreeSet<PathBuf>,
    /// Set of entire repo paths to be deleted.
    pub repos: BTreeSet<PathBuf>,
    /// Set of snapshots to be deleted (directory of symlinks).
    pub snapshots: BTreeSet<PathBuf>,
}

impl DeleteCacheStrategy {
    /// Expected size that will be freed as a human-readable string.
    ///
    /// Example: "42.2K".
    pub fn expected_freed_size_str(&self) -> String {
        format_size(self.expected_freed_size)
    }

    /// Execute the defined strategy.
    ///
    /// > [!WARNING]
    /// > If this method is interrupted, the cache might get corrupted. Deletion order is
    /// > implemented so that references and symlinks are deleted before the actual blob
    /// > files.
    ///
    /// > [!WARNING]
    /// > This method is irreversible. If executed, cached files are erased and must be
    /// > downloaded again.
    pub fn execute(&self) {
        // Deletion order matters. Blobs are deleted in last so that the user can't end
        // up in a state where a `ref`` refers to a missing snapshot or a snapshot
        // symlink refers to a deleted blob.

        let paths_to_delete = [
            (&self.repos, "repo"),
            (&self.snapshots, "snapshot"),
            (&self.refs, "ref"),
            (&self.blobs, "blob"),
        ];

        for (paths, label) in paths_to_delete {
            for path in paths.iter() {
                try_delete_path(path, label);
            }
        }

        log::info!(
            "Cache deletion done. Saved {}.",
            self.expected_freed_size_str()
        );
    }
}

/// Holds information about the entire cache-system.
///
/// > [!WARNING]
/// > Here `size_on_disk` is equal to the sum of all repo sizes (only blobs). However if
/// > some cached repos are corrupted, their sizes are not taken into account.
#[derive(Debug)]
pub struct HFCacheInfo {
    /// Sum of all valid repo sizes in the cache-system.
    pub size_on_disk: u64,
    /// "Set" of [`~CachedRepoInfo`] describing all valid cached repos found on the
    /// cache-system while scanning.
    pub repos: BTreeSet<CachedRepoInfo>,
    /// List of [`~CorruptedCacheException`] that occurred while scanning the cache.
    /// Those exceptions (errors) are captured so that the scan can continue. Corrupted repos
    /// are skipped from the scan.
    pub warnings: Vec<CorruptedCacheError>,
    /// New for Rust, Cache struct
    pub cache: Cache,
}

impl HFCacheInfo {
    /// Sum of all valid repo sizes in the cache-system as a human-readable string.
    ///
    /// Example: "42.2K".
    pub fn size_on_disk_str(&self) -> String {
        format_size(self.size_on_disk)
    }

    /// Prepare the strategy to delete one or more revisions cached locally.
    ///
    /// Input revisions can be any revision hash. If a revision hash is not found in the
    /// local cache, a warning is thrown but no error is raised. Revisions can be from
    /// different cached repos since hashes are unique across repos,
    ///
    /// TODO: examples docs
    ///
    /// > [!WARNING]
    /// > `delete_revisions` returns a [`~utils.DeleteCacheStrategy`] object that needs to
    /// > be executed. The [`~utils.DeleteCacheStrategy`] is not meant to be modified but
    /// > allows having a dry run before actually executing the deletion.
    pub fn delete_revisions(&self, revisions: &[&str]) -> DeleteCacheStrategy {
        let mut hashes_to_delete: BTreeSet<&str> = revisions.iter().copied().collect();

        let mut repos_with_revisions_to_delete: BTreeMap<
            &CachedRepoInfo,
            BTreeSet<&CachedRevisionInfo>,
        > = BTreeMap::new();

        for repo in self.repos.iter() {
            for revision in repo.revisions.iter() {
                let revision_commit_hash = revision.commit_hash.as_str();

                if hashes_to_delete.contains(revision_commit_hash) {
                    repos_with_revisions_to_delete
                        .entry(repo)
                        .or_default()
                        .insert(revision);

                    hashes_to_delete.remove(revision_commit_hash);
                }
            }
        }

        if !hashes_to_delete.is_empty() {
            log::warn!(
                "Revision(s) not found - cannot delete them: {:?}",
                hashes_to_delete
            );
        }

        let mut delete_strategy_blobs = BTreeSet::new();
        let mut delete_strategy_refs = BTreeSet::new();
        let mut delete_strategy_repos = BTreeSet::new();
        let mut delete_strategy_snapshots = BTreeSet::new();
        let mut delete_strategy_expected_freed_size = 0;

        for (affected_repo, revisions_to_delete) in repos_with_revisions_to_delete {
            let other_revisions: Vec<&CachedRevisionInfo> = affected_repo
                .revisions
                .iter() // Yields &CachedRevisionInfo
                .filter(|repo_revision| !revisions_to_delete.contains(repo_revision))
                .collect();

            // If no other revisions remain, it means all revisions are deleted
            // -> delete the entire cached repo
            if other_revisions.is_empty() {
                delete_strategy_repos.insert(affected_repo.repo_path.clone());
                delete_strategy_expected_freed_size += affected_repo.size_on_disk;
                continue;
            }

            // Some revisions of the repo will be deleted but not all. We need to filter
            // which blob files will not be linked anymore.
            for revision_to_delete in revisions_to_delete {
                // Snapshots dir
                delete_strategy_snapshots.insert(revision_to_delete.snapshot_path.clone());

                // Refs dir
                for revision_ref in &revision_to_delete.refs {
                    let ref_path = get_ref_path(&affected_repo.repo_path, revision_ref);

                    delete_strategy_refs.insert(ref_path);
                }

                // Blobs dir
                for file in &revision_to_delete.files {
                    if !delete_strategy_blobs.contains(&file.blob_path) {
                        let mut is_file_alone = true;

                        for revision in &other_revisions {
                            for revision_file in &revision.files {
                                if file.blob_path == revision_file.blob_path {
                                    is_file_alone = false;
                                    break;
                                }
                            }

                            if !is_file_alone {
                                break;
                            }
                        }

                        if is_file_alone {
                            delete_strategy_blobs.insert(file.blob_path.clone());
                            delete_strategy_expected_freed_size += file.size_on_disk;
                        }
                    }
                }
            }
        }

        DeleteCacheStrategy {
            expected_freed_size: delete_strategy_expected_freed_size,
            blobs: delete_strategy_blobs,
            refs: delete_strategy_refs,
            repos: delete_strategy_repos,
            snapshots: delete_strategy_snapshots,
        }
    }

    // /// Generate a table from the [`HFCacheInfo`] object.
    // pub fn export_as_table(&self) {
    //     todo!("`export_as_table` strictly compatible with python version not yet implemented yet");
    // }

    #[cfg(feature = "cache-manager-display")]
    /// Generate a table from the [`HFCacheInfo`] object.
    pub fn export_as_table_comfy(&self) -> String {
        let mut table = Table::new();

        table.set_header(HF_CACHE_INFO_EXPORT_TABLE_HEADERS);

        for repo in self.repos.iter() {
            let refs_str = repo.refs().keys().copied().collect::<Vec<_>>().join(", ");

            table.add_row(vec![
                repo.repo.repo_id.clone(),
                repo.repo.repo_type.to_string(),
                // Right align to 12 chars matching python's "{:>12}".format(...)
                format!("{:>12}", repo.size_on_disk_str()),
                repo.nb_files.to_string(),
                repo.last_accessed_str(),
                repo.last_modified_str(),
                refs_str,
                repo.repo_path.to_string_lossy().to_string(),
            ]);
        }

        table.to_string()
    }

    /// Scan the entire HF cache-system and return a [`~HFCacheInfo`] structure.
    ///
    /// Use `scan_cache_dir` in order to programmatically scan your cache-system. The cache
    /// will be scanned repo by repo. If a repo is corrupted, a [`~CorruptedCacheException`]
    /// will be thrown internally but captured and returned in the [`~HFCacheInfo`]
    /// structure. Only valid repos get a proper report.
    ///
    /// TODO: rest of docs
    pub fn scan_cache_dir(cache_dir: Option<&Path>) -> Result<Self, CorruptedCacheError> {
        let cache_dir = match cache_dir {
            Some(cache_dir) => cache_dir.to_path_buf(),
            None => get_hub_dir().ok_or(CorruptedCacheError::MissingCacheDir {
                path: PathBuf::new(),
            })?,
        };

        Cache::validate_cache_dir_path(&cache_dir)?;

        let mut repos = BTreeSet::new();
        let mut warnings = Vec::new();

        let cache_dir_walker = WalkDir::new(&cache_dir).min_depth(1).max_depth(1);

        for cache_dir_entry_result in cache_dir_walker
            .into_iter()
            .filter_entry(|entry| !is_ignored(entry) && !is_locks_dir(entry))
        {
            let cache_dir_entry = cache_dir_entry_result?;

            // Use a match statement instead of `?` so we can keep scanning and not
            // fail / return on first error
            match CachedRepoInfo::scan_cached_repo(cache_dir_entry.path()) {
                Ok(cached_repo) => {
                    repos.insert(cached_repo);
                }
                Err(e) => {
                    warnings.push(e);
                }
            }
        }

        let size_on_disk: u64 = repos.iter().map(|repo| repo.size_on_disk).sum();
        let cache = Cache::new(cache_dir.to_path_buf());

        Ok(Self {
            repos,
            size_on_disk,
            warnings,
            cache,
        })
    }
}

/// Try to delete a local file or folder.
///
/// If the path does not exist, error is logged as a warning and then ignored.
///
/// TODO: move under HFCacheInfo struct? DeleteCacheStrat?
fn try_delete_path(path: &Path, path_type: &str) {
    log::info!("Delete {}: {:?}", path_type, path);

    let result = if path.is_file() {
        fs::remove_file(path)
    } else {
        fs::remove_dir_all(path)
    };

    if let Err(e) = result {
        log::warn!("Couldn't delete {}: {} ({:?})", path_type, e, path);
    }
}
