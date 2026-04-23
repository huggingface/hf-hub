use std::path::PathBuf;

use serde::Serialize;
use typed_builder::TypedBuilder;

use super::commit::{AddSource, CommitOperation};
use super::progress::Progress;
use super::repo::{GatedApprovalMode, GatedNotificationsMode};

/// Parameters for fetching repository info.
///
/// Used with [`HFRepository::info`](crate::repository::HFRepository::info).
#[derive(Default, TypedBuilder)]
pub struct RepoInfoParams {
    /// Git revision (branch, tag, or commit SHA) to fetch info for. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// List of properties to expand in the response (e.g. `"trendingScore"`, `"cardData"`).
    /// When set, only the listed properties (plus `_id` and `id`) are returned.
    /// Available values vary by repo type — see the Hub API documentation.
    #[builder(default, setter(strip_option))]
    pub expand: Option<Vec<String>>,
}

/// Parameters for checking whether a revision exists in a repository.
///
/// Used with [`HFRepository::revision_exists`](crate::repository::HFRepository::revision_exists).
#[derive(TypedBuilder)]
pub struct RepoRevisionExistsParams {
    /// Git revision (branch, tag, or commit SHA) to check for existence.
    #[builder(setter(into))]
    pub revision: String,
}

/// Parameters for checking whether a file exists in a repository.
///
/// Used with [`HFRepository::file_exists`](crate::repository::HFRepository::file_exists).
#[derive(TypedBuilder)]
pub struct RepoFileExistsParams {
    /// Path of the file to check within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Git revision to check. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for listing files in a repository.
///
/// Used with [`HFRepository::list_files`](crate::repository::HFRepository::list_files).
#[derive(Default, TypedBuilder)]
pub struct RepoListFilesParams {
    /// Git revision to list files from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for listing the tree of entries in a repository.
///
/// Used with [`HFRepository::list_tree`](crate::repository::HFRepository::list_tree).
#[derive(Default, TypedBuilder)]
pub struct RepoListTreeParams {
    /// Git revision to list the tree from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Whether to list files recursively in subdirectories.
    #[builder(default)]
    pub recursive: bool,
    /// Whether to include expanded metadata (size, LFS info) for each entry.
    #[builder(default)]
    pub expand: bool,
    /// Maximum number of tree entries to return.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for fetching metadata about a single file in a repository.
///
/// Used with [`HFRepository::get_file_metadata`](crate::repository::HFRepository::get_file_metadata).
#[derive(TypedBuilder)]
pub struct RepoGetFileMetadataParams {
    /// Path of the file within the repository.
    #[builder(setter(into))]
    pub filepath: String,
    /// Git revision to query. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for fetching info about a set of paths in a repository.
///
/// Used with [`HFRepository::get_paths_info`](crate::repository::HFRepository::get_paths_info).
#[derive(TypedBuilder)]
pub struct RepoGetPathsInfoParams {
    /// List of file paths within the repository to retrieve info for.
    pub paths: Vec<String>,
    /// Git revision to query. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for downloading a single file from a repository.
///
/// Used with [`HFRepository::download_file`](crate::repository::HFRepository::download_file).
#[derive(TypedBuilder)]
pub struct RepoDownloadFileParams {
    /// Path of the file to download within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Local directory to download the file into. When set, the file is saved with its repo path structure.
    #[builder(default, setter(strip_option))]
    pub local_dir: Option<PathBuf>,
    /// Git revision to download from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// If `true`, re-download the file even if a cached copy exists.
    #[builder(default, setter(strip_option))]
    pub force_download: Option<bool>,
    /// If `true`, only return the file if it is already cached locally; never make a network request.
    #[builder(default, setter(strip_option))]
    pub local_files_only: Option<bool>,
    /// Optional progress handler for tracking download progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for streaming a file download from a repository.
///
/// Used with [`HFRepository::download_file_stream`](crate::repository::HFRepository::download_file_stream)
/// and [`HFRepository::download_file_to_bytes`](crate::repository::HFRepository::download_file_to_bytes).
#[derive(TypedBuilder)]
pub struct RepoDownloadFileStreamParams {
    /// Path of the file to stream within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Git revision to stream from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Byte range to request (HTTP Range header). Useful for partial downloads.
    #[builder(default, setter(strip_option))]
    pub range: Option<std::ops::Range<u64>>,
}

pub type RepoDownloadFileToBytesParams = RepoDownloadFileStreamParams;
pub type RepoDownloadFileToBytesParamsBuilder = RepoDownloadFileStreamParamsBuilder;

/// Parameters for downloading a full repository snapshot.
///
/// Used with [`HFRepository::snapshot_download`](crate::repository::HFRepository::snapshot_download).
#[derive(Default, TypedBuilder)]
pub struct RepoSnapshotDownloadParams {
    /// Git revision to download. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Glob patterns for files to include in the download. Only matching files are downloaded.
    #[builder(default, setter(strip_option))]
    pub allow_patterns: Option<Vec<String>>,
    /// Glob patterns for files to exclude from the download.
    #[builder(default, setter(strip_option))]
    pub ignore_patterns: Option<Vec<String>>,
    /// Local directory to download the snapshot into.
    #[builder(default, setter(strip_option))]
    pub local_dir: Option<PathBuf>,
    /// If `true`, re-download all files even if cached copies exist.
    #[builder(default, setter(strip_option))]
    pub force_download: Option<bool>,
    /// If `true`, only return files already cached locally; never make network requests.
    #[builder(default, setter(strip_option))]
    pub local_files_only: Option<bool>,
    /// Maximum number of concurrent file downloads.
    #[builder(default, setter(strip_option))]
    pub max_workers: Option<usize>,
    /// Optional progress handler for tracking download progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for uploading a single file to a repository.
///
/// Used with [`HFRepository::upload_file`](crate::repository::HFRepository::upload_file).
#[derive(TypedBuilder)]
pub struct RepoUploadFileParams {
    /// Source of the file content to upload (bytes or file path).
    pub source: AddSource,
    /// Destination path within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to upload to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the upload.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Expected parent commit SHA. The upload fails if the branch head has moved past this commit.
    #[builder(default, setter(into, strip_option))]
    pub parent_commit: Option<String>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for uploading a local folder to a repository.
///
/// Used with [`HFRepository::upload_folder`](crate::repository::HFRepository::upload_folder).
#[derive(TypedBuilder)]
pub struct RepoUploadFolderParams {
    /// Local folder path to upload.
    #[builder(setter(into))]
    pub folder_path: PathBuf,
    /// Destination directory within the repository. Defaults to the repo root.
    #[builder(default, setter(into, strip_option))]
    pub path_in_repo: Option<String>,
    /// Git revision (branch) to upload to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the upload.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Glob patterns for files to include from the local folder.
    #[builder(default, setter(strip_option))]
    pub allow_patterns: Option<Vec<String>>,
    /// Glob patterns for files to exclude from the local folder.
    #[builder(default, setter(strip_option))]
    pub ignore_patterns: Option<Vec<String>>,
    /// Glob patterns for remote files to delete that are not present locally.
    #[builder(default, setter(strip_option))]
    pub delete_patterns: Option<Vec<String>>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for deleting a single file from a repository.
///
/// Used with [`HFRepository::delete_file`](crate::repository::HFRepository::delete_file).
#[derive(TypedBuilder)]
pub struct RepoDeleteFileParams {
    /// Path of the file to delete within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to delete from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the deletion.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
}

/// Parameters for deleting a folder from a repository.
///
/// Used with [`HFRepository::delete_folder`](crate::repository::HFRepository::delete_folder).
#[derive(TypedBuilder)]
pub struct RepoDeleteFolderParams {
    /// Path of the folder to delete within the repository.
    #[builder(setter(into))]
    pub path_in_repo: String,
    /// Git revision (branch) to delete from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Commit message for the deletion.
    #[builder(default, setter(into, strip_option))]
    pub commit_message: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
}

/// Parameters for creating a commit composed of multiple file operations.
///
/// Used with [`HFRepository::create_commit`](crate::repository::HFRepository::create_commit).
#[derive(TypedBuilder)]
pub struct RepoCreateCommitParams {
    /// List of file operations (additions, deletions, copies) to include in the commit.
    pub operations: Vec<CommitOperation>,
    /// Commit message.
    #[builder(setter(into))]
    pub commit_message: String,
    /// Extended description for the commit.
    #[builder(default, setter(into, strip_option))]
    pub commit_description: Option<String>,
    /// Git revision (branch) to commit to. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// If `true`, create a pull request instead of committing directly.
    #[builder(default, setter(strip_option))]
    pub create_pr: Option<bool>,
    /// Expected parent commit SHA. The commit fails if the branch head has moved past this commit.
    #[builder(default, setter(into, strip_option))]
    pub parent_commit: Option<String>,
    /// Optional progress handler for tracking upload progress.
    #[builder(default)]
    pub progress: Option<Progress>,
}

/// Parameters for listing commits on a repository revision.
///
/// Used with [`HFRepository::list_commits`](crate::repository::HFRepository::list_commits).
#[derive(Default, TypedBuilder)]
pub struct RepoListCommitsParams {
    /// Git revision (branch, tag, or commit SHA) to list commits from. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Maximum number of commits to return.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for listing refs (branches, tags, ...) on a repository.
///
/// Used with [`HFRepository::list_refs`](crate::repository::HFRepository::list_refs).
#[derive(Default, TypedBuilder)]
pub struct RepoListRefsParams {
    /// Whether to include pull request refs in the listing.
    #[builder(default)]
    pub include_pull_requests: bool,
}

/// Parameters for fetching the parsed diff between a revision and its parent.
///
/// Used with [`HFRepository::get_commit_diff`](crate::repository::HFRepository::get_commit_diff).
#[derive(TypedBuilder)]
pub struct RepoGetCommitDiffParams {
    /// Revision to compare against the parent (branch, tag, or commit SHA).
    #[builder(setter(into))]
    pub compare: String,
}

/// Parameters for fetching the raw git diff between a revision and its parent.
///
/// Used with [`HFRepository::get_raw_diff`](crate::repository::HFRepository::get_raw_diff)
/// and [`HFRepository::get_raw_diff_stream`](crate::repository::HFRepository::get_raw_diff_stream).
#[derive(TypedBuilder)]
pub struct RepoGetRawDiffParams {
    /// Revision to compare against the parent (branch, tag, or commit SHA).
    #[builder(setter(into))]
    pub compare: String,
}

/// Parameters for creating a branch on a repository.
///
/// Used with [`HFRepository::create_branch`](crate::repository::HFRepository::create_branch).
#[derive(TypedBuilder)]
pub struct RepoCreateBranchParams {
    /// Name of the branch to create.
    #[builder(setter(into))]
    pub branch: String,
    /// Revision to branch from. Defaults to the current main branch head.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for deleting a branch on a repository.
///
/// Used with [`HFRepository::delete_branch`](crate::repository::HFRepository::delete_branch).
#[derive(TypedBuilder)]
pub struct RepoDeleteBranchParams {
    /// Name of the branch to delete.
    #[builder(setter(into))]
    pub branch: String,
}

/// Parameters for creating a tag on a repository.
///
/// Used with [`HFRepository::create_tag`](crate::repository::HFRepository::create_tag).
#[derive(TypedBuilder)]
pub struct RepoCreateTagParams {
    /// Name of the tag to create.
    #[builder(setter(into))]
    pub tag: String,
    /// Revision to tag. Defaults to the current main branch head.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
    /// Annotation message for the tag.
    #[builder(default, setter(into, strip_option))]
    pub message: Option<String>,
}

/// Parameters for deleting a tag on a repository.
///
/// Used with [`HFRepository::delete_tag`](crate::repository::HFRepository::delete_tag).
#[derive(TypedBuilder)]
pub struct RepoDeleteTagParams {
    /// Name of the tag to delete.
    #[builder(setter(into))]
    pub tag: String,
}

/// Parameters for updating repository settings (visibility, gating, description, ...).
///
/// Used with [`HFRepository::update_settings`](crate::repository::HFRepository::update_settings).
#[derive(Default, TypedBuilder, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoUpdateSettingsParams {
    /// Whether the repository should be private.
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub private: Option<bool>,
    /// Access-gating mode for the repository (e.g. `auto`, `manual`).
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gated: Option<GatedApprovalMode>,
    /// Repository description shown on the Hub page.
    #[builder(default, setter(into, strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether discussions are disabled on this repository.
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discussions_disabled: Option<bool>,
    /// Email address to receive gated-access request notifications.
    #[builder(default, setter(into, strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gated_notifications_email: Option<String>,
    /// When to send gated-access notifications (e.g. `each`, `daily`).
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gated_notifications_mode: Option<GatedNotificationsMode>,
}

/// Parameters for requesting a hardware change on a Space.
///
/// Used with [`HFSpace::request_hardware`](crate::repository::HFSpace::request_hardware).
#[derive(TypedBuilder)]
pub struct SpaceHardwareRequestParams {
    /// Hardware flavor to request (e.g. `"cpu-basic"`, `"t4-small"`, `"a10g-small"`).
    #[builder(setter(into))]
    pub hardware: String,
    /// Number of seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(default, setter(strip_option))]
    pub sleep_time: Option<u64>,
}

/// Parameters for setting the idle sleep time on a Space.
///
/// Used with [`HFSpace::set_sleep_time`](crate::repository::HFSpace::set_sleep_time).
#[derive(TypedBuilder)]
pub struct SpaceSleepTimeParams {
    /// Number of seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    pub sleep_time: u64,
}

/// Parameters for adding or updating a secret on a Space.
///
/// Used with [`HFSpace::add_secret`](crate::repository::HFSpace::add_secret).
#[derive(TypedBuilder)]
pub struct SpaceSecretParams {
    /// Secret key name.
    #[builder(setter(into))]
    pub key: String,
    /// Secret value.
    #[builder(setter(into))]
    pub value: String,
    /// Human-readable description of the secret.
    #[builder(default, setter(into, strip_option))]
    pub description: Option<String>,
}

/// Parameters for deleting a secret from a Space.
///
/// Used with [`HFSpace::delete_secret`](crate::repository::HFSpace::delete_secret).
#[derive(TypedBuilder)]
pub struct SpaceSecretDeleteParams {
    /// Secret key name to delete.
    #[builder(setter(into))]
    pub key: String,
}

/// Parameters for adding or updating a public variable on a Space.
///
/// Used with [`HFSpace::add_variable`](crate::repository::HFSpace::add_variable).
#[derive(TypedBuilder)]
pub struct SpaceVariableParams {
    /// Variable key name.
    #[builder(setter(into))]
    pub key: String,
    /// Variable value.
    #[builder(setter(into))]
    pub value: String,
    /// Human-readable description of the variable.
    #[builder(default, setter(into, strip_option))]
    pub description: Option<String>,
}

/// Parameters for deleting a public variable from a Space.
///
/// Used with [`HFSpace::delete_variable`](crate::repository::HFSpace::delete_variable).
#[derive(TypedBuilder)]
pub struct SpaceVariableDeleteParams {
    /// Variable key name to delete.
    #[builder(setter(into))]
    pub key: String,
}
