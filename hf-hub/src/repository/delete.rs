//! Delete builders on [`HFRepository`].
//!
//! Deletes never move file content (no LFS negotiation, no xet upload), so
//! unlike [`upload`](super::upload) this module is always available
//! regardless of the `upload` feature.

use futures::stream::StreamExt;

use super::{CommitInfo, HFRepository, RepoTreeEntry, RepoType};
use crate::constants;
use crate::error::HFResult;

/// Internal options struct for [`HFRepository::delete_file`].
struct DeleteFileParams {
    path_in_repo: String,
    revision: Option<String>,
    commit_message: Option<String>,
    create_pr: bool,
}

/// Internal options struct for [`HFRepository::delete_folder`].
struct DeleteFolderParams {
    path_in_repo: String,
    revision: Option<String>,
    commit_message: Option<String>,
    create_pr: bool,
}

impl<T: RepoType> HFRepository<T> {
    async fn delete_file_impl(&self, params: DeleteFileParams) -> HFResult<CommitInfo> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION).to_string();
        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| format!("Delete {}", params.path_in_repo));

        let header = super::commit::commit_header_line(&commit_message, None, None)?;
        let delete_line = serde_json::to_vec(&serde_json::json!({
            "key": "deletedFile",
            "value": {"path": params.path_in_repo}
        }))?;

        self.send_commit(&revision, vec![header, delete_line], params.create_pr, &None)
            .await
    }

    async fn delete_folder_impl(&self, params: DeleteFolderParams) -> HFResult<CommitInfo> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION).to_string();

        let stream = self.list_tree().revision(revision.clone()).recursive(true).send()?;
        futures::pin_mut!(stream);

        let prefix = if params.path_in_repo.ends_with('/') {
            params.path_in_repo.clone()
        } else {
            format!("{}/", params.path_in_repo)
        };

        let mut ndjson_lines = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, .. } = entry
                && (path.starts_with(&prefix) || path == params.path_in_repo)
            {
                ndjson_lines.push(serde_json::to_vec(&serde_json::json!({
                    "key": "deletedFile",
                    "value": {"path": path}
                }))?);
            }
        }

        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| format!("Delete {}", params.path_in_repo));
        let header = super::commit::commit_header_line(&commit_message, None, None)?;
        ndjson_lines.insert(0, header);

        self.send_commit(&revision, ndjson_lines, params.create_pr, &None).await
    }
}

#[bon::bon]
impl<T: RepoType> HFRepository<T> {
    /// Delete a file from a repository.
    ///
    /// Convenience wrapper around the commit endpoint. If `commit_message` is omitted, a default
    /// `"Delete {path}"` message is used.
    ///
    /// # Parameters
    ///
    /// - `path_in_repo` (required): path of the file to delete.
    /// - `revision`: branch to delete from. Defaults to the main branch.
    /// - `commit_message`: commit message.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_file(
        &self,
        /// Path of the file to delete.
        #[builder(into)]
        path_in_repo: String,
        /// Branch to delete from. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message.
        #[builder(into)]
        commit_message: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.delete_file_impl(DeleteFileParams {
            path_in_repo,
            revision,
            commit_message,
            create_pr,
        }))
        .await
    }

    /// Delete all files under a repository path.
    ///
    /// The current tree is listed recursively and every file at or below `path_in_repo` is turned
    /// into a delete operation. Directories disappear as a consequence of deleting their contents.
    ///
    /// # Parameters
    ///
    /// - `path_in_repo` (required): folder path within the repository.
    /// - `revision`: branch to delete from. Defaults to the main branch.
    /// - `commit_message`: commit message.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_folder(
        &self,
        /// Folder path within the repository.
        #[builder(into)]
        path_in_repo: String,
        /// Branch to delete from. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message.
        #[builder(into)]
        commit_message: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.delete_folder_impl(DeleteFolderParams {
            path_in_repo,
            revision,
            commit_message,
            create_pr,
        }))
        .await
    }
}

#[cfg(all(feature = "blocking", not(target_family = "wasm")))]
#[bon::bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::delete_file`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_file(
        &self,
        #[builder(into)] path_in_repo: String,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(default)] create_pr: bool,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .delete_file()
                .path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .create_pr(create_pr)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_folder`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_folder(
        &self,
        #[builder(into)] path_in_repo: String,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(default)] create_pr: bool,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .delete_folder()
                .path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .create_pr(create_pr)
                .send(),
        )
    }
}
