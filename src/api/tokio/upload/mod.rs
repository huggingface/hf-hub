use commit_api::{CommitError, CommitOperationAdd, UploadSource};
use commit_info::CommitInfo;

use super::ApiRepo;

mod commit_api;
mod commit_info;
mod completion_payload;
mod lfs;

impl ApiRepo {
    /// Upload a local file (up to 50 GB) to the given repo. The upload is done
    /// through an HTTP post request, and doesn't require git or git-lfs to be
    /// installed.
    pub async fn upload_file(
        &self,
        source: impl Into<UploadSource>,
        path_in_repo: &str,
        commit_message: Option<String>,
        commit_description: Option<String>,
        create_pr: bool,
    ) -> Result<CommitInfo, CommitError> {
        let commit_message =
            commit_message.unwrap_or_else(|| format!("Upload {path_in_repo} with hf_hub"));
        let operation =
            CommitOperationAdd::from_upload_source(path_in_repo.to_string(), source.into()).await?;

        let commit_info = self
            .create_commit(
                vec![operation.into()],
                commit_message,
                commit_description,
                Some(create_pr),
                None,
                None,
            )
            .await?;

        Ok(commit_info)
    }
}
