use super::ApiRepo;
use commit_api::CommitOperationAdd;
use commit_info::CommitInfo;
use futures::future::join_all;

pub use commit_api::{CommitError, UploadSource};

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
        self.upload_files(
            vec![(source.into(), path_in_repo.to_string())],
            commit_message,
            commit_description,
            create_pr,
        )
        .await
    }

    /// Upload multiple local files (up to 50 GB each) to the given repo. The upload is done
    /// through an HTTP post request, and doesn't require git or git-lfs to be
    /// installed.
    pub async fn upload_files(
        &self,
        files: Vec<(UploadSource, String)>,
        commit_message: Option<String>,
        commit_description: Option<String>,
        create_pr: bool,
    ) -> Result<CommitInfo, CommitError> {
        let commit_message =
            commit_message.unwrap_or_else(|| format!("Upload {} files with hf_hub", files.len()));

        let operations = join_all(
            files
                .into_iter()
                .map(|(source, path)| CommitOperationAdd::from_upload_source(path, source)),
        )
        .await
        .into_iter()
        .map(|operation| operation.map(|o| o.into()))
        .collect::<Result<_, _>>()?;

        let commit_info = self
            .create_commit(
                operations,
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
