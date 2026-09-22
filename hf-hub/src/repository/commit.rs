//! Shared commit-transport plumbing for the `upload` and `delete` submodules.
//!
//! Building the ndjson commit body differs by caller (uploads negotiate an
//! LFS/xet transfer for `Add` operations; deletes never do), but POSTing the
//! finished body to the commit endpoint and parsing the response is
//! identical either way. This module holds that shared tail, always
//! compiled regardless of the `upload` feature, so [`delete`](super::delete)
//! can send delete-only commits without depending on the `upload`-gated
//! [`upload`](super::upload) module.

use serde_json::json;

use super::{CommitInfo, HFRepository, RepoType};
use crate::client::encode_ref;
use crate::error::HFResult;
use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::retry;

/// Build the ndjson `header` line for a commit (summary/description/parentCommit).
pub(super) fn commit_header_line(
    commit_message: &str,
    commit_description: Option<&str>,
    parent_commit: Option<&str>,
) -> HFResult<Vec<u8>> {
    let mut header_value = json!({
        "summary": commit_message,
        "description": commit_description.unwrap_or(""),
    });
    if let Some(parent) = parent_commit {
        header_value["parentCommit"] = serde_json::Value::String(parent.to_string());
    }
    Ok(serde_json::to_vec(&json!({"key": "header", "value": header_value}))?)
}

impl<T: RepoType> HFRepository<T> {
    /// POST a finished set of ndjson commit lines to the commit endpoint and parse the response.
    pub(super) async fn send_commit(
        &self,
        revision: &str,
        ndjson_lines: Vec<Vec<u8>>,
        create_pr: bool,
        progress: &Option<Progress>,
    ) -> HFResult<CommitInfo> {
        let url = format!(
            "{}/commit/{}",
            self.hf_client.api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );

        let body: Vec<u8> = ndjson_lines
            .into_iter()
            .flat_map(|mut line| {
                line.push(b'\n');
                line
            })
            .collect();

        progress.emit(UploadEvent::Committing);

        let mut headers = self.hf_client.auth_headers();
        headers.insert(reqwest::header::CONTENT_TYPE, "application/x-ndjson".parse().unwrap());

        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self
                .hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .body(body.clone());
            if create_pr {
                req = req.query(&[("create_pr", "1")]);
            }
            req.send()
        })
        .await?;

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;

        progress.emit(UploadEvent::Complete);
        Ok(response.json().await?)
    }
}
