use crate::api::commit::CommitOperationAdd;
use crate::api::tokio::ApiError;
use reqwest::header::HeaderMap;
use reqwest::Client;
use std::path::Path;
use std::sync::Arc;

use xet::xet_session::{Sha256Policy, XetFileInfo, XetSessionBuilder};
pub(crate) use xet::xet_session::{UniqueID, XetSession};
use xet_client::cas_client::auth::{AuthError, TokenInfo, TokenRefresher};

/// Xet error type, re-exported from xet crate.
pub type XetError = xet::XetError;

pub(crate) struct XetConnectionInfo {
    pub access_token: String,
    pub expiration_unix_epoch: u64,
    pub cas_url: String,
}

#[derive(Clone, Copy)]
pub(crate) enum XetTokenType {
    Read,
    Write,
}

impl XetTokenType {
    fn as_str(&self) -> &str {
        match self {
            XetTokenType::Read => "read",
            XetTokenType::Write => "write",
        }
    }
}

/// GET /api/{repo_type}s/{repo_id}/xet-{token_type}-token/{revision}
///
/// Returns JSON: { "accessToken": str, "exp": u64, "casUrl": str }
pub(crate) async fn fetch_xet_token(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    token_type: XetTokenType,
) -> Result<XetConnectionInfo, ApiError> {
    let token_type_str = token_type.as_str();
    let url = format!("{endpoint}/api/{repo_type}/{repo_id}/xet-{token_type_str}-token/{revision}");

    let response = client
        .get(&url)
        .headers(headers.clone())
        .send()
        .await?
        .error_for_status()?;

    let body: serde_json::Value = response.json().await?;

    let access_token = body["accessToken"]
        .as_str()
        .ok_or_else(|| {
            ApiError::InvalidApiResponse("missing accessToken in xet token response".into())
        })?
        .to_string();

    let expiration_unix_epoch = body["exp"]
        .as_u64()
        .ok_or_else(|| ApiError::InvalidApiResponse("missing exp in xet token response".into()))?;

    let cas_url = body["casUrl"]
        .as_str()
        .ok_or_else(|| ApiError::InvalidApiResponse("missing casUrl in xet token response".into()))?
        .to_string();

    Ok(XetConnectionInfo {
        access_token,
        expiration_unix_epoch,
        cas_url,
    })
}

struct HfTokenRefresher {
    client: Client,
    endpoint: String,
    repo_type: String,
    repo_id: String,
    revision: String,
    headers: HeaderMap,
    token_type: XetTokenType,
}

#[async_trait::async_trait]
impl TokenRefresher for HfTokenRefresher {
    async fn refresh(&self) -> Result<TokenInfo, AuthError> {
        let info = fetch_xet_token(
            &self.client,
            &self.endpoint,
            &self.repo_type,
            &self.repo_id,
            &self.revision,
            &self.headers,
            self.token_type,
        )
        .await
        .map_err(|e| AuthError::TokenRefreshFailure(e.to_string()))?;
        Ok((info.access_token, info.expiration_unix_epoch))
    }
}

/// Parsed from HEAD response headers on the resolve URL.
#[derive(Debug)]
pub(crate) struct XetFileData {
    pub file_hash: String,
}

/// Checks for X-Xet-Hash header in the response.
/// Returns None if not a xet file.
pub(crate) fn parse_xet_file_data(headers: &HeaderMap) -> Option<XetFileData> {
    let header_name = reqwest::header::HeaderName::from_static("x-xet-hash");
    headers.get(&header_name).and_then(|value| {
        value.to_str().ok().map(|hash| XetFileData {
            file_hash: hash.to_string(),
        })
    })
}

/// Create a new XetSession.
/// Fetches initial token, builds session with token refresher.
pub(crate) async fn create_session(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    token_type: XetTokenType,
) -> Result<XetSession, ApiError> {
    let info = fetch_xet_token(
        client, endpoint, repo_type, repo_id, revision, headers, token_type,
    )
    .await?;

    let refresher = HfTokenRefresher {
        client: client.clone(),
        endpoint: endpoint.to_string(),
        repo_type: repo_type.to_string(),
        repo_id: repo_id.to_string(),
        revision: revision.to_string(),
        headers: headers.clone(),
        token_type,
    };

    let session = XetSessionBuilder::new()
        .with_endpoint(info.cas_url)
        .with_token_info(info.access_token, info.expiration_unix_epoch)
        .with_token_refresher(Arc::new(refresher))
        .build_async()
        .await
        .map_err(ApiError::XetError)?;

    Ok(session)
}

/// Downloads a single xet file to the given path.
pub(crate) async fn xet_download(
    session: &XetSession,
    file_data: &XetFileData,
    file_size: u64,
    dest_path: &Path,
) -> Result<(), ApiError> {
    let download_group = session
        .new_download_group()
        .await
        .map_err(ApiError::XetError)?;

    let file_info = XetFileInfo {
        hash: file_data.file_hash.clone(),
        file_size,
        sha256: None,
    };

    download_group
        .download_file_to_path(file_info, dest_path.to_path_buf())
        .await
        .map_err(ApiError::XetError)?;

    let results = download_group.finish().await.map_err(ApiError::XetError)?;

    // Check for any download errors
    for result in results.values() {
        if let Err(e) = result.as_ref() {
            return Err(ApiError::XetError(xet::XetError::Internal(e.to_string())));
        }
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) struct XetUploadResult {
    pub path_in_repo: String,
    pub xet_hash: String,
    pub file_size: u64,
    pub sha256: Option<String>,
}

/// Uploads multiple files via the shared XetSession.
pub(crate) async fn xet_upload(
    session: &XetSession,
    files: &[&CommitOperationAdd],
) -> Result<Vec<XetUploadResult>, ApiError> {
    let upload_commit = session
        .new_upload_commit()
        .await
        .map_err(ApiError::XetError)?;

    let mut task_ids_and_paths: Vec<(UniqueID, String)> = Vec::new();

    for file in files {
        if file.should_ignore {
            continue;
        }
        let handle = upload_commit
            .upload_from_path(file.local_path.clone(), Sha256Policy::Compute)
            .await
            .map_err(ApiError::XetError)?;
        task_ids_and_paths.push((handle.task_id, file.path_in_repo.clone()));
    }

    let results = upload_commit.commit().await.map_err(ApiError::XetError)?;

    let mut upload_results = Vec::new();
    for (task_id, path_in_repo) in &task_ids_and_paths {
        if let Some(result) = results.get(task_id) {
            let metadata = result
                .as_ref()
                .as_ref()
                .map_err(|e| ApiError::XetError(xet::XetError::Internal(e.to_string())))?;
            upload_results.push(XetUploadResult {
                path_in_repo: path_in_repo.clone(),
                xet_hash: metadata.hash.clone(),
                file_size: metadata.file_size,
                sha256: metadata.sha256.clone(),
            });
        }
    }

    Ok(upload_results)
}
