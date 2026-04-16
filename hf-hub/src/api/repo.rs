use futures::Stream;
use url::Url;

use crate::client::HFClient;
use crate::constants;
use crate::error::{HFError, Result};
use crate::repository::HFRepository;
use crate::types::{
    CreateRepoParams, DatasetInfo, DeleteRepoParams, ListDatasetsParams, ListModelsParams, ListSpacesParams, ModelInfo,
    MoveRepoParams, RepoFileExistsParams, RepoRevisionExistsParams, RepoUpdateSettingsParams, RepoUrl, SpaceInfo,
};

impl HFRepository {
    /// Get info about a model repository.
    /// Endpoint: GET /api/models/{repo_id} or /api/models/{repo_id}/revision/{revision}
    pub(crate) async fn model_info(&self, revision: Option<String>, expand: Option<Vec<String>>) -> Result<ModelInfo> {
        let mut url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        if let Some(ref revision) = revision {
            url = format!("{url}/revision/{revision}");
        }
        let mut request = self.hf_client.http_client().get(&url).headers(self.hf_client.auth_headers());
        if let Some(ref expand) = expand {
            let expand_params: Vec<(&str, &str)> = expand.iter().map(|v| ("expand", v.as_str())).collect();
            request = request.query(&expand_params);
        }
        let response = request.send().await?;
        let repo_path = self.repo_path();
        let not_found_ctx = match revision {
            Some(rev) => crate::error::NotFoundContext::Revision { revision: rev },
            None => crate::error::NotFoundContext::Repo,
        };
        let response = self.hf_client.check_response(response, Some(&repo_path), not_found_ctx).await?;
        Ok(response.json().await?)
    }

    /// Get info about a dataset repository.
    /// Endpoint: GET /api/datasets/{repo_id} or /api/datasets/{repo_id}/revision/{revision}
    pub(crate) async fn dataset_info(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> Result<DatasetInfo> {
        let mut url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        if let Some(ref revision) = revision {
            url = format!("{url}/revision/{revision}");
        }
        let mut request = self.hf_client.http_client().get(&url).headers(self.hf_client.auth_headers());
        if let Some(ref expand) = expand {
            let expand_params: Vec<(&str, &str)> = expand.iter().map(|v| ("expand", v.as_str())).collect();
            request = request.query(&expand_params);
        }
        let response = request.send().await?;
        let repo_path = self.repo_path();
        let not_found_ctx = match revision {
            Some(rev) => crate::error::NotFoundContext::Revision { revision: rev },
            None => crate::error::NotFoundContext::Repo,
        };
        let response = self.hf_client.check_response(response, Some(&repo_path), not_found_ctx).await?;
        Ok(response.json().await?)
    }

    /// Get info about a space.
    /// Endpoint: GET /api/spaces/{repo_id} or /api/spaces/{repo_id}/revision/{revision}
    pub(crate) async fn space_info(&self, revision: Option<String>, expand: Option<Vec<String>>) -> Result<SpaceInfo> {
        let mut url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        if let Some(ref revision) = revision {
            url = format!("{url}/revision/{revision}");
        }
        let mut request = self.hf_client.http_client().get(&url).headers(self.hf_client.auth_headers());
        if let Some(ref expand) = expand {
            let expand_params: Vec<(&str, &str)> = expand.iter().map(|v| ("expand", v.as_str())).collect();
            request = request.query(&expand_params);
        }
        let response = request.send().await?;
        let repo_path = self.repo_path();
        let not_found_ctx = match revision {
            Some(rev) => crate::error::NotFoundContext::Revision { revision: rev },
            None => crate::error::NotFoundContext::Repo,
        };
        let response = self.hf_client.check_response(response, Some(&repo_path), not_found_ctx).await?;
        Ok(response.json().await?)
    }

    /// Return `true` if the repository exists and is accessible with the current credentials.
    pub async fn exists(&self) -> Result<bool> {
        let url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        match response.status().as_u16() {
            200..=299 => Ok(true),
            404 => Ok(false),
            401 => Err(HFError::AuthRequired),
            status => {
                let url = response.url().to_string();
                let body = response.text().await.unwrap_or_default();
                Err(HFError::Http {
                    status: reqwest::StatusCode::from_u16(status).unwrap(),
                    url,
                    body,
                })
            },
        }
    }

    /// Return `true` if the given revision (branch, tag, or commit SHA) exists.
    pub async fn revision_exists(&self, params: &RepoRevisionExistsParams) -> Result<bool> {
        let url =
            format!("{}/revision/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.revision);
        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        match response.status().as_u16() {
            200..=299 => Ok(true),
            404 => Ok(false),
            401 => Err(HFError::AuthRequired),
            status => {
                let url_str = response.url().to_string();
                let body = response.text().await.unwrap_or_default();
                Err(HFError::Http {
                    status: reqwest::StatusCode::from_u16(status).unwrap(),
                    url: url_str,
                    body,
                })
            },
        }
    }

    /// Return `true` if the given file exists in the repository at the specified revision.
    pub async fn file_exists(&self, params: &RepoFileExistsParams) -> Result<bool> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &self.repo_path(), revision, &params.filename);
        let response = self
            .hf_client
            .http_client()
            .head(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        match response.status().as_u16() {
            200..=299 => Ok(true),
            404 => {
                if self
                    .revision_exists(&RepoRevisionExistsParams::builder().revision(revision.to_string()).build())
                    .await?
                {
                    Ok(false)
                } else {
                    Err(HFError::RevisionNotFound {
                        repo_id: self.repo_path(),
                        revision: revision.to_string(),
                    })
                }
            },
            401 => Err(HFError::AuthRequired),
            status => {
                let url_str = response.url().to_string();
                let body = response.text().await.unwrap_or_default();
                Err(HFError::Http {
                    status: reqwest::StatusCode::from_u16(status).unwrap(),
                    url: url_str,
                    body,
                })
            },
        }
    }

    /// Update repository settings such as visibility, gating policy, description,
    /// discussion settings, and gated notification preferences.
    /// Endpoint: PUT /api/{repo_type}s/{repo_id}/settings
    pub async fn update_settings(&self, params: &RepoUpdateSettingsParams) -> Result<()> {
        let url = format!("{}/settings", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()));

        let response = self
            .hf_client
            .http_client()
            .put(&url)
            .headers(self.hf_client.auth_headers())
            .json(params)
            .send()
            .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }
}

impl HFClient {
    /// List models on the Hub.
    /// Endpoint: GET /api/models
    pub fn list_models(&self, params: &ListModelsParams) -> Result<impl Stream<Item = Result<ModelInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/models", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref search) = params.search {
            query.push(("search".into(), search.clone()));
        }
        if let Some(ref author) = params.author {
            query.push(("author".into(), author.clone()));
        }
        if let Some(ref filter) = params.filter {
            query.push(("filter".into(), filter.clone()));
        }
        if let Some(ref sort) = params.sort {
            query.push(("sort".into(), sort.clone()));
        }
        if let Some(max) = params.limit {
            // The Hub API usually returns up to 1000 items per page by default,
            // so only set an explicit limit for smaller requests.
            if max < 1000 {
                query.push(("limit".into(), max.to_string()));
            }
        }
        if let Some(ref pipeline_tag) = params.pipeline_tag {
            query.push(("pipeline_tag".into(), pipeline_tag.clone()));
        }
        if params.full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        if params.card_data == Some(true) {
            query.push(("cardData".into(), "true".into()));
        }
        if params.fetch_config == Some(true) {
            query.push(("config".into(), "true".into()));
        }
        Ok(self.paginate(url, query, params.limit))
    }

    /// List datasets on the Hub.
    /// Endpoint: GET /api/datasets
    pub fn list_datasets(&self, params: &ListDatasetsParams) -> Result<impl Stream<Item = Result<DatasetInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/datasets", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref search) = params.search {
            query.push(("search".into(), search.clone()));
        }
        if let Some(ref author) = params.author {
            query.push(("author".into(), author.clone()));
        }
        if let Some(ref filter) = params.filter {
            query.push(("filter".into(), filter.clone()));
        }
        if let Some(ref sort) = params.sort {
            query.push(("sort".into(), sort.clone()));
        }
        if let Some(max) = params.limit {
            // The Hub API usually returns up to 1000 items per page by default,
            // so only set an explicit limit for smaller requests.
            if max < 1000 {
                query.push(("limit".into(), max.to_string()));
            }
        }
        if params.full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, params.limit))
    }

    /// List spaces on the Hub.
    /// Endpoint: GET /api/spaces
    pub fn list_spaces(&self, params: &ListSpacesParams) -> Result<impl Stream<Item = Result<SpaceInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/spaces", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref search) = params.search {
            query.push(("search".into(), search.clone()));
        }
        if let Some(ref author) = params.author {
            query.push(("author".into(), author.clone()));
        }
        if let Some(ref filter) = params.filter {
            query.push(("filter".into(), filter.clone()));
        }
        if let Some(ref sort) = params.sort {
            query.push(("sort".into(), sort.clone()));
        }
        if let Some(max) = params.limit {
            // The Hub API usually returns up to 1000 items per page by default,
            // so only set an explicit limit for smaller requests.
            if max < 1000 {
                query.push(("limit".into(), max.to_string()));
            }
        }
        if params.full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, params.limit))
    }

    /// Create a new repository.
    /// Endpoint: POST /api/repos/create
    pub async fn create_repo(&self, params: &CreateRepoParams) -> Result<RepoUrl> {
        let url = format!("{}/api/repos/create", self.endpoint());

        let (namespace, name) = split_repo_id(&params.repo_id);

        let mut body = serde_json::json!({
            "name": name,
            "private": params.private.unwrap_or(false),
        });

        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(ref repo_type) = params.repo_type {
            body["type"] = serde_json::Value::String(repo_type.to_string());
        }
        if let Some(ref sdk) = params.space_sdk {
            body["sdk"] = serde_json::Value::String(sdk.clone());
        }

        let response = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .await?;

        if response.status().as_u16() == 409 && params.exist_ok {
            // Already exists and exist_ok=true, return its URL
            let prefix = constants::repo_type_url_prefix(params.repo_type);
            return Ok(RepoUrl {
                url: format!("{}/{}{}", self.endpoint(), prefix, params.repo_id),
            });
        }

        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a repository.
    /// Endpoint: DELETE /api/repos/delete
    pub async fn delete_repo(&self, params: &DeleteRepoParams) -> Result<()> {
        let url = format!("{}/api/repos/delete", self.endpoint());

        let (namespace, name) = split_repo_id(&params.repo_id);

        let mut body = serde_json::json!({ "name": name });
        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(ref repo_type) = params.repo_type {
            body["type"] = serde_json::Value::String(repo_type.to_string());
        }

        let response = self
            .http_client()
            .delete(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .await?;

        if response.status().as_u16() == 404 && params.missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(&params.repo_id), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Move (rename) a repository.
    /// Endpoint: POST /api/repos/move
    pub async fn move_repo(&self, params: &MoveRepoParams) -> Result<RepoUrl> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let mut body = serde_json::json!({
            "fromRepo": params.from_id,
            "toRepo": params.to_id,
        });
        if let Some(ref repo_type) = params.repo_type {
            body["type"] = serde_json::Value::String(repo_type.to_string());
        }

        let response = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers())
            .json(&body)
            .send()
            .await?;

        self.check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        let prefix = constants::repo_type_url_prefix(params.repo_type);
        Ok(RepoUrl {
            url: format!("{}/{}{}", self.endpoint(), prefix, params.to_id),
        })
    }
}

/// Split "namespace/name" into (Some("namespace"), "name") or (None, "name")
fn split_repo_id(repo_id: &str) -> (Option<&str>, &str) {
    match repo_id.split_once('/') {
        Some((ns, name)) => (Some(ns), name),
        None => (None, repo_id),
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::split_repo_id;
    use crate::client::HFClient;
    use crate::types::{ListDatasetsParams, ListModelsParams, ListSpacesParams};

    #[test]
    fn test_split_repo_id() {
        assert_eq!(split_repo_id("user/repo"), (Some("user"), "repo"));
        assert_eq!(split_repo_id("repo"), (None, "repo"));
        assert_eq!(split_repo_id("org/sub/repo"), (Some("org"), "sub/repo"));
    }

    #[tokio::test]
    async fn test_list_models_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let params = ListModelsParams::builder().limit(0_usize).build();
        let stream = client.list_models(&params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_datasets_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let params = ListDatasetsParams::builder().limit(0_usize).build();
        let stream = client.list_datasets(&params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_spaces_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let params = ListSpacesParams::builder().limit(0_usize).build();
        let stream = client.list_spaces(&params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }
}

sync_api! {
    impl HFClient -> HFClientSync {
        fn create_repo(&self, params: &CreateRepoParams) -> Result<RepoUrl>;
        fn delete_repo(&self, params: &DeleteRepoParams) -> Result<()>;
        fn move_repo(&self, params: &MoveRepoParams) -> Result<RepoUrl>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_models(&self, params: &ListModelsParams) -> ModelInfo;
        fn list_datasets(&self, params: &ListDatasetsParams) -> DatasetInfo;
        fn list_spaces(&self, params: &ListSpacesParams) -> SpaceInfo;
    }
}

sync_api! {
    impl HFRepository -> HFRepositorySync {
        fn info(&self, params: &crate::types::RepoInfoParams) -> Result<crate::types::RepoInfo>;
        fn exists(&self) -> Result<bool>;
        fn revision_exists(&self, params: &RepoRevisionExistsParams) -> Result<bool>;
        fn file_exists(&self, params: &RepoFileExistsParams) -> Result<bool>;
        fn update_settings(&self, params: &RepoUpdateSettingsParams) -> Result<()>;
    }
}
