//! Repository handles, metadata types, and list/create/delete/move APIs.

pub mod commits;
pub mod diff;
pub mod download;
pub mod files;
pub mod listing;
pub mod upload;

use std::fmt;
use std::str::FromStr;

use bon::bon;
pub use commits::{CommitAuthor, DiffEntry, GitCommitInfo, GitRefInfo, GitRefs};
pub use diff::{GitStatus, HFDiffParseError, HFFileDiff};
pub use files::{AddSource, BlobLfsInfo, CommitInfo, CommitOperation, FileMetadataInfo, LastCommitInfo, RepoTreeEntry};
pub(crate) use files::{extract_file_size, extract_xet_hash};
use futures::Stream;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

pub(crate) mod _kind {
    use serde::{Deserialize, Serialize};

    /// The kind of repository on the Hugging Face Hub.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum RepoType {
        Model,
        Dataset,
        Space,
        Kernel,
    }
}

pub(crate) use _kind::RepoType;

/// Access-gating mode for a repository.
///
/// Controls whether users must request access and how requests are approved.
/// Serializes as `false` when [`GatedApprovalMode::Disabled`], or as the lowercase mode string otherwise.
#[derive(Debug, Clone)]
pub enum GatedApprovalMode {
    Disabled,
    Auto,
    Manual,
}

/// Notification cadence for gated-access requests on a repository.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatedNotificationsMode {
    Bulk,
    RealTime,
}

/// Repo-type-tagged wrapper over [`ModelInfo`], [`DatasetInfo`], and [`SpaceInfo`].
///
/// Returned by [`HFRepository::info`]; the active variant
/// matches the repository's [`RepoType`].
#[derive(Debug, Clone)]
pub enum RepoInfo {
    Model(ModelInfo),
    Dataset(DatasetInfo),
    Space(SpaceInfo),
}

/// A single file entry in a repository's flat "siblings" listing, as returned by the repo info endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSibling {
    pub rfilename: String,
    pub size: Option<u64>,
    pub lfs: Option<files::BlobLfsInfo>,
}

/// Metadata for a model repository on the Hub.
///
/// Returned by [`HFClient::list_models`] and by
/// [`HFRepository::info`] when the repo is a model.
/// Most fields are optional because they depend on the `expand` parameter and the repo's state.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    #[serde(rename = "_id")]
    pub mongo_id: Option<String>,
    pub model_id: Option<String>,
    pub author: Option<String>,
    pub sha: Option<String>,
    pub private: Option<bool>,
    pub gated: Option<serde_json::Value>,
    pub disabled: Option<bool>,
    pub downloads: Option<u64>,
    pub downloads_all_time: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    #[serde(rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    #[serde(rename = "library_name")]
    pub library_name: Option<String>,
    pub created_at: Option<String>,
    pub last_modified: Option<String>,
    pub siblings: Option<Vec<RepoSibling>>,
    pub card_data: Option<serde_json::Value>,
    pub config: Option<serde_json::Value>,
    pub trending_score: Option<f64>,
    pub gguf: Option<serde_json::Value>,
    pub spaces: Option<Vec<String>>,
    pub used_storage: Option<u64>,
    pub widget_data: Option<serde_json::Value>,
}

/// Metadata for a dataset repository on the Hub.
///
/// Returned by [`HFClient::list_datasets`] and by
/// [`HFRepository::info`] when the repo is a dataset.
/// Most fields are optional because they depend on the `expand` parameter and the repo's state.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetInfo {
    pub id: String,
    #[serde(rename = "_id")]
    pub mongo_id: Option<String>,
    pub author: Option<String>,
    pub sha: Option<String>,
    pub private: Option<bool>,
    pub gated: Option<serde_json::Value>,
    pub disabled: Option<bool>,
    pub downloads: Option<u64>,
    pub downloads_all_time: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub created_at: Option<String>,
    pub last_modified: Option<String>,
    pub siblings: Option<Vec<RepoSibling>>,
    pub card_data: Option<serde_json::Value>,
    pub trending_score: Option<f64>,
    pub description: Option<String>,
    pub used_storage: Option<u64>,
}

/// Metadata for a Space repository on the Hub.
///
/// Returned by [`HFClient::list_spaces`] and by
/// [`HFRepository::info`] when the repo is a Space.
/// Most fields are optional because they depend on the `expand` parameter and the Space's state.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceInfo {
    pub id: String,
    #[serde(rename = "_id")]
    pub mongo_id: Option<String>,
    pub author: Option<String>,
    pub sha: Option<String>,
    pub private: Option<bool>,
    pub gated: Option<serde_json::Value>,
    pub disabled: Option<bool>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
    pub created_at: Option<String>,
    pub last_modified: Option<String>,
    pub siblings: Option<Vec<RepoSibling>>,
    pub card_data: Option<serde_json::Value>,
    pub sdk: Option<String>,
    pub trending_score: Option<f64>,
    pub host: Option<String>,
    pub subdomain: Option<String>,
    pub runtime: Option<serde_json::Value>,
    pub used_storage: Option<u64>,
}

/// URL returned by create_repo/move_repo
#[derive(Debug, Clone, Deserialize)]
pub struct RepoUrl {
    pub url: String,
}

pub(crate) mod _handle {
    use super::{HFClient, RepoType};

    /// A handle for a single repository on the Hugging Face Hub.
    ///
    /// `HFRepository` is created via [`HFClient::repo`], [`HFClient::model`], or
    /// [`HFClient::dataset`] and binds together the client, owner, repo name, and repo type.
    /// All repo-scoped API operations are methods on this type.
    ///
    /// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use hf_hub::{HFClient, RepoType};
    /// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
    /// let client = HFClient::builder().build()?;
    /// let repo = client.model("openai-community", "gpt2");
    /// let info = repo.info().send().await?;
    /// # Ok(()) }
    /// ```
    #[derive(Clone)]
    pub struct HFRepository {
        pub(crate) hf_client: HFClient,
        pub(super) owner: String,
        pub(super) name: String,
        pub(crate) repo_type: RepoType,
    }
}

pub(crate) use _handle::HFRepository;

impl fmt::Display for RepoType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RepoType::Model => write!(f, "model"),
            RepoType::Dataset => write!(f, "dataset"),
            RepoType::Space => write!(f, "space"),
            RepoType::Kernel => write!(f, "kernel"),
        }
    }
}

impl FromStr for RepoType {
    type Err = crate::error::HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "model" => Ok(RepoType::Model),
            "dataset" => Ok(RepoType::Dataset),
            "space" => Ok(RepoType::Space),
            "kernel" => Ok(RepoType::Kernel),
            _ => Err(crate::error::HFError::Other(format!("Unknown repo type: {s}"))),
        }
    }
}

impl Serialize for GatedApprovalMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            GatedApprovalMode::Disabled => serializer.serialize_bool(false),
            GatedApprovalMode::Auto => serializer.serialize_str("auto"),
            GatedApprovalMode::Manual => serializer.serialize_str("manual"),
        }
    }
}

impl FromStr for GatedApprovalMode {
    type Err = crate::error::HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "false" | "disabled" => Ok(GatedApprovalMode::Disabled),
            "auto" => Ok(GatedApprovalMode::Auto),
            "manual" => Ok(GatedApprovalMode::Manual),
            _ => Err(crate::error::HFError::Other(format!(
                "Unknown gated approval mode: {s}. Expected 'auto', 'manual', or 'false'"
            ))),
        }
    }
}

impl FromStr for GatedNotificationsMode {
    type Err = crate::error::HFError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bulk" => Ok(GatedNotificationsMode::Bulk),
            "real-time" | "realtime" => Ok(GatedNotificationsMode::RealTime),
            _ => Err(crate::error::HFError::Other(format!(
                "Unknown gated notifications mode: {s}. Expected 'bulk' or 'real-time'"
            ))),
        }
    }
}

impl RepoInfo {
    pub fn repo_type(&self) -> RepoType {
        match self {
            RepoInfo::Model(_) => RepoType::Model,
            RepoInfo::Dataset(_) => RepoType::Dataset,
            RepoInfo::Space(_) => RepoType::Space,
        }
    }
}

impl fmt::Debug for HFRepository {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFRepository")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .field("repo_type", &self.repo_type)
            .finish()
    }
}

impl HFClient {
    /// Create an [`HFRepository`] handle for any repo type.
    pub fn repo(&self, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        HFRepository::new(self.clone(), repo_type, owner, name)
    }

    /// Create an [`HFRepository`] handle for a model repository.
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        self.repo(RepoType::Model, owner, name)
    }

    /// Create an [`HFRepository`] handle for a dataset repository.
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        self.repo(RepoType::Dataset, owner, name)
    }
}

#[bon]
impl HFClient {
    /// List models on the Hub. Endpoint: `GET /api/models`.
    ///
    /// Returns a stream of [`ModelInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`: filter models by a text query (matches model IDs and descriptions).
    /// - `author`: filter models by author or organization name.
    /// - `filter`: filter models by tags (e.g. `"text-generation"`, `"pytorch"`).
    /// - `sort`: property to sort results by (e.g. `"downloads"`, `"lastModified"`).
    /// - `pipeline_tag`: filter models by pipeline tag.
    /// - `full`: fetch the full model information including all fields.
    /// - `card_data`: include the model card metadata in the response.
    /// - `fetch_config`: include the model configuration in the response.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_models(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        #[builder(into)] pipeline_tag: Option<String>,
        full: Option<bool>,
        card_data: Option<bool>,
        fetch_config: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<ModelInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/models", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit {
            // The Hub API usually returns up to 1000 items per page by default,
            // so only set an explicit limit for smaller requests.
            if max < 1000 {
                query.push(("limit".into(), max.to_string()));
            }
        }
        if let Some(ref pt) = pipeline_tag {
            query.push(("pipeline_tag".into(), pt.clone()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        if card_data == Some(true) {
            query.push(("cardData".into(), "true".into()));
        }
        if fetch_config == Some(true) {
            query.push(("config".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// List datasets on the Hub. Endpoint: `GET /api/datasets`.
    ///
    /// Returns a stream of [`DatasetInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`, `author`, `filter`, `sort`: query filters and sort key.
    /// - `full`: fetch the full dataset information including all fields.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_datasets(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<DatasetInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/datasets", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit
            && max < 1000
        {
            query.push(("limit".into(), max.to_string()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// List Spaces on the Hub. Endpoint: `GET /api/spaces`.
    ///
    /// Returns a stream of [`SpaceInfo`] entries. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `search`, `author`, `filter`, `sort`: query filters and sort key.
    /// - `full`: fetch the full Space information including all fields.
    /// - `limit`: cap on the total number of items yielded by the stream. When less than 1000, also used as the server
    ///   page size.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_spaces(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<SpaceInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/spaces", self.endpoint()))?;
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = search {
            query.push(("search".into(), s.clone()));
        }
        if let Some(ref a) = author {
            query.push(("author".into(), a.clone()));
        }
        if let Some(ref f) = filter {
            query.push(("filter".into(), f.clone()));
        }
        if let Some(ref s) = sort {
            query.push(("sort".into(), s.clone()));
        }
        if let Some(max) = limit
            && max < 1000
        {
            query.push(("limit".into(), max.to_string()));
        }
        if full == Some(true) {
            query.push(("full".into(), "true".into()));
        }
        Ok(self.paginate(url, query, limit))
    }

    /// Create a new repository. Endpoint: `POST /api/repos/create`.
    ///
    /// # Parameters
    ///
    /// - `repo_id` (required): repository ID in `"owner/name"` or `"name"` format.
    /// - `repo_type`: type of repository to create (model, dataset, space, kernel).
    /// - `private`: whether the repository should be private.
    /// - `exist_ok` (default `false`): if `true`, do not error when the repository already exists.
    /// - `space_sdk`: SDK for a Space (e.g. `"gradio"`, `"streamlit"`, `"docker"`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_repo(
        &self,
        #[builder(into)] repo_id: String,
        repo_type: Option<RepoType>,
        private: Option<bool>,
        #[builder(default)] exist_ok: bool,
        #[builder(into)] space_sdk: Option<String>,
    ) -> HFResult<RepoUrl> {
        let url = format!("{}/api/repos/create", self.endpoint());

        let (namespace, name) = split_repo_id(&repo_id);

        let mut body = serde_json::json!({
            "name": name,
            "private": private.unwrap_or(false),
        });

        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(ref rt) = repo_type {
            body["type"] = serde_json::Value::String(rt.to_string());
        }
        if let Some(ref sdk) = space_sdk {
            body["sdk"] = serde_json::Value::String(sdk.clone());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        if response.status().as_u16() == 409 && exist_ok {
            let prefix = constants::repo_type_url_prefix(repo_type);
            return Ok(RepoUrl {
                url: format!("{}/{}{}", self.endpoint(), prefix, repo_id),
            });
        }

        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a repository. Endpoint: `DELETE /api/repos/delete`.
    ///
    /// # Parameters
    ///
    /// - `repo_id` (required): repository ID in `"owner/name"` or `"name"` format.
    /// - `repo_type`: type of repository.
    /// - `missing_ok` (default `false`): if `true`, do not error when the repository does not exist.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_repo(
        &self,
        #[builder(into)] repo_id: String,
        repo_type: Option<RepoType>,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        let url = format!("{}/api/repos/delete", self.endpoint());

        let (namespace, name) = split_repo_id(&repo_id);

        let mut body = serde_json::json!({ "name": name });
        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(ref rt) = repo_type {
            body["type"] = serde_json::Value::String(rt.to_string());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().delete(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(&repo_id), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Move (rename) a repository. Endpoint: `POST /api/repos/move`.
    ///
    /// # Parameters
    ///
    /// - `from_id` (required): current repository ID in `"owner/name"` format.
    /// - `to_id` (required): new repository ID in `"owner/name"` format.
    /// - `repo_type`: type of repository to move.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn move_repo(
        &self,
        #[builder(into)] from_id: String,
        #[builder(into)] to_id: String,
        repo_type: Option<RepoType>,
    ) -> HFResult<RepoUrl> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let mut body = serde_json::json!({
            "fromRepo": from_id,
            "toRepo": to_id,
        });
        if let Some(ref rt) = repo_type {
            body["type"] = serde_json::Value::String(rt.to_string());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        self.check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        let prefix = constants::repo_type_url_prefix(repo_type);
        Ok(RepoUrl {
            url: format!("{}/{}{}", self.endpoint(), prefix, to_id),
        })
    }
}

impl HFRepository {
    /// Construct a new repository handle. Prefer the factory methods on [`HFClient`] instead.
    pub fn new(client: HFClient, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
            repo_type,
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The repository owner (user or organization name).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The repository name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` identifier used in Hub API calls.
    ///
    /// If no owner is set, returns just the name (for repos using short-form IDs like `"gpt2"`).
    pub fn repo_path(&self) -> String {
        if self.owner.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.owner, self.name)
        }
    }

    /// The type of this repository (model, dataset, or space).
    pub fn repo_type(&self) -> RepoType {
        self.repo_type
    }

    /// Fetch repo info for this repository's `repo_type`, deserializing into `T`.
    /// Endpoint: GET /api/{repo_type}s/{repo_id}[/revision/{revision}]
    async fn fetch_repo_info<T: DeserializeOwned>(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<T> {
        let mut url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        if let Some(ref revision) = revision {
            url = format!("{url}/revision/{revision}");
        }
        let headers = self.hf_client.auth_headers();
        let expand_params: Option<Vec<(&str, &str)>> =
            expand.as_ref().map(|e| e.iter().map(|v| ("expand", v.as_str())).collect());
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self.hf_client.http_client().get(&url).headers(headers.clone());
            if let Some(ref params) = expand_params {
                req = req.query(params);
            }
            req.send()
        })
        .await?;
        let repo_path = self.repo_path();
        let not_found_ctx = match revision {
            Some(rev) => crate::error::NotFoundContext::Revision { revision: rev },
            None => crate::error::NotFoundContext::Repo,
        };
        let response = self.hf_client.check_response(response, Some(&repo_path), not_found_ctx).await?;
        Ok(response.json().await?)
    }

    pub(crate) async fn model_info(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<ModelInfo> {
        self.fetch_repo_info(revision, expand).await
    }

    pub(crate) async fn dataset_info(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<DatasetInfo> {
        self.fetch_repo_info(revision, expand).await
    }

    pub(crate) async fn space_info(
        &self,
        revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<SpaceInfo> {
        self.fetch_repo_info(revision, expand).await
    }
}

#[bon]
impl HFRepository {
    /// Fetch repository metadata, returning the appropriate [`RepoInfo`] variant.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision (branch, tag, or commit SHA). Defaults to the main branch.
    /// - `expand`: list of properties to expand in the response (e.g. `"trendingScore"`, `"cardData"`). When set, only
    ///   the listed properties (plus `_id` and `id`) are returned.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn info(
        &self,
        #[builder(into)] revision: Option<String>,
        expand: Option<Vec<String>>,
    ) -> HFResult<RepoInfo> {
        match self.repo_type {
            RepoType::Model => self.model_info(revision, expand).await.map(RepoInfo::Model),
            RepoType::Dataset => self.dataset_info(revision, expand).await.map(RepoInfo::Dataset),
            RepoType::Space => self.space_info(revision, expand).await.map(RepoInfo::Space),
            RepoType::Kernel => {
                Err(HFError::Other("Repository info is not implemented yet for kernel repositories".to_string()))
            },
        }
    }

    /// Return `true` if the repository exists and is accessible with the current credentials.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn exists(&self) -> HFResult<bool> {
        let url = self.hf_client.api_url(Some(self.repo_type), &self.repo_path());
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Generic)
            .await?;
        Ok(true)
    }

    /// Return `true` if the given revision (branch, tag, or commit SHA) exists.
    ///
    /// # Parameters
    ///
    /// - `revision` (required): Git revision to check for existence.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn revision_exists(&self, #[builder(into)] revision: String) -> HFResult<bool> {
        let url = format!("{}/revision/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), revision);
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Generic)
            .await?;
        Ok(true)
    }

    /// Return `true` if the given file exists in the repository at the specified revision.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to check within the repository.
    /// - `revision`: Git revision to check. Defaults to the main branch.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn file_exists(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<bool> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &self.repo_path(), revision, &filename);
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().head(&url).headers(headers.clone()).send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            if self.revision_exists().revision(revision.to_string()).send().await? {
                return Ok(false);
            }
            return Err(HFError::RevisionNotFound {
                repo_id: self.repo_path(),
                revision: revision.to_string(),
                context: None,
            });
        }
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Generic)
            .await?;
        Ok(true)
    }

    /// Update repository settings such as visibility, gating policy, description,
    /// discussion settings, and gated notification preferences.
    ///
    /// Endpoint: `PUT /api/{repo_type}s/{repo_id}/settings`.
    ///
    /// # Parameters
    ///
    /// - `private`: whether the repository should be private.
    /// - `gated`: access-gating mode for the repository (e.g. `auto`, `manual`, disabled).
    /// - `description`: repository description shown on the Hub page.
    /// - `discussions_disabled`: whether discussions are disabled on this repository.
    /// - `gated_notifications_email`: email address to receive gated-access request notifications.
    /// - `gated_notifications_mode`: when to send gated-access notifications (e.g. `bulk`, `real-time`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn update_settings(
        &self,
        private: Option<bool>,
        gated: Option<GatedApprovalMode>,
        #[builder(into)] description: Option<String>,
        discussions_disabled: Option<bool>,
        #[builder(into)] gated_notifications_email: Option<String>,
        gated_notifications_mode: Option<GatedNotificationsMode>,
    ) -> HFResult<()> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct UpdateSettingsBody<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            private: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated: Option<&'a GatedApprovalMode>,
            #[serde(skip_serializing_if = "Option::is_none")]
            description: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            discussions_disabled: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated_notifications_email: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            gated_notifications_mode: Option<&'a GatedNotificationsMode>,
        }

        let body = UpdateSettingsBody {
            private,
            gated: gated.as_ref(),
            description: description.as_deref(),
            discussions_disabled,
            gated_notifications_email: gated_notifications_email.as_deref(),
            gated_notifications_mode: gated_notifications_mode.as_ref(),
        };

        let url = format!("{}/settings", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()));
        let headers = self.hf_client.auth_headers();

        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .put(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(response, Some(&repo_path), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }
}

/// Split "namespace/name" into (Some("namespace"), "name") or (None, "name")
fn split_repo_id(repo_id: &str) -> (Option<&str>, &str) {
    match repo_id.split_once('/') {
        Some((ns, name)) => (Some(ns), name),
        None => (None, repo_id),
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::list_models`]. Returns the collected stream as a
    /// `Vec<ModelInfo>`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_models(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        #[builder(into)] pipeline_tag: Option<String>,
        full: Option<bool>,
        card_data: Option<bool>,
        fetch_config: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<ModelInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_models()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_pipeline_tag(pipeline_tag)
                .maybe_full(full)
                .maybe_card_data(card_data)
                .maybe_fetch_config(fetch_config)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_datasets`]. Returns the collected stream as a
    /// `Vec<DatasetInfo>`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_datasets(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<DatasetInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_datasets()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_full(full)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_spaces`]. Returns the collected stream as a
    /// `Vec<SpaceInfo>`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_spaces(
        &self,
        #[builder(into)] search: Option<String>,
        #[builder(into)] author: Option<String>,
        #[builder(into)] filter: Option<String>,
        #[builder(into)] sort: Option<String>,
        full: Option<bool>,
        limit: Option<usize>,
    ) -> HFResult<Vec<SpaceInfo>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_spaces()
                .maybe_search(search)
                .maybe_author(author)
                .maybe_filter(filter)
                .maybe_sort(sort)
                .maybe_full(full)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::create_repo`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_repo(
        &self,
        #[builder(into)] repo_id: String,
        repo_type: Option<RepoType>,
        private: Option<bool>,
        #[builder(default)] exist_ok: bool,
        #[builder(into)] space_sdk: Option<String>,
    ) -> HFResult<RepoUrl> {
        self.runtime.block_on(
            self.inner
                .create_repo()
                .repo_id(repo_id)
                .maybe_repo_type(repo_type)
                .maybe_private(private)
                .exist_ok(exist_ok)
                .maybe_space_sdk(space_sdk)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::delete_repo`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_repo(
        &self,
        #[builder(into)] repo_id: String,
        repo_type: Option<RepoType>,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .delete_repo()
                .repo_id(repo_id)
                .maybe_repo_type(repo_type)
                .missing_ok(missing_ok)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::move_repo`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn move_repo(
        &self,
        #[builder(into)] from_id: String,
        #[builder(into)] to_id: String,
        repo_type: Option<RepoType>,
    ) -> HFResult<RepoUrl> {
        self.runtime.block_on(
            self.inner
                .move_repo()
                .from_id(from_id)
                .to_id(to_id)
                .maybe_repo_type(repo_type)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync {
    /// Blocking counterpart of [`HFRepository::info`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn info(&self, #[builder(into)] revision: Option<String>, expand: Option<Vec<String>>) -> HFResult<RepoInfo> {
        self.runtime
            .block_on(self.inner.info().maybe_revision(revision).maybe_expand(expand).send())
    }

    /// Blocking counterpart of [`HFRepository::exists`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn exists(&self) -> HFResult<bool> {
        self.runtime.block_on(self.inner.exists().send())
    }

    /// Blocking counterpart of [`HFRepository::revision_exists`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn revision_exists(&self, #[builder(into)] revision: String) -> HFResult<bool> {
        self.runtime.block_on(self.inner.revision_exists().revision(revision).send())
    }

    /// Blocking counterpart of [`HFRepository::file_exists`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn file_exists(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<bool> {
        self.runtime
            .block_on(self.inner.file_exists().filename(filename).maybe_revision(revision).send())
    }

    /// Blocking counterpart of [`HFRepository::update_settings`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn update_settings(
        &self,
        private: Option<bool>,
        gated: Option<GatedApprovalMode>,
        #[builder(into)] description: Option<String>,
        discussions_disabled: Option<bool>,
        #[builder(into)] gated_notifications_email: Option<String>,
        gated_notifications_mode: Option<GatedNotificationsMode>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .update_settings()
                .maybe_private(private)
                .maybe_gated(gated)
                .maybe_description(description)
                .maybe_discussions_disabled(discussions_disabled)
                .maybe_gated_notifications_email(gated_notifications_email)
                .maybe_gated_notifications_mode(gated_notifications_mode)
                .send(),
        )
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::{HFRepository, RepoType, split_repo_id};
    use crate::client::HFClient;

    #[test]
    fn test_repo_path_and_accessors() {
        let client = crate::HFClient::builder().build().unwrap();
        let repo = HFRepository::new(client, RepoType::Model, "openai-community", "gpt2");

        assert_eq!(repo.owner(), "openai-community");
        assert_eq!(repo.name(), "gpt2");
        assert_eq!(repo.repo_path(), "openai-community/gpt2");
        assert_eq!(repo.repo_type(), RepoType::Model);
    }

    #[test]
    fn test_repo_type_from_str() {
        assert_eq!("model".parse::<RepoType>().unwrap(), RepoType::Model);
        assert_eq!("dataset".parse::<RepoType>().unwrap(), RepoType::Dataset);
        assert_eq!("space".parse::<RepoType>().unwrap(), RepoType::Space);
        assert_eq!("kernel".parse::<RepoType>().unwrap(), RepoType::Kernel);
        assert_eq!("MODEL".parse::<RepoType>().unwrap(), RepoType::Model);
        assert_eq!("KERNEL".parse::<RepoType>().unwrap(), RepoType::Kernel);
        assert!("invalid".parse::<RepoType>().is_err());
    }

    #[test]
    fn test_repo_type_display() {
        assert_eq!(RepoType::Model.to_string(), "model");
        assert_eq!(RepoType::Dataset.to_string(), "dataset");
        assert_eq!(RepoType::Space.to_string(), "space");
        assert_eq!(RepoType::Kernel.to_string(), "kernel");
    }

    #[test]
    fn test_split_repo_id() {
        assert_eq!(split_repo_id("user/repo"), (Some("user"), "repo"));
        assert_eq!(split_repo_id("repo"), (None, "repo"));
        assert_eq!(split_repo_id("org/sub/repo"), (Some("org"), "sub/repo"));
    }

    #[tokio::test]
    async fn test_list_models_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_models().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_datasets_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_datasets().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_spaces_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let stream = client.list_spaces().limit(0_usize).send().unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }
}
