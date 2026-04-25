//! Repository handles, parameter and metadata types, and list/create/delete/move APIs.

mod commits;
mod diff;
mod download;
mod files;
mod listing;
mod upload;

use std::fmt;
use std::str::FromStr;

pub use commits::{
    CommitAuthor, DiffEntry, GitCommitInfo, GitRefInfo, GitRefs, RepoCreateBranchParams, RepoCreateTagParams,
    RepoDeleteBranchParams, RepoDeleteTagParams, RepoGetCommitDiffParams, RepoGetRawDiffParams, RepoListCommitsParams,
    RepoListRefsParams,
};
pub use diff::{GitStatus, HFDiffParseError, HFFileDiff};
pub use files::{
    AddSource, BlobLfsInfo, CommitInfo, CommitOperation, FileMetadataInfo, LastCommitInfo, RepoCreateCommitParams,
    RepoDeleteFileParams, RepoDeleteFolderParams, RepoDownloadFileParams, RepoDownloadFileStreamParams,
    RepoDownloadFileStreamParamsBuilder, RepoDownloadFileToBytesParams, RepoDownloadFileToBytesParamsBuilder,
    RepoGetFileMetadataParams, RepoGetPathsInfoParams, RepoListFilesParams, RepoListTreeParams,
    RepoSnapshotDownloadParams, RepoTreeEntry, RepoUploadFileParams, RepoUploadFolderParams,
};
pub(crate) use files::{extract_file_size, extract_xet_hash};
use futures::Stream;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use typed_builder::TypedBuilder;
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
    /// let info = repo.info(Default::default()).await?;
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

/// Parameters for listing models on the Hub.
///
/// Used with [`HFClient::list_models`].
#[derive(TypedBuilder)]
pub struct ListModelsParams {
    /// Filter models by a text query (matches model IDs and descriptions).
    #[builder(default, setter(into, strip_option))]
    pub search: Option<String>,
    /// Filter models by author or organization name.
    #[builder(default, setter(into, strip_option))]
    pub author: Option<String>,
    /// Filter models by tags (e.g. `"text-generation"`, `"pytorch"`).
    #[builder(default, setter(into, strip_option))]
    pub filter: Option<String>,
    /// Property to sort results by (e.g. `"downloads"`, `"lastModified"`).
    #[builder(default, setter(into, strip_option))]
    pub sort: Option<String>,
    /// Filter models by pipeline tag (e.g. `"text-generation"`, `"image-classification"`).
    #[builder(default, setter(into, strip_option))]
    pub pipeline_tag: Option<String>,
    /// Whether to fetch the full model information including all fields.
    #[builder(default, setter(strip_option))]
    pub full: Option<bool>,
    /// Whether to include the model card metadata in the response.
    #[builder(default, setter(strip_option))]
    pub card_data: Option<bool>,
    /// Whether to include the model configuration in the response.
    #[builder(default, setter(strip_option))]
    pub fetch_config: Option<bool>,
    /// Cap on the total number of items returned.
    /// Pagination stops once this many items have been yielded.
    /// When less than 1000, also used as the server page size for efficiency.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for listing datasets on the Hub.
///
/// Used with [`HFClient::list_datasets`].
#[derive(TypedBuilder)]
pub struct ListDatasetsParams {
    /// Filter datasets by a text query (matches dataset IDs and descriptions).
    #[builder(default, setter(into, strip_option))]
    pub search: Option<String>,
    /// Filter datasets by author or organization name.
    #[builder(default, setter(into, strip_option))]
    pub author: Option<String>,
    /// Filter datasets by tags.
    #[builder(default, setter(into, strip_option))]
    pub filter: Option<String>,
    /// Property to sort results by (e.g. `"downloads"`, `"lastModified"`).
    #[builder(default, setter(into, strip_option))]
    pub sort: Option<String>,
    /// Whether to fetch the full dataset information including all fields.
    #[builder(default, setter(strip_option))]
    pub full: Option<bool>,
    /// Cap on the total number of items returned.
    /// Pagination stops once this many items have been yielded.
    /// When less than 1000, also used as the server page size for efficiency.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for listing Spaces on the Hub.
///
/// Used with [`HFClient::list_spaces`].
#[derive(TypedBuilder)]
pub struct ListSpacesParams {
    /// Filter spaces by a text query (matches space IDs and descriptions).
    #[builder(default, setter(into, strip_option))]
    pub search: Option<String>,
    /// Filter spaces by author or organization name.
    #[builder(default, setter(into, strip_option))]
    pub author: Option<String>,
    /// Filter spaces by tags.
    #[builder(default, setter(into, strip_option))]
    pub filter: Option<String>,
    /// Property to sort results by (e.g. `"downloads"`, `"lastModified"`).
    #[builder(default, setter(into, strip_option))]
    pub sort: Option<String>,
    /// Whether to fetch the full space information including all fields.
    #[builder(default, setter(strip_option))]
    pub full: Option<bool>,
    /// Cap on the total number of items returned.
    /// Pagination stops once this many items have been yielded.
    /// When less than 1000, also used as the server page size for efficiency.
    #[builder(default, setter(strip_option))]
    pub limit: Option<usize>,
}

/// Parameters for creating a new repository on the Hub.
///
/// Used with [`HFClient::create_repo`].
#[derive(TypedBuilder)]
pub struct CreateRepoParams {
    /// Repository ID in `"owner/name"` or `"name"` format.
    #[builder(setter(into))]
    pub repo_id: String,
    /// Type of repository to create (model, dataset, or space).
    #[builder(default, setter(into, strip_option))]
    pub repo_type: Option<RepoType>,
    /// Whether the repository should be private.
    #[builder(default, setter(strip_option))]
    pub private: Option<bool>,
    /// If `true`, do not error when the repository already exists.
    #[builder(default)]
    pub exist_ok: bool,
    /// SDK to use for a Space (e.g. `"gradio"`, `"streamlit"`, `"docker"`). Only applicable when creating a Space.
    #[builder(default, setter(into, strip_option))]
    pub space_sdk: Option<String>,
}

/// Parameters for deleting a repository on the Hub.
///
/// Used with [`HFClient::delete_repo`].
#[derive(TypedBuilder)]
pub struct DeleteRepoParams {
    /// Repository ID in `"owner/name"` or `"name"` format.
    #[builder(setter(into))]
    pub repo_id: String,
    /// Type of repository to delete (model, dataset, or space).
    #[builder(default, setter(into, strip_option))]
    pub repo_type: Option<RepoType>,
    /// If `true`, do not error when the repository does not exist.
    #[builder(default)]
    pub missing_ok: bool,
}

/// Parameters for renaming/moving a repository on the Hub.
///
/// Used with [`HFClient::move_repo`].
#[derive(TypedBuilder)]
pub struct MoveRepoParams {
    /// Current repository ID in `"owner/name"` format.
    #[builder(setter(into))]
    pub from_id: String,
    /// New repository ID in `"owner/name"` format.
    #[builder(setter(into))]
    pub to_id: String,
    /// Type of repository to move (model, dataset, or space).
    #[builder(default, setter(into, strip_option))]
    pub repo_type: Option<RepoType>,
}

/// Parameters for fetching repository info.
///
/// Used with [`HFRepository::info`].
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
/// Used with [`HFRepository::revision_exists`].
#[derive(TypedBuilder)]
pub struct RepoRevisionExistsParams {
    /// Git revision (branch, tag, or commit SHA) to check for existence.
    #[builder(setter(into))]
    pub revision: String,
}

/// Parameters for checking whether a file exists in a repository.
///
/// Used with [`HFRepository::file_exists`].
#[derive(TypedBuilder)]
pub struct RepoFileExistsParams {
    /// Path of the file to check within the repository.
    #[builder(setter(into))]
    pub filename: String,
    /// Git revision to check. Defaults to the main branch.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

/// Parameters for updating repository settings (visibility, gating, description, ...).
///
/// Used with [`HFRepository::update_settings`].
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

    /// List models on the Hub.
    /// Endpoint: GET /api/models
    pub fn list_models(&self, params: ListModelsParams) -> HFResult<impl Stream<Item = HFResult<ModelInfo>> + '_> {
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
    pub fn list_datasets(
        &self,
        params: ListDatasetsParams,
    ) -> HFResult<impl Stream<Item = HFResult<DatasetInfo>> + '_> {
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
    pub fn list_spaces(&self, params: ListSpacesParams) -> HFResult<impl Stream<Item = HFResult<SpaceInfo>> + '_> {
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
    pub async fn create_repo(&self, params: CreateRepoParams) -> HFResult<RepoUrl> {
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

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
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
    pub async fn delete_repo(&self, params: DeleteRepoParams) -> HFResult<()> {
        let url = format!("{}/api/repos/delete", self.endpoint());

        let (namespace, name) = split_repo_id(&params.repo_id);

        let mut body = serde_json::json!({ "name": name });
        if let Some(ns) = namespace {
            body["organization"] = serde_json::Value::String(ns.to_string());
        }
        if let Some(ref repo_type) = params.repo_type {
            body["type"] = serde_json::Value::String(repo_type.to_string());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().delete(&url).headers(headers.clone()).json(&body).send()
        })
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
    pub async fn move_repo(&self, params: MoveRepoParams) -> HFResult<RepoUrl> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let mut body = serde_json::json!({
            "fromRepo": params.from_id,
            "toRepo": params.to_id,
        });
        if let Some(ref repo_type) = params.repo_type {
            body["type"] = serde_json::Value::String(repo_type.to_string());
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().post(&url).headers(headers.clone()).json(&body).send()
        })
        .await?;

        self.check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        let prefix = constants::repo_type_url_prefix(params.repo_type);
        Ok(RepoUrl {
            url: format!("{}/{}{}", self.endpoint(), prefix, params.to_id),
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

    /// Fetch repository metadata, returning the appropriate [`RepoInfo`] variant.
    pub async fn info(&self, params: RepoInfoParams) -> HFResult<RepoInfo> {
        match self.repo_type {
            RepoType::Model => self
                .model_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Model),
            RepoType::Dataset => self
                .dataset_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Dataset),
            RepoType::Space => self
                .space_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Space),
            RepoType::Kernel => {
                Err(HFError::Other("Repository info is not implemented yet for kernel repositories".to_string()))
            },
        }
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

    /// Return `true` if the repository exists and is accessible with the current credentials.
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
    pub async fn revision_exists(&self, params: RepoRevisionExistsParams) -> HFResult<bool> {
        let url =
            format!("{}/revision/{}", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()), params.revision);
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
    pub async fn file_exists(&self, params: RepoFileExistsParams) -> HFResult<bool> {
        let revision = params.revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = self
            .hf_client
            .download_url(Some(self.repo_type), &self.repo_path(), revision, &params.filename);
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().head(&url).headers(headers.clone()).send()
        })
        .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            if self
                .revision_exists(RepoRevisionExistsParams::builder().revision(revision.to_string()).build())
                .await?
            {
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
    /// Endpoint: PUT /api/{repo_type}s/{repo_id}/settings
    pub async fn update_settings(&self, params: RepoUpdateSettingsParams) -> HFResult<()> {
        let url = format!("{}/settings", self.hf_client.api_url(Some(self.repo_type), &self.repo_path()));
        let headers = self.hf_client.auth_headers();

        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .put(&url)
                .headers(headers.clone())
                .json(&params)
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

sync_api! {
    impl HFClient -> HFClientSync {
        fn create_repo(&self, params: CreateRepoParams) -> HFResult<RepoUrl>;
        fn delete_repo(&self, params: DeleteRepoParams) -> HFResult<()>;
        fn move_repo(&self, params: MoveRepoParams) -> HFResult<RepoUrl>;
    }
}

sync_api_stream! {
    impl HFClient -> HFClientSync {
        fn list_models(&self, params: ListModelsParams) -> ModelInfo;
        fn list_datasets(&self, params: ListDatasetsParams) -> DatasetInfo;
        fn list_spaces(&self, params: ListSpacesParams) -> SpaceInfo;
    }
}

sync_api! {
    impl HFRepository -> HFRepositorySync {
        fn info(&self, params: RepoInfoParams) -> HFResult<RepoInfo>;
        fn exists(&self) -> HFResult<bool>;
        fn revision_exists(&self, params: RepoRevisionExistsParams) -> HFResult<bool>;
        fn file_exists(&self, params: RepoFileExistsParams) -> HFResult<bool>;
        fn update_settings(&self, params: RepoUpdateSettingsParams) -> HFResult<()>;
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::{HFRepository, ListDatasetsParams, ListModelsParams, ListSpacesParams, RepoType, split_repo_id};
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
        let params = ListModelsParams::builder().limit(0_usize).build();
        let stream = client.list_models(params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_datasets_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let params = ListDatasetsParams::builder().limit(0_usize).build();
        let stream = client.list_datasets(params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_list_spaces_limit_zero_returns_empty() {
        let client = HFClient::builder().build().unwrap();
        let params = ListSpacesParams::builder().limit(0_usize).build();
        let stream = client.list_spaces(params).unwrap();
        futures::pin_mut!(stream);
        assert!(stream.next().await.is_none());
    }
}
