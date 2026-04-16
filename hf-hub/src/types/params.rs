use typed_builder::TypedBuilder;

use super::repo::RepoType;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XetTokenType {
    Read,
    Write,
}

impl XetTokenType {
    pub fn as_str(&self) -> &'static str {
        match self {
            XetTokenType::Read => "read",
            XetTokenType::Write => "write",
        }
    }
}

#[derive(TypedBuilder)]
pub struct GetXetTokenParams {
    /// Repository ID in `"owner/name"` or `"name"` format.
    #[builder(setter(into))]
    pub repo_id: String,
    /// Whether to request a read or write token.
    pub token_type: XetTokenType,
    /// Type of repository (model, dataset, or space).
    #[builder(default, setter(into, strip_option))]
    pub repo_type: Option<RepoType>,
    /// Git revision to scope the token to.
    #[builder(default, setter(into, strip_option))]
    pub revision: Option<String>,
}

#[derive(TypedBuilder)]
pub struct DuplicateSpaceParams {
    /// Destination repository ID in `"owner/name"` format. Defaults to the authenticated user's namespace with the
    /// same name.
    #[builder(default, setter(into, strip_option))]
    pub to_id: Option<String>,
    /// Whether the duplicated Space should be private.
    #[builder(default, setter(strip_option))]
    pub private: Option<bool>,
    /// Hardware to run the duplicated Space on (e.g. `"cpu-basic"`, `"t4-small"`).
    #[builder(default, setter(into, strip_option))]
    pub hardware: Option<String>,
    /// Persistent storage tier for the duplicated Space (e.g. `"small"`, `"medium"`, `"large"`).
    #[builder(default, setter(into, strip_option))]
    pub storage: Option<String>,
    /// Number of seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(default, setter(strip_option))]
    pub sleep_time: Option<u64>,
    /// Secrets to set on the duplicated Space (list of JSON objects with `key` and `value`).
    #[builder(default, setter(into, strip_option))]
    pub secrets: Option<Vec<serde_json::Value>>,
    /// Environment variables to set on the duplicated Space (list of JSON objects with `key` and `value`).
    #[builder(default, setter(into, strip_option))]
    pub variables: Option<Vec<serde_json::Value>>,
}
