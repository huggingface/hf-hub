//! Spaces component: the [`HFSpace`] handle, Space-specific parameter structs,
//! Space response types, and all Space API operations (runtime, hardware,
//! secrets, variables, duplicate, pause/restart).

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use serde::Deserialize;
use typed_builder::TypedBuilder;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::repository::{HFRepository, RepoType, RepoUrl};
use crate::retry;

/// Runtime state of a Space: stage, hardware, storage, and replica info.
///
/// Returned by Space lifecycle methods such as
/// [`HFSpace::runtime`], [`HFSpace::pause`], and [`HFSpace::restart`]. The `raw`
/// field preserves the full JSON payload for fields not modeled explicitly.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceRuntime {
    pub stage: Option<String>,
    pub hardware: Option<serde_json::Value>,
    pub storage: Option<serde_json::Value>,
    pub sleep_time: Option<u64>,
    pub replicas: Option<serde_json::Value>,
    #[serde(default)]
    pub raw: serde_json::Value,
}

/// A public environment variable set on a Space (non-secret).
///
/// Secrets are not returned — only variables declared via the Space's variables API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpaceVariable {
    pub key: String,
    pub value: Option<String>,
    pub description: Option<String>,
    pub updated_at: Option<String>,
}

/// Parameters for duplicating a Space.
///
/// Used with [`HFSpace::duplicate`].
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

/// Parameters for requesting a hardware change on a Space.
///
/// Used with [`HFSpace::request_hardware`].
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
/// Used with [`HFSpace::set_sleep_time`].
#[derive(TypedBuilder)]
pub struct SpaceSleepTimeParams {
    /// Number of seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    pub sleep_time: u64,
}

/// Parameters for adding or updating a secret on a Space.
///
/// Used with [`HFSpace::add_secret`].
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
/// Used with [`HFSpace::delete_secret`].
#[derive(TypedBuilder)]
pub struct SpaceSecretDeleteParams {
    /// Secret key name to delete.
    #[builder(setter(into))]
    pub key: String,
}

/// Parameters for adding or updating a public variable on a Space.
///
/// Used with [`HFSpace::add_variable`].
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
/// Used with [`HFSpace::delete_variable`].
#[derive(TypedBuilder)]
pub struct SpaceVariableDeleteParams {
    /// Variable key name to delete.
    #[builder(setter(into))]
    pub key: String,
}

/// A handle for a Space repository, providing Space-specific operations on top of [`HFRepository`].
///
/// `HFSpace` wraps an [`HFRepository`] fixed to [`RepoType::Space`] and exposes hardware,
/// secret, and variable management. It derefs to [`HFRepository`], so all general repo
/// methods (e.g. `exists`, `info`, `download_file`) are accessible directly.
///
/// Created via [`HFClient::space`] or [`TryFrom<HFRepository>`].
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let space = client.space("huggingface", "diffusers-gallery");
/// // General repo methods are available via Deref:
/// let exists = space.exists().await?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFSpace {
    pub(crate) repo: Arc<HFRepository>,
}

impl fmt::Debug for HFSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFSpace").field("repo", &self.repo).finish()
    }
}

impl HFClient {
    /// Create an [`HFSpace`] handle for a Space repository.
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFSpace {
        HFSpace::new(self.clone(), owner, name)
    }
}

impl HFSpace {
    /// Construct a new Space handle. Prefer [`HFClient::space`] in most cases.
    pub fn new(client: HFClient, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            repo: Arc::new(HFRepository::new(client, RepoType::Space, owner, name)),
        }
    }

    pub fn repo(&self) -> &HFRepository {
        &self.repo
    }

    /// Fetch the current runtime state of the Space (hardware, stage, URL, etc.).
    pub async fn runtime(&self) -> HFResult<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/runtime", self.hf_client.endpoint(), self.repo_path());
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Request an upgrade or downgrade of the Space's hardware tier.
    pub async fn request_hardware(&self, params: SpaceHardwareRequestParams) -> HFResult<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/hardware", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({ "flavor": params.hardware });
        if let Some(sleep_time) = params.sleep_time {
            body["sleepTime"] = serde_json::json!(sleep_time);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Configure the number of seconds of inactivity before the Space is put to sleep.
    pub async fn set_sleep_time(&self, params: SpaceSleepTimeParams) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/sleeptime", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "seconds": params.sleep_time });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Pause the Space, stopping it from consuming compute resources.
    pub async fn pause(&self) -> HFResult<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/pause", self.hf_client.endpoint(), self.repo_path());
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().post(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Restart a paused or errored Space.
    pub async fn restart(&self) -> HFResult<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/restart", self.hf_client.endpoint(), self.repo_path());
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client.http_client().post(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Add or update a secret (encrypted environment variable) on the Space.
    pub async fn add_secret(&self, params: SpaceSecretParams) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({
            "key": params.key,
            "value": params.value,
        });
        if let Some(ref desc) = params.description {
            body["description"] = serde_json::json!(desc);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a secret from the Space by key.
    pub async fn delete_secret(&self, params: SpaceSecretDeleteParams) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": params.key });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Add or update a public environment variable on the Space.
    pub async fn add_variable(&self, params: SpaceVariableParams) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({
            "key": params.key,
            "value": params.value,
        });
        if let Some(ref desc) = params.description {
            body["description"] = serde_json::json!(desc);
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a public environment variable from the Space by key.
    pub async fn delete_variable(&self, params: SpaceVariableDeleteParams) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": params.key });
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Duplicate this Space to a new repository.
    pub async fn duplicate(&self, params: DuplicateSpaceParams) -> HFResult<RepoUrl> {
        let url = format!("{}/api/spaces/{}/duplicate", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::Map::new();
        if let Some(ref to_id) = params.to_id {
            body.insert("repository".into(), serde_json::json!(to_id));
        }
        if let Some(private) = params.private {
            body.insert("private".into(), serde_json::json!(private));
        }
        if let Some(ref hw) = params.hardware {
            body.insert("hardware".into(), serde_json::json!(hw));
        }
        if let Some(ref storage) = params.storage {
            body.insert("storage".into(), serde_json::json!(storage));
        }
        if let Some(sleep_time) = params.sleep_time {
            body.insert("sleepTime".into(), serde_json::json!(sleep_time));
        }
        if let Some(ref secrets) = params.secrets {
            body.insert("secrets".into(), serde_json::json!(secrets));
        }
        if let Some(ref variables) = params.variables {
            body.insert("variables".into(), serde_json::json!(variables));
        }
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }
}

impl TryFrom<HFRepository> for HFSpace {
    type Error = HFError;

    fn try_from(repo: HFRepository) -> HFResult<Self> {
        if repo.repo_type() != RepoType::Space {
            return Err(HFError::InvalidRepoType {
                expected: RepoType::Space,
                actual: repo.repo_type(),
            });
        }
        Ok(Self { repo: Arc::new(repo) })
    }
}

impl From<HFSpace> for Arc<HFRepository> {
    fn from(space: HFSpace) -> Self {
        space.repo.clone()
    }
}

impl Deref for HFSpace {
    type Target = HFRepository;

    fn deref(&self) -> &Self::Target {
        &self.repo
    }
}

sync_api! {
    impl HFSpace -> HFSpaceSync {
        fn runtime(&self) -> HFResult<SpaceRuntime>;
        fn request_hardware(&self, params: SpaceHardwareRequestParams) -> HFResult<SpaceRuntime>;
        fn set_sleep_time(&self, params: SpaceSleepTimeParams) -> HFResult<()>;
        fn pause(&self) -> HFResult<SpaceRuntime>;
        fn restart(&self) -> HFResult<SpaceRuntime>;
        fn add_secret(&self, params: SpaceSecretParams) -> HFResult<()>;
        fn delete_secret(&self, params: SpaceSecretDeleteParams) -> HFResult<()>;
        fn add_variable(&self, params: SpaceVariableParams) -> HFResult<()>;
        fn delete_variable(&self, params: SpaceVariableDeleteParams) -> HFResult<()>;
        fn duplicate(&self, params: DuplicateSpaceParams) -> HFResult<RepoUrl>;
    }
}

#[cfg(test)]
mod tests {
    use super::{HFSpace, SpaceRuntime, SpaceVariable};
    use crate::repository::{HFRepository, RepoType};

    #[test]
    fn test_hfspace_constructor_and_deref() {
        let client = crate::HFClient::builder().build().unwrap();
        let space = HFSpace::new(client, "huggingface-projects", "diffusers-gallery");

        assert_eq!(space.repo_type(), RepoType::Space);
        assert_eq!(space.repo_path(), "huggingface-projects/diffusers-gallery");
    }

    #[test]
    fn test_hfspace_try_from_repo() {
        let client = crate::HFClient::builder().build().unwrap();
        let space_repo = HFRepository::new(client.clone(), RepoType::Space, "owner", "space");
        assert!(HFSpace::try_from(space_repo).is_ok());

        let model_repo = HFRepository::new(client, RepoType::Model, "owner", "model");
        let error = HFSpace::try_from(model_repo).unwrap_err();
        match error {
            crate::HFError::InvalidRepoType { expected, actual } => {
                assert_eq!(expected, RepoType::Space);
                assert_eq!(actual, RepoType::Model);
            },
            _ => panic!("expected invalid repo type error"),
        }
    }

    #[test]
    fn test_space_runtime_deserialize() {
        let json = r#"{"stage":"RUNNING","hardware":{"current":null,"requested":null},"storage":null,"replicas":{"requested":1,"current":1}}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.stage.as_deref(), Some("RUNNING"));
        assert!(runtime.hardware.is_some());
    }

    #[test]
    fn test_space_runtime_deserialize_minimal() {
        let json = r#"{"stage":"BUILDING"}"#;
        let runtime: SpaceRuntime = serde_json::from_str(json).unwrap();
        assert_eq!(runtime.stage.as_deref(), Some("BUILDING"));
        assert!(runtime.hardware.is_none());
    }

    #[test]
    fn test_space_variable_deserialize() {
        let json = r#"{"key":"MODEL_ID","value":"gpt2","description":"The model"}"#;
        let var: SpaceVariable = serde_json::from_str(json).unwrap();
        assert_eq!(var.key, "MODEL_ID");
        assert_eq!(var.value.as_deref(), Some("gpt2"));
    }
}
