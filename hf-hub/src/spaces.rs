//! Space handles, response types, and runtime/hardware/secrets APIs.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use bon::bon;
use serde::Deserialize;

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

pub(crate) mod _handle {
    use std::sync::Arc;

    #[allow(unused_imports)]
    use super::{HFClient, HFRepository, RepoType};

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
    /// let exists = space.exists().send().await?;
    /// # Ok(()) }
    /// ```
    #[derive(Clone)]
    pub struct HFSpace {
        pub(crate) repo: Arc<HFRepository>,
    }
}

pub(crate) use _handle::HFSpace;

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
}

#[bon]
impl HFSpace {
    /// Fetch the current runtime state of the Space (hardware, stage, URL, etc.).
    ///
    /// Endpoint: `GET /api/spaces/{repo_id}/runtime`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use hf_hub::HFClient;
    /// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
    /// let client = HFClient::builder().build()?;
    /// let runtime = client.space("owner", "name").runtime().send().await?;
    /// # let _ = runtime;
    /// # Ok(()) }
    /// ```
    #[builder(finish_fn = send)]
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/hardware`.
    ///
    /// # Parameters
    ///
    /// - `hardware` (required): hardware flavor to request (e.g. `"cpu-basic"`, `"t4-small"`, `"a10g-small"`).
    /// - `sleep_time`: number of seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(finish_fn = send)]
    pub async fn request_hardware(
        &self,
        #[builder(into)] hardware: String,
        sleep_time: Option<u64>,
    ) -> HFResult<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/hardware", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({ "flavor": hardware });
        if let Some(sleep_time) = sleep_time {
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/sleeptime`.
    ///
    /// # Parameters
    ///
    /// - `sleep_time` (required): seconds of inactivity before the Space is put to sleep. `0` means never sleep.
    #[builder(finish_fn = send)]
    pub async fn set_sleep_time(&self, sleep_time: u64) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/sleeptime", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "seconds": sleep_time });
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/pause`.
    #[builder(finish_fn = send)]
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/restart`.
    #[builder(finish_fn = send)]
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/secrets`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): secret key name.
    /// - `value` (required): secret value.
    /// - `description`: human-readable description of the secret.
    #[builder(finish_fn = send)]
    pub async fn add_secret(
        &self,
        #[builder(into)] key: String,
        #[builder(into)] value: String,
        #[builder(into)] description: Option<String>,
    ) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({ "key": key, "value": value });
        if let Some(ref desc) = description {
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
    ///
    /// Endpoint: `DELETE /api/spaces/{repo_id}/secrets`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): secret key name to delete.
    #[builder(finish_fn = send)]
    pub async fn delete_secret(&self, #[builder(into)] key: String) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": key });
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/variables`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): variable key name.
    /// - `value` (required): variable value.
    /// - `description`: human-readable description of the variable.
    #[builder(finish_fn = send)]
    pub async fn add_variable(
        &self,
        #[builder(into)] key: String,
        #[builder(into)] value: String,
        #[builder(into)] description: Option<String>,
    ) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({ "key": key, "value": value });
        if let Some(ref desc) = description {
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
    ///
    /// Endpoint: `DELETE /api/spaces/{repo_id}/variables`.
    ///
    /// # Parameters
    ///
    /// - `key` (required): variable key name to delete.
    #[builder(finish_fn = send)]
    pub async fn delete_variable(&self, #[builder(into)] key: String) -> HFResult<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": key });
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
    ///
    /// Endpoint: `POST /api/spaces/{repo_id}/duplicate`.
    ///
    /// # Parameters
    ///
    /// - `to_id`: destination repository ID in `"owner/name"` format. Defaults to the authenticated user's namespace
    ///   with the same name.
    /// - `private`: whether the duplicated Space should be private.
    /// - `hardware`: hardware to run the duplicated Space on (e.g. `"cpu-basic"`, `"t4-small"`).
    /// - `storage`: persistent storage tier (e.g. `"small"`, `"medium"`, `"large"`).
    /// - `sleep_time`: seconds of inactivity before the Space is put to sleep. `0` means never.
    /// - `secrets`: secrets to set on the duplicated Space (list of JSON objects with `key` and `value`).
    /// - `variables`: environment variables to set on the duplicated Space (list of JSON objects with `key` and
    ///   `value`).
    #[builder(finish_fn = send)]
    pub async fn duplicate(
        &self,
        #[builder(into)] to_id: Option<String>,
        private: Option<bool>,
        #[builder(into)] hardware: Option<String>,
        #[builder(into)] storage: Option<String>,
        sleep_time: Option<u64>,
        secrets: Option<Vec<serde_json::Value>>,
        variables: Option<Vec<serde_json::Value>>,
    ) -> HFResult<RepoUrl> {
        let url = format!("{}/api/spaces/{}/duplicate", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::Map::new();
        if let Some(ref to_id) = to_id {
            body.insert("repository".into(), serde_json::json!(to_id));
        }
        if let Some(private) = private {
            body.insert("private".into(), serde_json::json!(private));
        }
        if let Some(ref hw) = hardware {
            body.insert("hardware".into(), serde_json::json!(hw));
        }
        if let Some(ref storage) = storage {
            body.insert("storage".into(), serde_json::json!(storage));
        }
        if let Some(sleep_time) = sleep_time {
            body.insert("sleepTime".into(), serde_json::json!(sleep_time));
        }
        if let Some(ref secrets) = secrets {
            body.insert("secrets".into(), serde_json::json!(secrets));
        }
        if let Some(ref variables) = variables {
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

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFSpaceSync {
    /// Blocking counterpart of [`HFSpace::runtime`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn runtime(&self) -> HFResult<SpaceRuntime> {
        self.repo_sync.runtime.block_on(self.inner.runtime().send())
    }

    /// Blocking counterpart of [`HFSpace::request_hardware`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn request_hardware(
        &self,
        #[builder(into)] hardware: String,
        sleep_time: Option<u64>,
    ) -> HFResult<SpaceRuntime> {
        self.repo_sync.runtime.block_on(
            self.inner
                .request_hardware()
                .hardware(hardware)
                .maybe_sleep_time(sleep_time)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFSpace::set_sleep_time`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn set_sleep_time(&self, sleep_time: u64) -> HFResult<()> {
        self.repo_sync
            .runtime
            .block_on(self.inner.set_sleep_time().sleep_time(sleep_time).send())
    }

    /// Blocking counterpart of [`HFSpace::pause`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn pause(&self) -> HFResult<SpaceRuntime> {
        self.repo_sync.runtime.block_on(self.inner.pause().send())
    }

    /// Blocking counterpart of [`HFSpace::restart`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn restart(&self) -> HFResult<SpaceRuntime> {
        self.repo_sync.runtime.block_on(self.inner.restart().send())
    }

    /// Blocking counterpart of [`HFSpace::add_secret`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn add_secret(
        &self,
        #[builder(into)] key: String,
        #[builder(into)] value: String,
        #[builder(into)] description: Option<String>,
    ) -> HFResult<()> {
        self.repo_sync.runtime.block_on(
            self.inner
                .add_secret()
                .key(key)
                .value(value)
                .maybe_description(description)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFSpace::delete_secret`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn delete_secret(&self, #[builder(into)] key: String) -> HFResult<()> {
        self.repo_sync.runtime.block_on(self.inner.delete_secret().key(key).send())
    }

    /// Blocking counterpart of [`HFSpace::add_variable`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn add_variable(
        &self,
        #[builder(into)] key: String,
        #[builder(into)] value: String,
        #[builder(into)] description: Option<String>,
    ) -> HFResult<()> {
        self.repo_sync.runtime.block_on(
            self.inner
                .add_variable()
                .key(key)
                .value(value)
                .maybe_description(description)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFSpace::delete_variable`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send)]
    pub fn delete_variable(&self, #[builder(into)] key: String) -> HFResult<()> {
        self.repo_sync.runtime.block_on(self.inner.delete_variable().key(key).send())
    }

    /// Blocking counterpart of [`HFSpace::duplicate`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send)]
    pub fn duplicate(
        &self,
        #[builder(into)] to_id: Option<String>,
        private: Option<bool>,
        #[builder(into)] hardware: Option<String>,
        #[builder(into)] storage: Option<String>,
        sleep_time: Option<u64>,
        secrets: Option<Vec<serde_json::Value>>,
        variables: Option<Vec<serde_json::Value>>,
    ) -> HFResult<RepoUrl> {
        self.repo_sync.runtime.block_on(
            self.inner
                .duplicate()
                .maybe_to_id(to_id)
                .maybe_private(private)
                .maybe_hardware(hardware)
                .maybe_storage(storage)
                .maybe_sleep_time(sleep_time)
                .maybe_secrets(secrets)
                .maybe_variables(variables)
                .send(),
        )
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
