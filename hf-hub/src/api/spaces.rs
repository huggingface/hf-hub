use crate::error::Result;
use crate::repository::HFSpace;
use crate::types::{
    DuplicateSpaceParams, RepoUrl, SpaceHardwareRequestParams, SpaceRuntime, SpaceSecretDeleteParams,
    SpaceSecretParams, SpaceSleepTimeParams, SpaceVariableDeleteParams, SpaceVariableParams,
};

impl HFSpace {
    /// Fetch the current runtime state of the Space (hardware, stage, URL, etc.).
    pub async fn runtime(&self) -> Result<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/runtime", self.hf_client.endpoint(), self.repo_path());
        let response = self
            .hf_client
            .http_client()
            .get(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Request an upgrade or downgrade of the Space's hardware tier.
    pub async fn request_hardware(&self, params: &SpaceHardwareRequestParams) -> Result<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/hardware", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({ "flavor": params.hardware });
        if let Some(sleep_time) = params.sleep_time {
            body["sleepTime"] = serde_json::json!(sleep_time);
        }
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Configure the number of seconds of inactivity before the Space is put to sleep.
    pub async fn set_sleep_time(&self, params: &SpaceSleepTimeParams) -> Result<()> {
        let url = format!("{}/api/spaces/{}/sleeptime", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "seconds": params.sleep_time });
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Pause the Space, stopping it from consuming compute resources.
    pub async fn pause(&self) -> Result<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/pause", self.hf_client.endpoint(), self.repo_path());
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Restart a paused or errored Space.
    pub async fn restart(&self) -> Result<SpaceRuntime> {
        let url = format!("{}/api/spaces/{}/restart", self.hf_client.endpoint(), self.repo_path());
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .send()
            .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }

    /// Add or update a secret (encrypted environment variable) on the Space.
    pub async fn add_secret(&self, params: &SpaceSecretParams) -> Result<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({
            "key": params.key,
            "value": params.value,
        });
        if let Some(ref desc) = params.description {
            body["description"] = serde_json::json!(desc);
        }
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a secret from the Space by key.
    pub async fn delete_secret(&self, params: &SpaceSecretDeleteParams) -> Result<()> {
        let url = format!("{}/api/spaces/{}/secrets", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": params.key });
        let response = self
            .hf_client
            .http_client()
            .delete(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Add or update a public environment variable on the Space.
    pub async fn add_variable(&self, params: &SpaceVariableParams) -> Result<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let mut body = serde_json::json!({
            "key": params.key,
            "value": params.value,
        });
        if let Some(ref desc) = params.description {
            body["description"] = serde_json::json!(desc);
        }
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Delete a public environment variable from the Space by key.
    pub async fn delete_variable(&self, params: &SpaceVariableDeleteParams) -> Result<()> {
        let url = format!("{}/api/spaces/{}/variables", self.hf_client.endpoint(), self.repo_path());
        let body = serde_json::json!({ "key": params.key });
        let response = self
            .hf_client
            .http_client()
            .delete(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        self.hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(())
    }

    /// Duplicate this Space to a new repository.
    pub async fn duplicate(&self, params: &DuplicateSpaceParams) -> Result<RepoUrl> {
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
        let response = self
            .hf_client
            .http_client()
            .post(&url)
            .headers(self.hf_client.auth_headers())
            .json(&body)
            .send()
            .await?;
        let response = self
            .hf_client
            .check_response(response, Some(&self.repo_path()), crate::error::NotFoundContext::Repo)
            .await?;
        Ok(response.json().await?)
    }
}

sync_api! {
    impl HFSpace -> HFSpaceSync {
        fn runtime(&self) -> Result<SpaceRuntime>;
        fn request_hardware(&self, params: &SpaceHardwareRequestParams) -> Result<SpaceRuntime>;
        fn set_sleep_time(&self, params: &SpaceSleepTimeParams) -> Result<()>;
        fn pause(&self) -> Result<SpaceRuntime>;
        fn restart(&self) -> Result<SpaceRuntime>;
        fn add_secret(&self, params: &SpaceSecretParams) -> Result<()>;
        fn delete_secret(&self, params: &SpaceSecretDeleteParams) -> Result<()>;
        fn add_variable(&self, params: &SpaceVariableParams) -> Result<()>;
        fn delete_variable(&self, params: &SpaceVariableDeleteParams) -> Result<()>;
        fn duplicate(&self, params: &DuplicateSpaceParams) -> Result<RepoUrl>;
    }
}
