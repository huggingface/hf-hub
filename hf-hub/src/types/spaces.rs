use serde::Deserialize;

/// Runtime state of a Space: stage, hardware, storage, and replica info.
///
/// Returned by Space lifecycle methods such as
/// [`HFSpace::runtime`](crate::repository::HFSpace::runtime),
/// [`HFSpace::pause`](crate::repository::HFSpace::pause), and
/// [`HFSpace::restart`](crate::repository::HFSpace::restart). The `raw` field preserves the
/// full JSON payload for fields not modeled explicitly.
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

#[cfg(test)]
mod tests {
    use super::*;

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
