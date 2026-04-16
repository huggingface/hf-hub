use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoType {
    Model,
    Dataset,
    Space,
    Kernel,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobLfsInfo {
    pub size: Option<u64>,
    pub sha256: Option<String>,
    pub pointer_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastCommitInfo {
    pub id: Option<String>,
    pub title: Option<String>,
    pub date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoSibling {
    pub rfilename: String,
    pub size: Option<u64>,
    pub lfs: Option<BlobLfsInfo>,
}

/// Tagged union for tree entries returned by list_repo_tree
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RepoTreeEntry {
    File {
        oid: String,
        size: u64,
        path: String,
        lfs: Option<BlobLfsInfo>,
        #[serde(default, rename = "lastCommit")]
        last_commit: Option<LastCommitInfo>,
    },
    Directory {
        oid: String,
        path: String,
    },
}

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

#[derive(Debug, Clone)]
pub enum RepoInfo {
    Model(ModelInfo),
    Dataset(DatasetInfo),
    Space(SpaceInfo),
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

/// URL returned by create_repo/move_repo
#[derive(Debug, Clone, Deserialize)]
pub struct RepoUrl {
    pub url: String,
}

#[derive(Debug, Clone)]
pub enum GatedApprovalMode {
    Disabled,
    Auto,
    Manual,
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GatedNotificationsMode {
    Bulk,
    RealTime,
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

#[cfg(test)]
mod tests {
    use super::{RepoTreeEntry, RepoType};

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
    fn test_repo_tree_entry_deserialize_file() {
        let json = r#"{"type":"file","oid":"abc123","size":100,"path":"test.txt"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::File { path, size, .. } => {
                assert_eq!(path, "test.txt");
                assert_eq!(size, 100);
            },
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_repo_tree_entry_deserialize_directory() {
        let json = r#"{"type":"directory","oid":"def456","path":"src"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::Directory { path, .. } => {
                assert_eq!(path, "src");
            },
            _ => panic!("Expected Directory variant"),
        }
    }
}
