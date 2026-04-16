/// Default Hugging Face Hub endpoint
pub(crate) const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// Default revision (branch)
pub(crate) const DEFAULT_REVISION: &str = "main";

pub(crate) const HF_ENDPOINT: &str = "HF_ENDPOINT";
pub(crate) const HF_TOKEN: &str = "HF_TOKEN";
pub(crate) const HF_TOKEN_PATH: &str = "HF_TOKEN_PATH";
pub(crate) const HF_HOME: &str = "HF_HOME";
pub(crate) const HF_HUB_CACHE: &str = "HF_HUB_CACHE";
pub(crate) const HUGGINGFACE_HUB_CACHE: &str = "HUGGINGFACE_HUB_CACHE";
pub(crate) const XDG_CACHE_HOME: &str = "XDG_CACHE_HOME";
pub(crate) const HF_HUB_DISABLE_IMPLICIT_TOKEN: &str = "HF_HUB_DISABLE_IMPLICIT_TOKEN";
pub(crate) const HF_HUB_USER_AGENT_ORIGIN: &str = "HF_HUB_USER_AGENT_ORIGIN";

/// Token filename within HF_HOME
pub(crate) const TOKEN_FILENAME: &str = "token";

pub(crate) const HEADER_X_XET_HASH: &str = "x-xet-hash";

pub(crate) const CACHE_LOCK_TIMEOUT_SECS: u64 = 10;
pub(crate) const HEADER_X_REPO_COMMIT: &str = "x-repo-commit";
pub(crate) const HEADER_X_LINKED_ETAG: &str = "x-linked-etag";
pub(crate) const HEADER_X_LINKED_SIZE: &str = "x-linked-size";

/// URL prefixes for different repo types
/// Models have no prefix, datasets use "datasets/", spaces use "spaces/", kernels use "kernels/"
pub fn repo_type_url_prefix(repo_type: Option<crate::types::repo::RepoType>) -> &'static str {
    match repo_type {
        None | Some(crate::types::repo::RepoType::Model) => "",
        Some(crate::types::repo::RepoType::Dataset) => "datasets/",
        Some(crate::types::repo::RepoType::Space) => "spaces/",
        Some(crate::types::repo::RepoType::Kernel) => "kernels/",
    }
}

/// API path segment for repo types: "models", "datasets", "spaces", "kernels"
pub fn repo_type_api_segment(repo_type: Option<crate::types::repo::RepoType>) -> &'static str {
    match repo_type {
        None | Some(crate::types::repo::RepoType::Model) => "models",
        Some(crate::types::repo::RepoType::Dataset) => "datasets",
        Some(crate::types::repo::RepoType::Space) => "spaces",
        Some(crate::types::repo::RepoType::Kernel) => "kernels",
    }
}

pub(crate) fn dirs_or_home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
}

pub fn hf_home() -> std::path::PathBuf {
    if let Ok(path) = std::env::var(HF_HOME) {
        return std::path::PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var(XDG_CACHE_HOME) {
        return std::path::PathBuf::from(xdg).join("huggingface");
    }
    let home = dirs_or_home();
    std::path::PathBuf::from(format!("{home}/.cache/huggingface"))
}

pub fn resolve_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var(HF_HUB_CACHE) {
        return std::path::PathBuf::from(cache);
    }
    if let Ok(cache) = std::env::var(HUGGINGFACE_HUB_CACHE) {
        return std::path::PathBuf::from(cache);
    }
    hf_home().join("hub")
}

#[cfg(test)]
mod tests {
    use super::{repo_type_api_segment, repo_type_url_prefix};
    use crate::types::repo::RepoType;

    #[test]
    fn test_repo_type_url_prefix() {
        assert_eq!(repo_type_url_prefix(None), "");
        assert_eq!(repo_type_url_prefix(Some(RepoType::Model)), "");
        assert_eq!(repo_type_url_prefix(Some(RepoType::Dataset)), "datasets/");
        assert_eq!(repo_type_url_prefix(Some(RepoType::Space)), "spaces/");
        assert_eq!(repo_type_url_prefix(Some(RepoType::Kernel)), "kernels/");
    }

    #[test]
    fn test_repo_type_api_segment() {
        assert_eq!(repo_type_api_segment(None), "models");
        assert_eq!(repo_type_api_segment(Some(RepoType::Model)), "models");
        assert_eq!(repo_type_api_segment(Some(RepoType::Dataset)), "datasets");
        assert_eq!(repo_type_api_segment(Some(RepoType::Space)), "spaces");
        assert_eq!(repo_type_api_segment(Some(RepoType::Kernel)), "kernels");
    }
}
