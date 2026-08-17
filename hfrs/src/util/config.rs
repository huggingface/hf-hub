use std::fs;
use std::path::PathBuf;

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

pub fn hf_home() -> PathBuf {
    if let Ok(path) = std::env::var("HF_HOME") {
        return PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".cache/huggingface")
}

pub fn resolve_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    hf_home().join("hub")
}

pub fn resolve_token(explicit_token: Option<String>) -> Option<String> {
    if explicit_token.is_some() {
        return explicit_token;
    }

    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|value| !value.is_empty()) {
        return None;
    }

    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        return Some(token);
    }

    if let Ok(path) = std::env::var("HF_TOKEN_PATH")
        && let Some(token) = read_token_file(PathBuf::from(path))
    {
        return Some(token);
    }

    read_token_file(hf_home().join("token"))
}

pub fn resolve_endpoint(cli_endpoint: Option<String>) -> String {
    cli_endpoint
        .or_else(|| std::env::var("HF_ENDPOINT").ok())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
}

pub fn resolve_user_agent() -> Option<String> {
    std::env::var("HF_HUB_USER_AGENT_ORIGIN")
        .ok()
        .map(|origin| format!("hf-hub/{}; {origin}", hf_hub::VERSION))
}

fn read_token_file(path: PathBuf) -> Option<String> {
    let token = fs::read_to_string(path).ok()?.trim().to_string();
    (!token.is_empty()).then_some(token)
}
