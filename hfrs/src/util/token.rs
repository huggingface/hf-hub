use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredTokens {
    #[serde(default)]
    tokens: HashMap<String, StoredToken>,
    #[serde(default)]
    active: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredToken {
    token: String,
}

#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub name: String,
    pub token_masked: String,
    pub is_active: bool,
}

fn hf_home() -> PathBuf {
    hf_hub::hf_home()
}

fn token_file_path() -> PathBuf {
    if let Ok(path) = std::env::var("HF_TOKEN_PATH") {
        return PathBuf::from(path);
    }
    hf_home().join("token")
}

fn stored_tokens_path() -> PathBuf {
    hf_home().join("stored_tokens")
}

fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        return "*".repeat(token.len());
    }
    let prefix = &token[..4];
    let suffix = &token[token.len() - 4..];
    format!("{prefix}...{suffix}")
}

fn read_stored_tokens() -> StoredTokens {
    let path = stored_tokens_path();
    if !path.exists() {
        return StoredTokens {
            tokens: HashMap::new(),
            active: None,
        };
    }
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(StoredTokens {
            tokens: HashMap::new(),
            active: None,
        })
}

fn write_stored_tokens(stored: &StoredTokens) -> anyhow::Result<()> {
    let path = stored_tokens_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(stored)?;
    fs::write(&path, json)?;
    Ok(())
}

fn write_active_token_file(token: &str) -> anyhow::Result<()> {
    let path = token_file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, token)?;
    Ok(())
}

pub fn save_token(name: &str, token: &str) -> anyhow::Result<()> {
    let mut stored = read_stored_tokens();
    stored.tokens.insert(
        name.to_string(),
        StoredToken {
            token: token.to_string(),
        },
    );
    if stored.active.is_none() {
        stored.active = Some(name.to_string());
    }
    write_stored_tokens(&stored)?;
    if stored.active.as_deref() == Some(name) {
        write_active_token_file(token)?;
    }
    Ok(())
}

pub fn delete_token(name: &str) -> anyhow::Result<()> {
    let mut stored = read_stored_tokens();
    stored.tokens.remove(name);
    if stored.active.as_deref() == Some(name) {
        stored.active = stored.tokens.keys().next().cloned();
        match &stored.active {
            Some(active_name) => {
                if let Some(entry) = stored.tokens.get(active_name) {
                    write_active_token_file(&entry.token)?;
                }
            },
            None => {
                let path = token_file_path();
                if path.exists() {
                    fs::remove_file(&path)?;
                }
            },
        }
    }
    write_stored_tokens(&stored)?;
    Ok(())
}

pub fn switch_token(name: &str) -> anyhow::Result<()> {
    let mut stored = read_stored_tokens();
    if !stored.tokens.contains_key(name) {
        anyhow::bail!("Token '{}' not found. Use `hfrs auth list` to see stored tokens.", name);
    }
    stored.active = Some(name.to_string());
    let token = stored.tokens[name].token.clone();
    write_stored_tokens(&stored)?;
    write_active_token_file(&token)?;
    Ok(())
}

pub fn list_tokens() -> Vec<TokenEntry> {
    let stored = read_stored_tokens();
    let mut entries: Vec<TokenEntry> = stored
        .tokens
        .iter()
        .map(|(name, entry)| TokenEntry {
            name: name.clone(),
            token_masked: mask_token(&entry.token),
            is_active: stored.active.as_deref() == Some(name.as_str()),
        })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}
