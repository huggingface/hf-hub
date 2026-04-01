//! General path utilities.
//!
//! This module provides helper functions for common operations like path
//! construction and character encoding / normalization.
//!
//! These utilities are used internally by the library but are also exposed
//! for use by other crates that need similar functionality.

use std::env;
use std::path::{Path, PathBuf};

use crate::{constants, RepoType};

/// Returns the default user cache directory path.
///
/// This follows platform conventions for cache storage:
/// - On Unix-like systems: `~/.cache`
/// - On Windows: `%USERPROFILE%\.cache`
/// - On macOS: `~/.cache` (not following macOS conventions for consistency)
pub(crate) fn default_cache_dir() -> Option<PathBuf> {
    let mut path = dirs::home_dir()?;
    path.push(constants::CACHE_DIR);
    Some(path)
}

/// Returns the default top-level Hugging Face directory within the cache.
///
/// The path follows the pattern: `{cache_dir}/huggingface`
pub(crate) fn default_hf_home() -> Option<PathBuf> {
    let mut path = default_cache_dir()?;
    path.push(constants::TOP_LEVEL_HF_DIR);
    Some(path)
}

/// Returns the default Hugging Face Hub directory for repository storage.
pub(crate) fn default_hub_dir() -> Option<PathBuf> {
    let mut path = default_hf_home()?;
    path.push(constants::HUB_DIR);
    Some(path)
}

/// Returns the resolved Hugging Face home directory using Python-compatible precedence.
pub(crate) fn get_hf_home() -> Option<PathBuf> {
    if let Some(path) = env_path(constants::HF_HOME) {
        return Some(path);
    }

    let mut path = if let Some(path) = env_path(constants::XDG_CACHE_HOME) {
        path
    } else {
        default_cache_dir()?
    };
    path.push(constants::TOP_LEVEL_HF_DIR);
    Some(path)
}

/// Returns the Hugging Face Hub directory for repository storage.
///
/// This follows the Python library precedence:
/// `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `HF_HOME`, then
/// `XDG_CACHE_HOME`, then the default home-directory cache location.
///
/// # Examples
///
/// ```rust
/// use hf_hub::paths::get_hub_dir;
///
/// let hub_dir = get_hub_dir().unwrap();
/// assert!(hub_dir.ends_with(std::path::Path::new("huggingface/hub")));
/// ```
pub fn get_hub_dir() -> Option<PathBuf> {
    if let Some(path) = env_path(constants::HF_HUB_CACHE) {
        return Some(path);
    }
    if let Some(path) = env_path(constants::HUGGINGFACE_HUB_CACHE) {
        return Some(path);
    }
    let mut path = get_hf_home()?;
    path.push(constants::HUB_DIR);
    Some(path)
}

#[cfg(feature = "cache-manager")]
/// Checks if a directory entry corresponds to the `.locks` directory.
pub(crate) fn is_locks_dir(entry: &walkdir::DirEntry) -> bool {
    entry.file_name() == constants::LOCKS_DIR
}

/// Convert path separators in a string to flattened separators.
///
/// # Examples
///
/// ```rust
/// use hf_hub::paths::flatten_separator;
///
/// let flattened = flatten_separator("HuggingFaceTB/SmolLM2-135M");
/// assert_eq!(flattened, "HuggingFaceTB--SmolLM2-135M");
///
/// let no_change = flatten_separator("simple-name");
/// assert_eq!(no_change, "simple-name");
/// ```
pub fn flatten_separator(input: &str) -> String {
    input.replace(constants::REPO_ID_SEPARATOR, constants::FLAT_SEPARATOR)
}

/// Encodes path separators in a string.
///
/// # Examples
///
/// ```rust
/// use hf_hub::paths::encode_separator;
///
/// let encoded = encode_separator("HuggingFaceTB/SmolLM2-135M");
/// assert_eq!(encoded, "HuggingFaceTB%2FSmolLM2-135M");
///
/// let no_change = encode_separator("simple-name");
/// assert_eq!(no_change, "simple-name");
/// ```
pub fn encode_separator(input: &str) -> String {
    input.replace(constants::REPO_ID_SEPARATOR, constants::ENCODED_SEPARATOR)
}

/// Builds a repo folder name: "models--user--repo"
pub fn flattened_repo_folder_name(repo_type: &RepoType, repo_id: &str) -> String {
    let prefix = repo_type.plural();
    let prefixed_repo_id = format!("{prefix}{}{repo_id}", constants::FLAT_SEPARATOR);
    flatten_separator(&prefixed_repo_id)
}

/// Returns the path to the snapshots directory for a given repo root
pub fn snapshots_dir(repo_path: &Path) -> PathBuf {
    repo_path.join(constants::SNAPSHOTS_DIR)
}

/// Returns the path to the refs directory
pub fn refs_dir(repo_path: &Path) -> PathBuf {
    repo_path.join(constants::REFS_DIR)
}

/// Returns the path to the blobs directory
pub fn blobs_dir(repo_path: &Path) -> PathBuf {
    repo_path.join(constants::BLOBS_DIR)
}

/// Returns the location of the token file
///
/// This honors `HF_TOKEN_PATH` before falling back to `<hub-parent>/token`.
pub fn token_path(hub_path: &Path) -> PathBuf {
    if let Some(path) = env_path(constants::HF_TOKEN_PATH) {
        return path;
    }

    hub_path
        .parent()
        .expect("hub path should have a parent directory")
        .join(constants::TOKEN_FILE)
}

/// Returns the path to a specific ref
pub fn get_ref_path(repo_path: &Path, repo_ref: &str) -> PathBuf {
    refs_dir(repo_path).join(repo_ref)
}

fn env_path(name: &str) -> Option<PathBuf> {
    let value = env::var(name).ok()?;
    expand_path(&value)
}

fn expand_path(value: &str) -> Option<PathBuf> {
    let value = expand_tilde(value)?;
    Some(PathBuf::from(expand_vars(&value)))
}

fn expand_tilde(value: &str) -> Option<String> {
    if value == "~" {
        return Some(dirs::home_dir()?.to_string_lossy().into_owned());
    }

    if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("~\\"))
    {
        let mut home = dirs::home_dir()?;
        home.push(rest);
        return Some(home.to_string_lossy().into_owned());
    }

    Some(value.to_string())
}

fn expand_vars(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut output = String::with_capacity(value.len());
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '$' => {
                if index + 1 < chars.len() && chars[index + 1] == '{' {
                    if let Some(end) = chars[index + 2..].iter().position(|&ch| ch == '}') {
                        let name: String = chars[index + 2..index + 2 + end].iter().collect();
                        if let Ok(expanded) = env::var(&name) {
                            output.push_str(&expanded);
                        }
                        index += end + 3;
                        continue;
                    }
                }

                let mut end = index + 1;
                while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }

                if end > index + 1 {
                    let name: String = chars[index + 1..end].iter().collect();
                    if let Ok(expanded) = env::var(&name) {
                        output.push_str(&expanded);
                    }
                    index = end;
                    continue;
                }
            }
            '%' => {
                if let Some(end) = chars[index + 1..].iter().position(|&ch| ch == '%') {
                    let name: String = chars[index + 1..index + 1 + end].iter().collect();
                    if let Ok(expanded) = env::var(&name) {
                        output.push_str(&expanded);
                        index += end + 2;
                        continue;
                    }
                }
            }
            _ => {}
        }

        output.push(chars[index]);
        index += 1;
    }

    output
}
