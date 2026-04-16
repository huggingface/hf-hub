//! General path utilities.
//!
//! This module provides helper functions for common operations like path
//! construction and character encoding / normalization.
//!
//! These utilities are used internally by the library but are also exposed
//! for use by other crates that need similar functionality.

use std::path::{Path, PathBuf};

use crate::{constants, RepoType};

/// Returns the user's cache directory path.
///
/// This follows platform conventions for cache storage:
/// - On Unix-like systems: `~/.cache`
/// - On Windows: `%USERPROFILE%\.cache`
/// - On macOS: `~/.cache` (not following macOS conventions for consistency)
pub(crate) fn get_cache_dir() -> Option<PathBuf> {
    let mut path = dirs::home_dir()?;
    path.push(constants::CACHE_DIR);
    Some(path)
}

/// Returns the top-level Hugging Face directory within the cache.
///
/// The path follows the pattern: `{cache_dir}/huggingface`
pub(crate) fn get_top_level_hf_dir() -> Option<PathBuf> {
    let mut path = get_cache_dir()?;
    path.push(constants::TOP_LEVEL_HF_DIR);
    Some(path)
}

/// Returns the Hugging Face Hub directory for repository storage.
///
/// The path follows the pattern: `{cache_dir}/{top_level_hf_dir}/hub`
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
    let mut path = get_top_level_hf_dir()?;
    path.push(constants::HUB_DIR);
    Some(path)
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
/// # Panics
///
/// Panics if the provided dir doesn't have a parent
pub fn token_path(hub_path: &Path) -> PathBuf {
    hub_path
        .parent()
        .expect("hub path should have a parent directory")
        .join(constants::TOKEN_FILE)
}

/// Returns the path to a specific ref
pub fn get_ref_path(repo_path: &Path, repo_ref: &str) -> PathBuf {
    refs_dir(repo_path).join(repo_ref)
}
