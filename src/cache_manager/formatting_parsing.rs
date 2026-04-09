//! Formatting and parsing utils
//!
//! Based on <https://github.com/huggingface/huggingface_hub/blob/02d0608b1ae51d24841f8f92e23879e08f57ddb8/src/huggingface_hub/utils/_parsing.py>

use std::time::SystemTime;

use thiserror::Error;

use crate::{constants, RepoType};

// TODO: use timeago and humansize crates?

// (label, divider, max_value)
pub const TIMESINCE_CHUNKS: [(&str, u64, Option<u64>); 7] = [
    ("second", 1, Some(60)),
    ("minute", 60, Some(60)),
    ("hour", 60 * 60, Some(24)),
    ("day", 60 * 60 * 24, Some(6)),
    ("week", 60 * 60 * 24 * 7, Some(6)),
    ("month", 60 * 60 * 24 * 30, Some(11)),
    ("year", 60 * 60 * 24 * 365, None),
];

pub const SIZE_UNITS: [&str; 8] = ["", "K", "M", "G", "T", "P", "E", "Z"];

#[derive(Debug, Error)]
pub enum RepoFolderParseError {
    #[error("Folder name does not contain the required separator")]
    MissingSeparator,
    #[error("Invalid repo type: {0}")]
    InvalidType(String),
}

/// Tries to parse a folder name into a RepoType and Repo ID.
pub fn parse_repo_folder_name(name: &str) -> Result<(RepoType, String), RepoFolderParseError> {
    let (repo_type_str, repo_id_str) = name
        .split_once(constants::FLAT_SEPARATOR)
        .ok_or(RepoFolderParseError::MissingSeparator)?;

    let repo_type = repo_type_str
        .parse::<RepoType>()
        .map_err(|_| RepoFolderParseError::InvalidType(repo_type_str.to_string()))?;

    let repo_id = repo_id_str.replace(constants::FLAT_SEPARATOR, constants::REPO_ID_SEPARATOR);

    Ok((repo_type, repo_id))
}

/// Format timestamp into a human-readable string, relative to now.
///
/// Vaguely inspired by Django's `timesince` formatter.
///
/// Adapted to take a `SystemTime`.
pub fn format_timesince(ts: SystemTime) -> String {
    // Get difference in seconds (defaulting to 0 if ts is in the future)
    let delta = SystemTime::now()
        .duration_since(ts)
        .unwrap_or_default()
        .as_secs();

    if delta < 20 {
        return "a few seconds ago".to_string();
    }

    for (label, divider, max_value) in TIMESINCE_CHUNKS {
        // Integer math for rounding: (delta + divider / 2) / divider
        let value = (delta + divider / 2) / divider;

        let within_bounds = match max_value {
            Some(max) => value <= max,
            None => true,
        };

        if within_bounds {
            let plural = if value > 1 { "s" } else { "" };
            return format!("{} {}{} ago", value, label, plural);
        }
    }

    // Unreachable due to the final chunk having `None` as max_value
    String::new()
}

/// Format size in bytes into a human-readable string.
///
/// Taken from https://stackoverflow.com/a/1094933
pub fn format_size(num: u64) -> String {
    let mut num_f = num as f64;

    for unit in SIZE_UNITS {
        if num_f < 1000.0 {
            return format!("{num_f:.1}{unit}");
        }

        num_f /= 1000.0;
    }

    format!("{num_f:.1}Y")
}
