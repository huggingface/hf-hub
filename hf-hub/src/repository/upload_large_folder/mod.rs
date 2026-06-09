//! `HFRepository::upload_large_folder`: resumable, xet-optimized upload of a
//! large local folder as a sequence of adaptively-batched commits.

pub mod local_folder;
pub mod pipeline;

use std::path::{Path, PathBuf};

use crate::error::{HFError, HFResult};
use crate::repository::files::matches_any_glob;

const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    ".git",
    ".git/**",
    "**/.git/**",
    ".cache/huggingface",
    ".cache/huggingface/**",
    "**/.cache/huggingface/**",
];

/// Recursively collects `(repo_path, absolute_file_path)` for every file under
/// `folder` that survives the default ignores, the user allow/ignore globs, and
/// is then prefixed by `path_in_repo`. Repo paths use `/` separators.
fn discover_files(
    folder: &Path,
    path_in_repo: Option<&str>,
    allow_patterns: &Option<Vec<String>>,
    ignore_patterns: &Option<Vec<String>>,
) -> HFResult<Vec<(String, PathBuf)>> {
    let mut default_and_user_ignores: Vec<String> = DEFAULT_IGNORE_PATTERNS.iter().map(|s| s.to_string()).collect();
    if let Some(user) = ignore_patterns {
        default_and_user_ignores.extend(user.iter().cloned());
    }

    let prefix = path_in_repo.map(|p| p.trim_matches('/').to_string()).filter(|p| !p.is_empty());

    let mut out = Vec::new();
    collect(folder, folder, allow_patterns, &default_and_user_ignores, prefix.as_deref(), &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn collect(
    root: &Path,
    current: &Path,
    allow: &Option<Vec<String>>,
    ignore: &[String],
    prefix: Option<&str>,
    out: &mut Vec<(String, PathBuf)>,
) -> HFResult<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect(root, &path, allow, ignore, prefix, out)?;
        } else if meta.is_file() {
            let relative = path.strip_prefix(root).map_err(|e| {
                HFError::InvalidParameter(format!("path {} not under {}: {e}", path.display(), root.display()))
            })?;
            let rel: String = relative
                .components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");

            if let Some(allow) = allow
                && !matches_any_glob(allow, &rel)
            {
                continue;
            }
            if matches_any_glob(ignore, &rel) {
                continue;
            }

            let repo_path = match prefix {
                Some(p) => format!("{p}/{rel}"),
                None => rel,
            };
            out.push((repo_path, path));
        }
    }
    Ok(())
}

#[cfg(test)]
mod discovery_tests {
    use super::*;

    fn write(root: &std::path::Path, rel: &str, bytes: &[u8]) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn discovers_files_applies_default_and_user_ignores_and_prefix() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.txt", b"1");
        write(dir.path(), "sub/b.bin", b"22");
        write(dir.path(), ".git/config", b"x");
        write(dir.path(), ".cache/huggingface/upload/a.txt.metadata", b"meta");
        write(dir.path(), "skip.me", b"z");

        let found = discover_files(dir.path(), Some("models"), &None, &Some(vec!["*.me".to_string()])).unwrap();

        let repo_paths: std::collections::BTreeSet<String> =
            found.iter().map(|(repo_path, _)| repo_path.clone()).collect();
        assert_eq!(
            repo_paths,
            ["models/a.txt".to_string(), "models/sub/b.bin".to_string()]
                .into_iter()
                .collect()
        );
    }
}
