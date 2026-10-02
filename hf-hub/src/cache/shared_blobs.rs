//! Read and clean up the shared-blob store written by Python `huggingface_hub`
//! (`utils/_shared_blobs.py`).
//!
//! Xet payloads are stored once at `<cache>/blobs/<hash[:2]>/<hash>`. A repo's
//! `blobs/<etag>` is a relative symlink into the store, and `<hash>.refs` lists every
//! `<repo_folder>/blobs/<etag>` linked to the payload, one per line. The manifest is only a
//! hint: a line counts only while that path is still a symlink resolving to the payload.

use std::fs::File;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const STORE_DIR: &str = "blobs";
const MARKER: &str = ".huggingface-shared-blobs";
const MARKER_CONTENT: &[u8] = b"1\n";

/// Canonical path of the store directory, if `cache_dir` holds a marked store.
pub(crate) fn store_dir(cache_dir: &Path) -> Option<PathBuf> {
    let store = cache_dir.join(STORE_DIR);
    let marker = store.join(MARKER);
    let is_marker_file = std::fs::symlink_metadata(&marker).is_ok_and(|m| m.is_file());
    if !is_marker_file || std::fs::read(&marker).ok()?.as_slice() != MARKER_CONTENT {
        return None;
    }
    std::fs::canonicalize(store).ok()
}

fn is_hash(name: &str) -> bool {
    name.len() == 64 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The store entry `repo_blob` (a repo's `blobs/<etag>`) links to, if it is a symlink into
/// the store at `store` with a valid `<prefix>/<hash>` layout.
pub(crate) fn store_target(store: &Path, repo_blob: &Path) -> Option<PathBuf> {
    if !std::fs::symlink_metadata(repo_blob).ok()?.file_type().is_symlink() {
        return None;
    }
    let target = std::fs::canonicalize(repo_blob).ok()?;
    let hash = target.file_name()?.to_str()?;
    let prefix_dir = target.parent()?;
    let prefix = prefix_dir.file_name()?.to_str()?;
    (is_hash(hash) && hash.starts_with(prefix) && prefix.len() == 2 && prefix_dir.parent() == Some(store))
        .then_some(target)
}

/// For a snapshot pointer, the repo-level `blobs/<etag>` it links to, if that is a symlink
/// into the store. The returned path has a canonical parent and an uncanonicalized file
/// name, so it names the link itself rather than the payload.
pub(crate) fn repo_link_for_pointer(store: &Path, pointer: &Path) -> Option<(PathBuf, PathBuf)> {
    let link_target = std::fs::read_link(pointer).ok()?;
    let resolved = pointer.parent()?.join(link_target);
    let repo_link = std::fs::canonicalize(resolved.parent()?).ok()?.join(resolved.file_name()?);
    let entry = store_target(store, &repo_link)?;
    Some((repo_link, entry))
}

fn manifest_path(entry: &Path) -> PathBuf {
    entry.with_extension("refs")
}

fn lock_path(entry: &Path) -> PathBuf {
    entry.with_extension("lock")
}

/// Mirrors Python's `_is_valid_ref`: a relative `<type>s--<name>/blobs/<etag>` path with no
/// `..`, which is currently a symlink resolving to `entry`.
fn is_valid_ref(cache_dir: &Path, store: &Path, entry: &Path, line: &str) -> bool {
    let rel = Path::new(line);
    let parts: Vec<&str> = match rel
        .components()
        .map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect()
    {
        Some(parts) => parts,
        None => return false,
    };
    let [repo_folder, "blobs", _etag] = parts.as_slice() else {
        return false;
    };
    let has_type_prefix = ["models--", "datasets--", "spaces--", "kernels--"]
        .iter()
        .any(|p| repo_folder.starts_with(p));
    has_type_prefix && store_target(store, &cache_dir.join(rel)).as_deref() == Some(entry)
}

/// Deletes the store entry if no valid reference to it remains, returning the bytes
/// freed; otherwise rewrites its manifest with only the valid lines. Runs under the entry's
/// `<hash>.lock`, shared with Python. A missing or unreadable manifest keeps the entry, as
/// in Python, since its references can't be proven gone.
pub(crate) fn sweep(cache_dir: &Path, store: &Path, entry: &Path) -> std::io::Result<u64> {
    let lock = File::create(lock_path(entry))?;
    lock.lock()?;

    let manifest = manifest_path(entry);
    let Ok(content) = std::fs::read_to_string(&manifest) else {
        return Ok(0);
    };
    let valid: Vec<&str> = content
        .lines()
        .filter(|line| is_valid_ref(cache_dir, store, entry, line))
        .collect();

    if valid.is_empty() {
        let size = std::fs::symlink_metadata(entry).map(|m| m.len()).unwrap_or(0);
        super::delete::try_delete(entry, "shared blob")?;
        super::delete::try_delete(&manifest, "shared blob manifest")?;
        return Ok(size);
    }

    let rewritten: String = valid.iter().map(|line| format!("{line}\n")).collect();
    if rewritten != content {
        let tmp = manifest.with_extension(format!("refs.{}.tmp", std::process::id()));
        let mut file = File::create(&tmp)?;
        file.write_all(rewritten.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&tmp, &manifest)?;
    }
    Ok(0)
}

/// Total size of every payload in the store, referenced or not.
pub(crate) fn payload_total(store: &Path) -> u64 {
    let Ok(prefixes) = std::fs::read_dir(store) else {
        return 0;
    };
    prefixes
        .flatten()
        .filter(|p| p.file_type().is_ok_and(|ft| ft.is_dir()))
        .filter_map(|p| std::fs::read_dir(p.path()).ok())
        .flat_map(|entries| entries.flatten())
        .filter(|e| e.file_name().to_str().is_some_and(is_hash))
        .filter_map(|e| e.metadata().ok().filter(|m| m.is_file()))
        .map(|m| m.len())
        .sum()
}
