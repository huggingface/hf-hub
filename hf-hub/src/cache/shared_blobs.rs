//! Read and clean up the shared-blob store written by Python `huggingface_hub`
//! (`utils/_shared_blobs.py`).
//!
//! Xet payloads are stored once at `<cache>/blobs/<hash[:2]>/<hash>`. A repo's
//! `blobs/<etag>` is a relative symlink into the store, and `<hash>.refs` lists every
//! `<repo_folder>/blobs/<etag>` linked to the payload, one per line. The manifest is only a
//! hint: a line counts only while that path is still a symlink resolving to the payload.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};

const STORE_DIR: &str = "blobs";
const MARKER: &str = ".huggingface-shared-blobs";
const MARKER_CONTENT: &[u8] = b"1\n";

/// Canonical path of the store directory, if `cache_dir` holds a marked store.
pub(crate) fn store_dir(cache_dir: &Path) -> Option<PathBuf> {
    let store = cache_dir.join(STORE_DIR);
    let marker = store.join(MARKER);
    if !is_regular_file(&marker) || std::fs::read(&marker).ok()?.as_slice() != MARKER_CONTENT {
        return None;
    }
    std::fs::canonicalize(store).ok()
}

fn is_regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}

fn is_hash(name: &str) -> bool {
    name.len() == 64 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The store entry `repo_blob` (a repo's `blobs/<etag>`) links to, if it is a symlink into
/// the store at `store` with a valid `<prefix>/<hash>` layout. A missing or dangling link
/// is `Ok(None)`; any other I/O error is returned, so a reference that can't be checked is
/// never mistaken for one that is gone.
pub(crate) fn store_target(store: &Path, repo_blob: &Path) -> std::io::Result<Option<PathBuf>> {
    let meta = match std::fs::symlink_metadata(repo_blob) {
        Ok(meta) => meta,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if !meta.file_type().is_symlink() {
        return Ok(None);
    }
    let target = match std::fs::canonicalize(repo_blob) {
        Ok(target) => target,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let is_entry = (|| {
        let hash = target.file_name()?.to_str()?;
        let prefix_dir = target.parent()?;
        let prefix = prefix_dir.file_name()?.to_str()?;
        Some(is_hash(hash) && prefix.len() == 2 && hash.starts_with(prefix) && prefix_dir.parent() == Some(store))
    })()
    .unwrap_or(false);
    Ok(is_entry.then_some(target))
}

/// For a snapshot pointer, the `blobs/<etag>` link in `repo_blobs_dir` it goes through, and
/// the store entry that link resolves to. The link's parent is canonical but its file name
/// isn't resolved, so it names the link itself rather than the payload. Pointers that go
/// anywhere other than `repo_blobs_dir` are ignored.
pub(crate) fn repo_link_for_pointer(store: &Path, repo_blobs_dir: &Path, pointer: &Path) -> Option<(PathBuf, PathBuf)> {
    let link_target = std::fs::read_link(pointer).ok()?;
    let resolved = pointer.parent()?.join(link_target);
    let link_dir = std::fs::canonicalize(resolved.parent()?).ok()?;
    if link_dir != repo_blobs_dir {
        return None;
    }
    let repo_link = link_dir.join(resolved.file_name()?);
    let entry = store_target(store, &repo_link).ok()??;
    Some((repo_link, entry))
}

fn manifest_path(entry: &Path) -> PathBuf {
    entry.with_extension("refs")
}

fn lock_path(entry: &Path) -> PathBuf {
    entry.with_extension("lock")
}

/// Opens `path` for writing without following a symlink and only if it is a regular file,
/// so a planted link can't redirect the write. Lock and manifest files are made
/// world-writable like Python's, so every user of a shared cache can lock and append.
fn open_regular_for_write(path: &Path, truncate: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(truncate);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW).mode(0o666);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(ErrorKind::InvalidInput, format!("{} is not a regular file", path.display())));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Fails when another user owns the file, which is fine: whoever created it already
        // set the mode.
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o666));
    }
    Ok(file)
}

/// Mirrors Python's `_is_valid_ref`: a relative `<type>s--<name>/blobs/<etag>` path with no
/// `..`, which is currently a symlink resolving to `entry`.
fn is_valid_ref(cache_dir: &Path, store: &Path, entry: &Path, line: &str) -> std::io::Result<bool> {
    let rel = Path::new(line);
    let parts: Option<Vec<&str>> = rel
        .components()
        .map(|c| match c {
            Component::Normal(s) => s.to_str(),
            _ => None,
        })
        .collect();
    let Some(parts) = parts else {
        return Ok(false);
    };
    let [repo_folder, "blobs", _etag] = parts.as_slice() else {
        return Ok(false);
    };
    let has_type_prefix = ["models--", "datasets--", "spaces--", "kernels--"]
        .iter()
        .any(|p| repo_folder.starts_with(p));
    if !has_type_prefix {
        return Ok(false);
    }
    Ok(store_target(store, &cache_dir.join(rel))?.as_deref() == Some(entry))
}

/// Deletes the store entry if no valid reference to it remains, returning the bytes freed;
/// otherwise rewrites its manifest with only the valid lines. Runs under the entry's
/// `<hash>.lock`, shared with Python. As in Python, the entry is kept and nothing is
/// reported freed when its references can't be checked: the manifest is missing or
/// unreadable, or any I/O error occurs while sweeping.
pub(crate) fn sweep(cache_dir: &Path, store: &Path, entry: &Path) -> u64 {
    match try_sweep(cache_dir, store, entry) {
        Ok(freed) => freed,
        Err(e) => {
            tracing::warn!(entry = %entry.display(), error = %e, "couldn't sweep shared blob");
            0
        },
    }
}

fn try_sweep(cache_dir: &Path, store: &Path, entry: &Path) -> std::io::Result<u64> {
    let lock = open_regular_for_write(&lock_path(entry), false)?;
    lock.lock()?;

    let manifest = manifest_path(entry);
    if !is_regular_file(&manifest) {
        return Ok(0);
    }
    let Ok(content) = std::fs::read_to_string(&manifest) else {
        return Ok(0);
    };
    let mut valid = Vec::new();
    for line in content.lines() {
        if is_valid_ref(cache_dir, store, entry, line)? {
            valid.push(line);
        }
    }

    if valid.is_empty() {
        if !is_regular_file(entry) {
            return Ok(0);
        }
        let size = std::fs::symlink_metadata(entry)?.len();
        super::delete::try_delete(entry, "shared blob")?;
        super::delete::try_delete(&manifest, "shared blob manifest")?;
        return Ok(size);
    }

    let rewritten: String = valid.iter().map(|line| format!("{line}\n")).collect();
    if rewritten != content {
        let tmp = manifest.with_extension(format!("refs.{}.tmp", std::process::id()));
        let mut file = open_regular_for_write(&tmp, true)?;
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
