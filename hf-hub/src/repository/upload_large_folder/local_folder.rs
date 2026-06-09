//! Byte-compatible reimplementation of Python `huggingface_hub`'s local upload
//! cache (`.cache/huggingface/upload/<path>.metadata`), enabling resumable
//! large-folder uploads interoperable with the Python tool.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;

use crate::error::HFResult;

pub(crate) struct LocalUploadFilePaths {
    pub path_in_repo: String,
    pub file_path: PathBuf,
    pub lock_path: PathBuf,
    pub metadata_path: PathBuf,
}

/// Returns `<local_dir>/.cache/huggingface`, creating it and its standard
/// `CACHEDIR.TAG` + `.gitignore` ("*") marker files if missing.
pub(crate) fn ensure_huggingface_dir(local_dir: &Path) -> HFResult<PathBuf> {
    let hf_dir = local_dir.join(".cache").join("huggingface");
    std::fs::create_dir_all(&hf_dir)?;

    let cachedir_tag = hf_dir.join("CACHEDIR.TAG");
    if !cachedir_tag.exists() {
        std::fs::write(
            &cachedir_tag,
            "Signature: 8a477f597d28d172789f06886806bc55\n# This file is a cache directory tag created by huggingface_hub.\n",
        )?;
    }
    let gitignore = hf_dir.join(".gitignore");
    if !gitignore.exists() {
        std::fs::write(&gitignore, "*")?;
    }
    Ok(hf_dir)
}

/// Computes the file, metadata, and lock paths for `path_in_repo`, eagerly
/// creating the parent directories of both the target file and its metadata.
/// `path_in_repo` is always stored with `/` separators (repo convention); the
/// on-disk paths use the OS separator via `Path::join`. The metadata/lock
/// suffixes are APPENDED to the full filename (Python parity): `model.safetensors`
/// -> `model.safetensors.metadata` / `model.safetensors.lock`.
pub(crate) fn get_local_upload_paths(local_dir: &Path, path_in_repo: &str) -> HFResult<LocalUploadFilePaths> {
    let hf_dir = ensure_huggingface_dir(local_dir)?;

    let mut file_path = local_dir.to_path_buf();
    for segment in path_in_repo.split('/') {
        file_path.push(segment);
    }

    let mut base = hf_dir.join("upload");
    for segment in path_in_repo.split('/') {
        base.push(segment);
    }
    let metadata_path = {
        let mut s = base.clone().into_os_string();
        s.push(".metadata");
        PathBuf::from(s)
    };
    let lock_path = {
        let mut s = base.into_os_string();
        s.push(".lock");
        PathBuf::from(s)
    };

    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = metadata_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(LocalUploadFilePaths {
        path_in_repo: path_in_repo.to_string(),
        file_path,
        lock_path,
        metadata_path,
    })
}

pub(crate) struct LocalUploadFileMetadata {
    pub size: u64,
    pub timestamp: Option<f64>,
    pub should_ignore: Option<bool>,
    pub sha256: Option<String>,
    pub upload_mode: Option<String>,
    pub remote_oid: Option<String>,
    pub is_uploaded: bool,
    pub is_committed: bool,
}

impl LocalUploadFileMetadata {
    /// Fresh metadata that knows only the file size (used for new files and as
    /// the recovery fallback for missing/corrupt/stale metadata).
    pub(crate) fn new(size: u64) -> Self {
        Self {
            size,
            timestamp: None,
            should_ignore: None,
            sha256: None,
            upload_mode: None,
            remote_oid: None,
            is_uploaded: false,
            is_committed: false,
        }
    }

    /// Serialize as Python's exact 8-line format under an advisory lock, then
    /// update `self.timestamp` to the value written.
    pub(crate) fn save(&mut self, paths: &LocalUploadFilePaths) -> HFResult<()> {
        let _guard = MetadataLock::acquire(&paths.lock_path)?;
        let now = now_unix_secs();

        let mut out = String::new();
        out.push_str(&format!("{now}\n"));
        out.push_str(&format!("{}\n", self.size));
        out.push_str(&opt_bool_line(self.should_ignore));
        out.push_str(&opt_str_line(self.sha256.as_deref()));
        out.push_str(&opt_str_line(self.upload_mode.as_deref()));
        out.push_str(&opt_str_line(self.remote_oid.as_deref()));
        out.push_str(&format!("{}\n", bool_digit(self.is_uploaded)));
        out.push_str(&format!("{}\n", bool_digit(self.is_committed)));

        std::fs::write(&paths.metadata_path, out)?;
        self.timestamp = Some(now);
        Ok(())
    }
}

fn bool_digit(b: bool) -> u8 {
    if b { 1 } else { 0 }
}

fn opt_bool_line(b: Option<bool>) -> String {
    match b {
        Some(v) => format!("{}\n", bool_digit(v)),
        None => "\n".to_string(),
    }
}

fn opt_str_line(s: Option<&str>) -> String {
    match s {
        Some(v) => format!("{v}\n"),
        None => "\n".to_string(),
    }
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Advisory exclusive lock on `<path>.lock`, flock-based and compatible with
/// Python `huggingface_hub`'s `WeakFileLock` (also flock on Unix). Released on drop.
pub(crate) struct MetadataLock {
    file: std::fs::File,
}

impl MetadataLock {
    pub(crate) fn acquire(lock_path: &Path) -> HFResult<Self> {
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        file.lock_exclusive()?;
        Ok(Self { file })
    }
}

impl Drop for MetadataLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_layout_matches_python() {
        let dir = tempfile::tempdir().unwrap();
        let paths = get_local_upload_paths(dir.path(), "sub/dir/model.safetensors").unwrap();
        assert_eq!(paths.path_in_repo, "sub/dir/model.safetensors");
        assert_eq!(paths.file_path, dir.path().join("sub/dir/model.safetensors"));
        assert_eq!(
            paths.metadata_path,
            dir.path().join(".cache/huggingface/upload/sub/dir/model.safetensors.metadata")
        );
        assert_eq!(paths.lock_path, dir.path().join(".cache/huggingface/upload/sub/dir/model.safetensors.lock"));
        // Parent dirs created eagerly.
        assert!(paths.metadata_path.parent().unwrap().is_dir());
        assert!(paths.file_path.parent().unwrap().is_dir());
    }

    #[test]
    fn cache_dir_scaffolding_written() {
        let dir = tempfile::tempdir().unwrap();
        ensure_huggingface_dir(dir.path()).unwrap();
        let hf = dir.path().join(".cache/huggingface");
        assert!(hf.join("CACHEDIR.TAG").is_file());
        assert_eq!(std::fs::read_to_string(hf.join(".gitignore")).unwrap(), "*");
    }

    #[test]
    fn save_writes_eight_lines_python_format() {
        let dir = tempfile::tempdir().unwrap();
        let paths = get_local_upload_paths(dir.path(), "a.bin").unwrap();
        let mut meta = LocalUploadFileMetadata {
            size: 123,
            timestamp: None,
            should_ignore: Some(false),
            sha256: Some("deadbeef".to_string()),
            upload_mode: Some("lfs".to_string()),
            remote_oid: None,
            is_uploaded: true,
            is_committed: false,
        };
        meta.save(&paths).unwrap();

        let content = std::fs::read_to_string(&paths.metadata_path).unwrap();
        let lines: Vec<&str> = content.split('\n').collect();
        assert_eq!(lines.len(), 9); // 8 content lines + trailing empty from final '\n'
        assert!(lines[0].parse::<f64>().is_ok()); // timestamp
        assert_eq!(lines[1], "123"); // size
        assert_eq!(lines[2], "0"); // should_ignore = false
        assert_eq!(lines[3], "deadbeef"); // sha256
        assert_eq!(lines[4], "lfs"); // upload_mode
        assert_eq!(lines[5], ""); // remote_oid = None
        assert_eq!(lines[6], "1"); // is_uploaded
        assert_eq!(lines[7], "0"); // is_committed
        assert_eq!(lines[8], ""); // trailing
        assert!(meta.timestamp.is_some());
    }

    #[test]
    fn save_blank_lines_for_none_optionals() {
        let dir = tempfile::tempdir().unwrap();
        let paths = get_local_upload_paths(dir.path(), "b.bin").unwrap();
        let mut meta = LocalUploadFileMetadata::new(7);
        meta.save(&paths).unwrap();
        let lines: Vec<String> = std::fs::read_to_string(&paths.metadata_path)
            .unwrap()
            .split('\n')
            .map(str::to_string)
            .collect();
        assert_eq!(lines[1], "7");
        assert_eq!(lines[2], ""); // should_ignore None
        assert_eq!(lines[3], ""); // sha256 None
        assert_eq!(lines[4], ""); // upload_mode None
        assert_eq!(lines[5], ""); // remote_oid None
        assert_eq!(lines[6], "0"); // is_uploaded false
        assert_eq!(lines[7], "0"); // is_committed false
    }
}
