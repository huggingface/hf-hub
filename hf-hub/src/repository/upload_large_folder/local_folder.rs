//! Byte-compatible reimplementation of Python `huggingface_hub`'s local upload
//! cache (`.cache/huggingface/upload/<path>.metadata`), enabling resumable
//! large-folder uploads interoperable with the Python tool.

use std::path::{Path, PathBuf};

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
}
