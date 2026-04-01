#![deny(missing_docs)]
#![cfg_attr(feature="ureq", doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md")))]
#![cfg_attr(
    not(feature = "ureq"),
    doc = "Documentation is meant to be compiled with default features (at least ureq)"
)]
use std::io::Write;
use std::path::{Path, PathBuf};

/// The actual Api to interact with the hub.
#[cfg(any(feature = "tokio", feature = "ureq"))]
pub mod api;

const HF_HOME: &str = "HF_HOME";
const HF_HUB_CACHE: &str = "HF_HUB_CACHE";

/// The type of repo to interact with
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RepoType {
    /// This is a model, usually it consists of weight files and some configuration
    /// files
    Model,
    /// This is a dataset, usually contains data within parquet files
    Dataset,
    /// This is a space, usually a demo showcashing a given model or dataset
    Space,
}

/// A local struct used to fetch information from the cache folder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cache {
    path: PathBuf,
}

/// A named cached ref and the commit hash it resolves to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheRef {
    name: String,
    commit_hash: String,
}

impl CacheRef {
    /// The ref name, relative to the repo's `refs/` directory.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The commit hash this ref points to.
    pub fn commit_hash(&self) -> &str {
        &self.commit_hash
    }
}

/// Parsed metadata for a path inside a cached snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachePath {
    repo: Repo,
    commit_hash: String,
    relative_path: String,
}

impl CachePath {
    /// The repo this path belongs to.
    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    /// The snapshot commit hash.
    pub fn commit_hash(&self) -> &str {
        &self.commit_hash
    }

    /// The path relative to the snapshot root.
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }
}

impl Cache {
    /// Creates a new cache object location
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Creates cache from environment.
    ///
    /// This honors `HF_HUB_CACHE` first, then `HF_HOME`, and otherwise
    /// defaults to [`home_dir`]/.cache/huggingface/hub.
    pub fn from_env() -> Self {
        if let Ok(path) = std::env::var(HF_HUB_CACHE) {
            let path = path.trim();
            if !path.is_empty() {
                return Self::new(path.into());
            }
        }
        match std::env::var(HF_HOME) {
            Ok(home) => {
                let mut path: PathBuf = home.into();
                path.push("hub");
                Self::new(path)
            }
            Err(_) => Self::default(),
        }
    }

    /// Creates a new cache object location
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Enumerates cached repos present under this cache root.
    pub fn repos(&self) -> Result<Vec<CacheRepo>, std::io::Error> {
        let mut repos = Vec::new();
        if !self.path.exists() {
            return Ok(repos);
        }

        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(repo) = Repo::from_folder_name(name) else {
                continue;
            };
            repos.push(self.repo(repo));
        }

        repos.sort_by(|left, right| left.repo.folder_name().cmp(&right.repo.folder_name()));
        Ok(repos)
    }

    /// Parses a path inside this cache root and returns its repo and snapshot metadata.
    pub fn path_info<P: AsRef<Path>>(&self, path: P) -> Option<CachePath> {
        let relative = path.as_ref().strip_prefix(&self.path).ok()?;
        let mut components = relative.components();
        let repo_dir = components.next()?.as_os_str().to_str()?;
        let repo = Repo::from_folder_name(repo_dir)?;
        if components.next()?.as_os_str() != "snapshots" {
            return None;
        }
        let commit_hash = components.next()?.as_os_str().to_str()?.to_string();
        let relative_path = components
            .map(|component| component.as_os_str().to_str())
            .collect::<Option<Vec<_>>>()?
            .join("/");
        Some(CachePath {
            repo,
            commit_hash,
            relative_path,
        })
    }

    /// Returns the location of the token file
    pub fn token_path(&self) -> PathBuf {
        let mut path = self.path.clone();
        // Remove `"hub"`
        path.pop();
        path.push("token");
        path
    }

    /// Returns the token value if it exists in the cache
    /// Use `huggingface-cli login` to set it up.
    pub fn token(&self) -> Option<String> {
        let token_filename = self.token_path();
        if token_filename.exists() {
            log::info!("Using token file found {token_filename:?}");
        }
        match std::fs::read_to_string(token_filename) {
            Ok(token_content) => {
                let token_content = token_content.trim();
                if token_content.is_empty() {
                    None
                } else {
                    Some(token_content.to_string())
                }
            }
            Err(_) => None,
        }
    }

    /// Creates a new handle [`CacheRepo`] which contains operations
    /// on a particular [`Repo`]
    pub fn repo(&self, repo: Repo) -> CacheRepo {
        CacheRepo::new(self.clone(), repo)
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Model));
    /// ```
    pub fn model(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Model))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Dataset));
    /// ```
    pub fn dataset(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Dataset))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{Cache, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let cache = Cache::new("/tmp/".into());
    /// let cache = cache.repo(Repo::new(model_id, RepoType::Space));
    /// ```
    pub fn space(&self, model_id: String) -> CacheRepo {
        self.repo(Repo::new(model_id, RepoType::Space))
    }
}

/// Shorthand for accessing things within a particular repo
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CacheRepo {
    cache: Cache,
    repo: Repo,
}

impl CacheRepo {
    fn new(cache: Cache, repo: Repo) -> Self {
        Self { cache, repo }
    }

    /// The repo handle for this cached repo.
    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    /// This will get the location of the file within the cache for the remote
    /// `filename`. Will return `None` if file is not already present in cache.
    pub fn get(&self, filename: &str) -> Option<PathBuf> {
        let commit_path = self.ref_path();
        let commit_hash = std::fs::read_to_string(commit_path).ok()?;
        let mut pointer_path = self.pointer_path(&commit_hash);
        pointer_path.push(filename);
        if pointer_path.exists() {
            Some(pointer_path)
        } else {
            None
        }
    }

    fn path(&self) -> PathBuf {
        let mut ref_path = self.cache.path.clone();
        ref_path.push(self.repo.folder_name());
        ref_path
    }

    fn ref_path(&self) -> PathBuf {
        let mut ref_path = self.path();
        ref_path.push("refs");
        ref_path.push(self.repo.revision());
        ref_path
    }

    /// Lists all cached refs for this repo.
    pub fn refs(&self) -> Result<Vec<CacheRef>, std::io::Error> {
        let refs_dir = self.path().join("refs");
        let mut refs = Vec::new();
        if !refs_dir.is_dir() {
            return Ok(refs);
        }
        collect_ref_files(&refs_dir, &refs_dir, &mut refs)?;
        refs.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(refs)
    }

    /// Creates a reference in the cache directory that points branches to the correct
    /// commits within the blobs.
    pub fn create_ref(&self, commit_hash: &str) -> Result<(), std::io::Error> {
        let ref_path = self.ref_path();
        // Needs to be done like this because revision might contain `/` creating subfolders here.
        std::fs::create_dir_all(ref_path.parent().unwrap())?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&ref_path)?;
        file.write_all(commit_hash.trim().as_bytes())?;
        Ok(())
    }

    /// Get the path of the blob with the given etag.
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn blob_path(&self, etag: &str) -> PathBuf {
        let mut blob_path = self.path();
        blob_path.push("blobs");
        blob_path.push(etag);
        blob_path
    }

    /// Get the path of the snapshot with the given commit hash.
    ///
    /// This path contains symlink pointers to the files for this commit.
    pub fn pointer_path(&self, commit_hash: &str) -> PathBuf {
        let mut pointer_path = self.path();
        pointer_path.push("snapshots");
        pointer_path.push(commit_hash);
        pointer_path
    }

    /// Lists files relative to the snapshot root for a cached commit.
    pub fn files(&self, commit_hash: &str) -> Result<Vec<String>, std::io::Error> {
        let root = self.pointer_path(commit_hash);
        let mut files = Vec::new();
        if !root.is_dir() {
            return Ok(files);
        }
        collect_snapshot_files(&root, &root, &mut files)?;
        files.sort();
        Ok(files)
    }
}

impl Default for Cache {
    fn default() -> Self {
        let mut path = dirs::home_dir().expect("Cache directory cannot be found");
        path.push(".cache");
        path.push("huggingface");
        path.push("hub");
        Self::new(path)
    }
}

/// The representation of a repo on the hub.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Repo {
    repo_id: String,
    repo_type: RepoType,
    revision: String,
}

impl Repo {
    /// Repo with the default branch ("main").
    pub fn new(repo_id: String, repo_type: RepoType) -> Self {
        Self::with_revision(repo_id, repo_type, "main".to_string())
    }

    /// fully qualified Repo
    pub fn with_revision(repo_id: String, repo_type: RepoType, revision: String) -> Self {
        Self {
            repo_id,
            repo_type,
            revision,
        }
    }

    fn from_folder_name(name: &str) -> Option<Self> {
        let (repo_type, repo_id) = if let Some(repo_id) = name.strip_prefix("models--") {
            (RepoType::Model, repo_id)
        } else if let Some(repo_id) = name.strip_prefix("datasets--") {
            (RepoType::Dataset, repo_id)
        } else if let Some(repo_id) = name.strip_prefix("spaces--") {
            (RepoType::Space, repo_id)
        } else {
            return None;
        };
        Some(Self::new(repo_id.replace("--", "/"), repo_type))
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Model`]
    pub fn model(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Model)
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Dataset`]
    pub fn dataset(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Dataset)
    }

    /// Shortcut for [`Repo::new`] with [`RepoType::Space`]
    pub fn space(repo_id: String) -> Self {
        Self::new(repo_id, RepoType::Space)
    }

    /// The normalized folder nameof the repo within the cache directory
    pub fn folder_name(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}--{}", self.repo_id).replace('/', "--")
    }

    /// The revision
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The repo id.
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// The repo type.
    pub fn repo_type(&self) -> RepoType {
        self.repo_type
    }

    /// The actual URL part of the repo
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn url(&self) -> String {
        match self.repo_type {
            RepoType::Model => self.repo_id.to_string(),
            RepoType::Dataset => {
                format!("datasets/{}", self.repo_id)
            }
            RepoType::Space => {
                format!("spaces/{}", self.repo_id)
            }
        }
    }

    /// Revision needs to be url escaped before being used in a URL
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn url_revision(&self) -> String {
        self.revision.replace('/', "%2F")
    }

    /// Used to compute the repo's url part when accessing the metadata of the repo
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn api_url(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}/{}/revision/{}", self.repo_id, self.url_revision())
    }
}

fn collect_ref_files(
    root: &Path,
    dir: &Path,
    refs: &mut Vec<CacheRef>,
) -> Result<(), std::io::Error> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_ref_files(root, &path, refs)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let Some(commit_hash) = read_trimmed_file(&path)? else {
            continue;
        };
        refs.push(CacheRef { name, commit_hash });
    }
    Ok(())
}

fn collect_snapshot_files(
    root: &Path,
    dir: &Path,
    files: &mut Vec<String>,
) -> Result<(), std::io::Error> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_snapshot_files(root, &path, files)?;
            continue;
        }
        if !file_type.is_file() && !file_type.is_symlink() {
            continue;
        }
        let file = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push(file);
    }
    Ok(())
}

fn read_trimmed_file(path: &Path) -> Result<Option<String>, std::io::Error> {
    let value = std::fs::read_to_string(path)?.trim().to_string();
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let mut path = std::env::temp_dir();
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            path.push(format!("hf-hub-tests-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Internal macro used to show cleaners errors
    /// on the payloads received from the hub.
    #[macro_export]
    macro_rules! assert_no_diff {
        ($left: expr, $right: expr) => {
            let left = serde_json::to_string_pretty(&$left).unwrap();
            let right = serde_json::to_string_pretty(&$right).unwrap();
            if left != right {
                use rand::Rng;
                use std::io::Write;
                use std::process::Command;
                let rand_string: String = rand::rng()
                    .sample_iter(&rand::distr::Alphanumeric)
                    .take(6)
                    .map(char::from)
                    .collect();
                let left_filename = format!("/tmp/left-{rand_string}.txt");
                let mut file = std::fs::File::create(&left_filename).unwrap();
                file.write_all(left.as_bytes()).unwrap();
                let right_filename = format!("/tmp/right-{rand_string}.txt");
                let mut file = std::fs::File::create(&right_filename).unwrap();
                file.write_all(right.as_bytes()).unwrap();
                let output = Command::new("diff")
                    // Reverse order seems to be more appropriate for how we set up the tests.
                    .args(["-U5", &right_filename, &left_filename])
                    .output()
                    .expect("Failed to diff")
                    .stdout;
                let diff = String::from_utf8(output).expect("Invalid utf-8 diff output");
                // eprintln!("assertion `left == right` failed\n{diff}");
                assert!(false, "{diff}")
            };
        };
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn token_path() {
        let cache = Cache::from_env();
        let token_path = cache.token_path().to_str().unwrap().to_string();
        if let Ok(hf_hub_cache) = std::env::var(HF_HUB_CACHE) {
            let mut expected = PathBuf::from(hf_hub_cache);
            expected.pop();
            expected.push("token");
            assert_eq!(token_path, expected.to_string_lossy());
        } else if let Ok(hf_home) = std::env::var(HF_HOME) {
            assert_eq!(token_path, format!("{hf_home}/token"));
        } else {
            let n = "huggingface/token".len();
            assert_eq!(&token_path[token_path.len() - n..], "huggingface/token");
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn token_path() {
        let cache = Cache::from_env();
        let token_path = cache.token_path().to_str().unwrap().to_string();
        if let Ok(hf_hub_cache) = std::env::var(HF_HUB_CACHE) {
            let mut expected = PathBuf::from(hf_hub_cache);
            expected.pop();
            expected.push("token");
            assert_eq!(token_path, expected.to_string_lossy());
        } else if let Ok(hf_home) = std::env::var(HF_HOME) {
            assert_eq!(token_path, format!("{hf_home}\\token"));
        } else {
            let n = "huggingface/token".len();
            assert_eq!(&token_path[token_path.len() - n..], "huggingface\\token");
        }
    }

    #[test]
    fn from_env_prefers_hf_hub_cache() {
        let tmp = TestDir::new();
        let prev_hub_cache = std::env::var_os(HF_HUB_CACHE);
        let prev_hf_home = std::env::var_os(HF_HOME);
        std::env::set_var(HF_HUB_CACHE, tmp.path());
        std::env::set_var(HF_HOME, "/tmp/ignored-hf-home");

        let cache = Cache::from_env();
        assert_eq!(cache.path(), &tmp.path().to_path_buf());

        restore_env(HF_HUB_CACHE, prev_hub_cache);
        restore_env(HF_HOME, prev_hf_home);
    }

    #[test]
    fn repos_lists_cached_repos() {
        let tmp = TestDir::new();
        std::fs::create_dir_all(tmp.path().join("models--gpt2")).unwrap();
        std::fs::create_dir_all(tmp.path().join("datasets--org--corpus")).unwrap();
        std::fs::create_dir_all(tmp.path().join("spaces--org--demo")).unwrap();
        std::fs::create_dir_all(tmp.path().join("not-a-repo")).unwrap();

        let cache = Cache::new(tmp.path().to_path_buf());
        let repos = cache.repos().unwrap();
        let names: Vec<_> = repos
            .iter()
            .map(|repo| {
                (
                    repo.repo().repo_type(),
                    repo.repo().repo_id().to_string(),
                    repo.repo().revision().to_string(),
                )
            })
            .collect();

        assert_eq!(
            names,
            vec![
                (
                    RepoType::Dataset,
                    "org/corpus".to_string(),
                    "main".to_string()
                ),
                (RepoType::Model, "gpt2".to_string(), "main".to_string()),
                (RepoType::Space, "org/demo".to_string(), "main".to_string()),
            ]
        );
    }

    #[test]
    fn refs_list_nested_ref_names() {
        let tmp = TestDir::new();
        let cache = Cache::new(tmp.path().to_path_buf());
        let repo = cache.repo(Repo::model("org/model".to_string()));
        let main = repo.path().join("refs").join("main");
        std::fs::create_dir_all(main.parent().unwrap()).unwrap();
        std::fs::write(&main, "commit-main\n").unwrap();
        let pr = repo.path().join("refs").join("refs").join("pr").join("1");
        std::fs::create_dir_all(pr.parent().unwrap()).unwrap();
        std::fs::write(&pr, "commit-pr\n").unwrap();

        let refs = repo.refs().unwrap();
        let refs: Vec<_> = refs
            .into_iter()
            .map(|cache_ref| {
                (
                    cache_ref.name().to_string(),
                    cache_ref.commit_hash().to_string(),
                )
            })
            .collect();

        assert_eq!(
            refs,
            vec![
                ("main".to_string(), "commit-main".to_string()),
                ("refs/pr/1".to_string(), "commit-pr".to_string()),
            ]
        );
    }

    #[test]
    fn files_list_snapshot_contents() {
        let tmp = TestDir::new();
        let cache = Cache::new(tmp.path().to_path_buf());
        let repo = cache.repo(Repo::model("org/model".to_string()));
        let root = repo.pointer_path("commit-hash");
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
        std::fs::write(root.join("nested").join("model.gguf"), "x").unwrap();

        let files = repo.files("commit-hash").unwrap();
        assert_eq!(
            files,
            vec!["config.json".to_string(), "nested/model.gguf".to_string()]
        );
    }

    #[test]
    fn path_info_parses_snapshot_paths() {
        let tmp = TestDir::new();
        let cache = Cache::new(tmp.path().to_path_buf());
        let path = tmp
            .path()
            .join("models--org--model")
            .join("snapshots")
            .join("abc123")
            .join("weights")
            .join("model.gguf");

        let info = cache.path_info(&path).unwrap();
        assert_eq!(info.repo().repo_type(), RepoType::Model);
        assert_eq!(info.repo().repo_id(), "org/model");
        assert_eq!(info.commit_hash(), "abc123");
        assert_eq!(info.relative_path(), "weights/model.gguf");
    }

    fn restore_env(key: &str, value: Option<OsString>) {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}
