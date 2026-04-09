#![deny(missing_docs)]
#![cfg_attr(feature="ureq", doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md")))]
#![cfg_attr(
    not(feature = "ureq"),
    doc = "Documentation is meant to be compiled with default features (at least ureq)"
)]
use std::io::Write;
use std::path::PathBuf;

#[cfg(feature = "cache-manager")]
use std::path::Path;

/// The actual Api to interact with the hub.
#[cfg(any(feature = "tokio", feature = "ureq"))]
pub mod api;
#[cfg(feature = "cache-manager")]
pub mod cache_manager;
pub mod paths;

/// Configuration constants
pub(crate) mod constants {
    /// HF home env var
    pub(crate) const HF_HOME: &str = "HF_HOME";
    /// HF endpoint env var
    pub(crate) const HF_ENDPOINT: &str = "HF_ENDPOINT";
    /// HF endpoint
    pub(crate) const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

    /// Path separator used in repository names and URLs.
    pub(crate) const REPO_ID_SEPARATOR: &str = "/";
    /// Flattened separator used to replace path separators in repo names.
    pub(crate) const FLAT_SEPARATOR: &str = "--";
    /// URL-encoded separator.
    pub(crate) const ENCODED_SEPARATOR: &str = "%2F";

    /// Default cache directory name relative to the user's home directory.
    pub(crate) const CACHE_DIR: &str = ".cache";
    /// Top-level Hugging Face directory name within the cache.
    pub(crate) const TOP_LEVEL_HF_DIR: &str = "huggingface";
    /// Hub-specific directory name for storing data.
    pub(crate) const HUB_DIR: &str = "hub";
    /// Directory name for storing refs.
    pub(crate) const REFS_DIR: &str = "refs";
    /// Directory name for storing snapshots.
    pub(crate) const SNAPSHOTS_DIR: &str = "snapshots";
    /// Directory name for storing blobs.
    pub(crate) const BLOBS_DIR: &str = "blobs";
    #[cfg(feature = "cache-manager")]
    /// Directory name for locks dir (to manage concurrent file access during downloads)
    pub(crate) const LOCKS_DIR: &str = ".locks";

    /// Filename for storing authentication token.
    pub(crate) const TOKEN_FILE: &str = "token";

    /// Default branch name to use when none is specified.
    pub(crate) const DEFAULT_BRANCH: &str = "main";
}

/// The type of repo to interact with
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub enum RepoType {
    /// This is a model, usually it consists of weight files and some configuration
    /// files
    Model,
    /// This is a dataset, usually contains data within parquet files
    Dataset,
    /// This is a space, usually a demo showcashing a given model or dataset
    Space,
}

impl RepoType {
    /// Returns the plural form used in API routes, URLs, and cache folder names.
    pub fn plural(&self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        }
    }
}

/// Display is for human-facing output (CLI logs, errors).
impl std::fmt::Display for RepoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepoType::Model => write!(f, "model"),
            RepoType::Dataset => write!(f, "dataset"),
            RepoType::Space => write!(f, "space"),
        }
    }
}

/// Allows parsing from CLI args or config strings.
impl std::str::FromStr for RepoType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "model" | "models" => Ok(RepoType::Model),
            "dataset" | "datasets" => Ok(RepoType::Dataset),
            "space" | "spaces" => Ok(RepoType::Space),
            _ => Err(format!("Invalid repo type: '{}'", s)),
        }
    }
}

/// A local struct used to fetch information from the cache folder.
#[derive(Clone, Debug)]
pub struct Cache {
    path: PathBuf,
}

impl Cache {
    /// Creates a new cache object location
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Creates a `Cache` using the default hub directory, returning `None` if the
    /// home directory cannot be determined.
    ///
    /// Prefer this over [`Cache::default`] in environments where the home
    /// directory may not be available.
    pub fn try_default() -> Option<Self> {
        paths::get_hub_dir().map(Self::new)
    }

    /// Creates cache from environment variable HF_HOME (if defined) otherwise
    /// defaults to [`home_dir`]/.cache/huggingface/
    ///
    /// # Panics
    ///
    /// Panics if `HF_HOME` is not set and the home directory cannot be
    /// determined (which is needed for the `Cache::default` call).
    /// Use [`Cache::try_from_env`] to avoid this.
    pub fn from_env() -> Self {
        match std::env::var(constants::HF_HOME) {
            Ok(home) => {
                let mut path: PathBuf = home.into();
                path.push(constants::HUB_DIR);
                Self::new(path)
            }
            Err(_) => Self::default(),
        }
    }

    /// Like [`Cache::from_env`] but returns an Option.
    ///
    /// Creates cache from environment variable HF_HOME (if defined) otherwise
    /// defaults to [`home_dir`]/.cache/huggingface/.
    pub fn try_from_env() -> Option<Self> {
        match std::env::var(constants::HF_HOME) {
            Ok(home) => {
                let mut path: PathBuf = home.into();
                path.push(constants::HUB_DIR);
                Some(Self::new(path))
            }
            Err(_) => Self::try_default(),
        }
    }

    #[cfg(feature = "cache-manager")]
    pub(crate) fn validate_cache_dir_path(
        cache_dir: &Path,
    ) -> Result<(), cache_manager::CorruptedCacheError> {
        // TODO: replace with is_dir call since that checks exists too?
        if !cache_dir.exists() {
            return Err(cache_manager::CorruptedCacheError::MissingCacheDir {
                path: cache_dir.to_path_buf(),
            });
        }

        if cache_dir.is_file() {
            return Err(cache_manager::CorruptedCacheError::CacheDirCantBeFile {
                path: cache_dir.to_path_buf(),
            });
        }

        Ok(())
    }

    /// Creates a new cache object location
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Returns the location of the token file
    pub fn token_path(&self) -> PathBuf {
        paths::token_path(&self.path)
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

impl Default for Cache {
    /// Default for Cache
    ///
    /// # Panics
    ///
    /// Panics if the call to [`paths::get_hub_dir`] returns `None`, likely if
    /// the user's home directory cannot be determined. This typically
    /// only happens in very restricted environments or when the HOME environment
    /// variable is not set.
    fn default() -> Self {
        let path = paths::get_hub_dir().expect(
            "Hub directory cannot be found, possibly because HOME directory cannot be found.",
        );
        Self::new(path)
    }
}

/// Shorthand for accessing things within a particular repo
#[derive(Debug)]
pub struct CacheRepo {
    cache: Cache,
    repo: Repo,
}

impl CacheRepo {
    fn new(cache: Cache, repo: Repo) -> Self {
        Self { cache, repo }
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
        let mut cache_repo_path = self.cache.path.clone();
        cache_repo_path.push(self.repo.folder_name());
        cache_repo_path
    }

    fn ref_path(&self) -> PathBuf {
        let mut ref_path = paths::refs_dir(&self.path());
        ref_path.push(self.repo.revision());
        ref_path
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
        let mut blob_path = paths::blobs_dir(&self.path());
        blob_path.push(etag);
        blob_path
    }

    /// Get the path of the snapshot with the given commit hash.
    ///
    /// This path contains symlink pointers to the files for this commit.
    pub fn pointer_path(&self, commit_hash: &str) -> PathBuf {
        let mut pointer_path = paths::snapshots_dir(&self.path());
        pointer_path.push(commit_hash);
        pointer_path
    }
}

/// The representation of a repo on the hub.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Repo {
    repo_id: String,
    repo_type: RepoType,
    revision: String,
}

impl Repo {
    /// Repo with the default branch ("main").
    pub fn new(repo_id: String, repo_type: RepoType) -> Self {
        Self::with_revision(repo_id, repo_type, constants::DEFAULT_BRANCH.to_string())
    }

    /// fully qualified Repo
    pub fn with_revision(repo_id: String, repo_type: RepoType, revision: String) -> Self {
        Self {
            repo_id,
            repo_type,
            revision,
        }
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

    /// The flattened folder name of the repo within the cache directory
    pub fn folder_name(&self) -> String {
        paths::flattened_repo_folder_name(&self.repo_type, &self.repo_id)
    }

    /// The revision
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The actual URL part of the repo
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn url(&self) -> String {
        match self.repo_type {
            RepoType::Model => self.repo_id.to_string(),
            _ => format!("{}/{}", self.repo_type.plural(), self.repo_id),
        }
    }

    /// Revision needs to be url escaped before being used in a URL
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn url_revision(&self) -> String {
        paths::encode_separator(&self.revision)
    }

    /// Used to compute the repo's url part when accessing the metadata of the repo
    #[cfg(any(feature = "tokio", feature = "ureq"))]
    pub fn api_url(&self) -> String {
        format!(
            "{}/{}/revision/{}",
            self.repo_type.plural(),
            self.repo_id,
            self.url_revision()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        if let Ok(hf_home) = std::env::var(constants::HF_HOME) {
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
        if let Ok(hf_home) = std::env::var(HF_HOME) {
            assert_eq!(token_path, format!("{hf_home}\\token"));
        } else {
            let n = "huggingface/token".len();
            assert_eq!(&token_path[token_path.len() - n..], "huggingface\\token");
        }
    }
}
