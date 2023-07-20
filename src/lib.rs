#![deny(missing_docs)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
use std::io::Write;
use std::num::ParseIntError;
use std::path::PathBuf;

/// The actual Api to interact with the hub.
pub mod api;

/// hf-hub's error type
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O Error
    #[error("I/O error {0}")]
    Io(#[from] std::io::Error),

    /// Error parsing some range value
    #[error("Cannot parse int")]
    ParseInt(#[from] ParseIntError),

    /// We tried to download chunk too many times
    #[error("Too many retries: {0}")]
    TooManyRetries(Box<Error>),

    /// The header exists, but the value is not conform to what the Api expects.
    #[cfg(feature = "sync")]
    #[error("Header {0} is invalid")]
    InvalidHeader(api::sync::HeaderName),

    /// Api expects certain header to be present in the results to derive some information
    #[cfg(feature = "sync")]
    #[error("Header {0} is missing")]
    MissingHeader(api::sync::HeaderName),

    /// Error in the request
    #[cfg(feature = "sync")]
    #[error("request error: {0}")]
    Request(#[from] ureq::Error),

    /// Semaphore cannot be acquired
    #[cfg(feature = "tokio")]
    #[error("Acquire: {0}")]
    Acquire(#[from] tokio::sync::AcquireError),

    /// The header exists, but the value is not conform to what the Api expects.
    #[cfg(feature = "tokio")]
    #[error("Header {0} is invalid")]
    InvalidHeader(reqwest::header::HeaderName),

    /// The value cannot be used as a header during request header construction
    #[cfg(feature = "tokio")]
    #[error("Invalid header value {0}")]
    InvalidHeaderValue(#[from] reqwest::header::InvalidHeaderValue),

    /// Api expects certain header to be present in the results to derive some information
    #[cfg(feature = "tokio")]
    #[error("Header {0} is missing")]
    MissingHeader(reqwest::header::HeaderName),

    /// Error in the request
    #[cfg(feature = "tokio")]
    #[error("request error: {0}")]
    Request(#[from] reqwest::Error),

    /// The header value is not valid utf-8
    #[cfg(feature = "tokio")]
    #[error("header value is not a string")]
    ToStr(#[from] reqwest::header::ToStrError),

    /// Semaphore cannot be acquired
    #[cfg(feature = "tokio")]
    #[error("Try acquire: {0}")]
    TryAcquire(#[from] tokio::sync::TryAcquireError),
}

/// The type of repo to interact with
#[derive(Debug, Clone, Copy)]
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
pub struct Cache {
    path: PathBuf,
}

impl Cache {
    /// Creates a new cache object location
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Creates a new cache object location
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// This will get the location of the file within the cache for the remote
    /// `filename`. Will return `None` if file is not already present in cache.
    pub fn get(&self, repo: &Repo, filename: &str) -> Option<PathBuf> {
        let mut commit_path = self.path.clone();
        commit_path.push(repo.folder_name());
        commit_path.push("refs");
        commit_path.push(repo.revision());
        let commit_hash = std::fs::read_to_string(commit_path).ok()?;
        let mut pointer_path = self.pointer_path(repo, &commit_hash);
        pointer_path.push(filename);
        if pointer_path.exists() {
            Some(pointer_path)
        } else {
            None
        }
    }

    /// Creates a reference in the cache directory that points branches to the correct
    /// commits within the blobs.
    pub fn create_ref(&self, repo: &Repo, commit_hash: &str) -> Result<(), std::io::Error> {
        let mut ref_path = self.path.clone();
        ref_path.push(repo.folder_name());
        ref_path.push("refs");
        ref_path.push(repo.revision());
        // Needs to be done like this because revision might contain `/` creating subfolders here.
        std::fs::create_dir_all(ref_path.parent().unwrap())?;
        let mut file1 = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(&ref_path)?;
        file1.write_all(commit_hash.trim().as_bytes())?;
        Ok(())
    }

    #[cfg(any(feature = "sync", feature = "tokio"))]
    pub(crate) fn blob_path(&self, repo: &Repo, etag: &str) -> PathBuf {
        let mut blob_path = self.path.clone();
        blob_path.push(repo.folder_name());
        blob_path.push("blobs");
        blob_path.push(etag);
        blob_path
    }

    pub(crate) fn pointer_path(&self, repo: &Repo, commit_hash: &str) -> PathBuf {
        let mut pointer_path = self.path.clone();
        pointer_path.push(repo.folder_name());
        pointer_path.push("snapshots");
        pointer_path.push(commit_hash);
        pointer_path
    }
}

impl Default for Cache {
    fn default() -> Self {
        let path = match std::env::var("HF_HOME") {
            Ok(home) => home.into(),
            Err(_) => {
                let mut cache = dirs::home_dir().expect("Cache directory cannot be found");
                cache.push(".cache");
                cache.push("huggingface");
                cache.push("hub");
                cache
            }
        };
        Self::new(path)
    }
}

/// The representation of a repo on the hub.
#[allow(dead_code)] // Repo type unused in offline mode
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

    /// The actual URL part of the repo
    #[cfg(any(feature = "sync", feature = "tokio"))]
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
    #[cfg(any(feature = "sync", feature = "tokio"))]
    pub fn url_revision(&self) -> String {
        self.revision.replace('/', "%2F")
    }

    /// Used to compute the repo's url part when accessing the metadata of the repo
    #[cfg(any(feature = "sync", feature = "tokio"))]
    pub fn api_url(&self) -> String {
        let prefix = match self.repo_type {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
            RepoType::Space => "spaces",
        };
        format!("{prefix}/{}/revision/{}", self.repo_id, self.url_revision())
    }
}
