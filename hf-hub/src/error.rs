use thiserror::Error;

#[derive(Error, Debug)]
pub enum HFError {
    #[error("HTTP error: {status} {url}")]
    Http {
        status: reqwest::StatusCode,
        url: String,
        body: String,
    },

    #[error("Authentication required")]
    AuthRequired,

    #[error("Repository not found: {repo_id}")]
    RepoNotFound { repo_id: String },

    #[error("Revision not found: {revision} in {repo_id}")]
    RevisionNotFound { repo_id: String, revision: String },

    #[error("Entry not found: {path} in {repo_id}")]
    EntryNotFound { path: String, repo_id: String },

    #[error("Bucket not found: {bucket_id}")]
    BucketNotFound { bucket_id: String },

    #[error("Invalid repository type: expected {expected}, got {actual}")]
    InvalidRepoType {
        expected: crate::types::RepoType,
        actual: crate::types::RepoType,
    },

    #[error("Forbidden")]
    Forbidden,

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("File not found in local cache: {path}")]
    LocalEntryNotFound { path: String },

    #[error(
        "Cache is not enabled — set cache_enabled(true) on HFClientBuilder, or provide local_dir in download params"
    )]
    CacheNotEnabled,

    #[error("Cache lock timed out: {}", path.display())]
    CacheLockTimeout { path: std::path::PathBuf },

    #[error(transparent)]
    Request(#[from] reqwest::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Url(#[from] url::ParseError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error(transparent)]
    DiffParse(#[from] crate::diff::HFDiffParseError),

    #[error("{0}")]
    Other(String),
}

impl HFError {
    /// Returns true for errors that indicate transient network/server issues
    /// where falling back to a cached version is appropriate.
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            HFError::Request(e) => e.is_connect() || e.is_timeout(),
            HFError::Http { status, .. } => {
                matches!(status.as_u16(), 500 | 502 | 503 | 504)
            },
            _ => false,
        }
    }
}

pub type HFResult<T> = std::result::Result<T, HFError>;

/// Context for mapping HTTP 404 errors to specific HFError variants.
pub(crate) enum NotFoundContext {
    /// 404 means the repository does not exist
    Repo,
    /// 404 means the bucket does not exist
    Bucket,
    /// 404 means a file/path does not exist within the repo
    Entry { path: String },
    /// 404 means the revision does not exist
    Revision { revision: String },
    /// No special mapping — use generic Http error
    Generic,
}
