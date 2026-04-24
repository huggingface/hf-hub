use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::repository::{HFRepository, RepoType};
use crate::spaces::HFSpace;

fn build_runtime() -> HFResult<Arc<tokio::runtime::Runtime>> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map(Arc::new)
        .map_err(|e| HFError::Other(format!("Failed to create tokio runtime: {e}")))
}

/// Synchronous/blocking counterpart to [`HFClient`].
///
/// Wraps an `HFClient` together with a dedicated single-threaded tokio runtime so
/// that every async API method can be called from synchronous code. The runtime is
/// shared with all repo/space handles derived from this client.
#[derive(Clone)]
pub struct HFClientSync {
    pub(crate) inner: HFClient,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

/// Synchronous/blocking counterpart to [`HFRepository`].
///
/// Holds a reference to the underlying async handle and the shared tokio runtime.
/// Derefs to [`HFRepository`], so all accessor methods (owner, name, repo_path,
/// etc.) are available directly. Blocking API methods are defined via the `sync_api!`
/// macro in the corresponding component modules.
#[derive(Clone)]
pub struct HFRepositorySync {
    pub(crate) inner: Arc<HFRepository>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

/// Synchronous/blocking counterpart to [`HFSpace`].
///
/// Derefs to [`HFRepositorySync`] so all blocking repository methods and accessors
/// are available directly. Space-specific blocking methods are defined via the
/// `sync_api!` macro.
#[derive(Clone)]
pub struct HFSpaceSync {
    repo_sync: Arc<HFRepositorySync>,
    pub(crate) inner: Arc<HFSpace>,
}

/// Synchronous/blocking counterpart to [`crate::buckets::HFBucket`].
///
/// Holds a reference to the underlying async handle and the shared tokio runtime.
/// Blocking API methods are defined via the `sync_api!` macro in `buckets/mod.rs`.
#[derive(Clone)]
pub struct HFBucketSync {
    pub(crate) inner: Arc<crate::buckets::HFBucket>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl fmt::Debug for HFClientSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFClientSync").finish()
    }
}

impl fmt::Debug for HFRepositorySync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFRepositorySync").field("inner", &self.inner).finish()
    }
}

impl fmt::Debug for HFSpaceSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFSpaceSync")
            .field("inner", &self.inner)
            .field("repo_sync", &self.repo_sync)
            .finish()
    }
}

impl fmt::Debug for HFBucketSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFBucketSync").field("inner", &self.inner).finish()
    }
}

impl HFClientSync {
    /// Creates an `HFClientSync` using the default configuration from the environment.
    ///
    /// Reads the `HF_TOKEN`, `HF_ENDPOINT`, and other the standard environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokio runtime cannot be created or if `HFClient::new` fails.
    pub fn new() -> HFResult<Self> {
        Ok(Self {
            inner: HFClient::new()?,
            runtime: build_runtime()?,
        })
    }

    /// Creates an `HFClientSync` wrapping an already-configured [`HFClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the tokio runtime cannot be created.
    pub fn from_inner(inner: HFClient) -> HFResult<Self> {
        Ok(Self {
            inner,
            runtime: build_runtime()?,
        })
    }

    /// Creates a blocking repository handle for the given repo type, owner, and name.
    pub fn repo(&self, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        HFRepositorySync::new(self.clone(), repo_type, owner, name)
    }

    /// Creates a blocking handle for a model repository.
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        self.repo(RepoType::Model, owner, name)
    }

    /// Creates a blocking handle for a dataset repository.
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        self.repo(RepoType::Dataset, owner, name)
    }

    /// Creates a blocking handle for a space repository.
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFSpaceSync {
        HFSpaceSync::new(self.clone(), owner, name)
    }

    /// Creates a blocking handle for a bucket.
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucketSync {
        HFBucketSync::new(self.clone(), owner, name)
    }
}

impl Deref for HFClientSync {
    type Target = HFClient;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl HFRepositorySync {
    /// Creates a blocking repository handle from a client, repo type, owner, and name.
    pub fn new(client: HFClientSync, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(HFRepository::new(client.inner.clone(), repo_type, owner, name)),
            runtime: client.runtime.clone(),
        }
    }
}

impl Deref for HFRepositorySync {
    type Target = HFRepository;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl HFSpaceSync {
    /// Creates a blocking space handle for the given owner and name.
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        let repo_sync = Arc::new(HFRepositorySync::new(client, RepoType::Space, owner, name));
        let inner = Arc::new(HFSpace {
            repo: repo_sync.inner.clone(),
        });
        Self { repo_sync, inner }
    }

    /// Converts this space handle into a plain [`HFRepositorySync`], discarding space-specific state.
    pub fn repo(&self) -> &HFRepositorySync {
        &self.repo_sync
    }
}

impl Deref for HFSpaceSync {
    type Target = HFRepositorySync;

    fn deref(&self) -> &Self::Target {
        &self.repo_sync
    }
}

impl HFBucketSync {
    /// Creates a blocking bucket handle from a client, owner, and name.
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(crate::buckets::HFBucket::new(client.inner.clone(), owner, name)),
            runtime: client.runtime.clone(),
        }
    }
}

impl Deref for HFBucketSync {
    type Target = crate::buckets::HFBucket;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TryFrom<HFRepositorySync> for HFSpaceSync {
    type Error = HFError;

    fn try_from(repo: HFRepositorySync) -> HFResult<Self> {
        if repo.inner.repo_type() != RepoType::Space {
            return Err(HFError::InvalidRepoType {
                expected: RepoType::Space,
                actual: repo.inner.repo_type(),
            });
        }
        let inner = Arc::new(HFSpace {
            repo: repo.inner.clone(),
        });
        Ok(Self {
            repo_sync: Arc::new(repo),
            inner,
        })
    }
}

impl From<HFSpaceSync> for Arc<HFRepositorySync> {
    fn from(space: HFSpaceSync) -> Self {
        space.repo_sync
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hfapisync_creation() {
        let client = HFClientSync::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_hfapisync_from_api() {
        let async_client = HFClient::builder().build().unwrap();
        let client = HFClientSync::from_inner(async_client);
        assert!(client.is_ok());
    }

    #[test]
    fn test_sync_repo_constructors() {
        let client = HFClientSync::from_inner(HFClient::builder().build().unwrap()).unwrap();
        let repo = client.model("openai-community", "gpt2");
        let space = client.space("huggingface", "transformers-benchmarks");

        assert_eq!(repo.owner(), "openai-community");
        assert_eq!(repo.name(), "gpt2");
        assert_eq!(repo.repo_type(), RepoType::Model);
        assert_eq!(space.repo_type(), RepoType::Space);
    }

    #[test]
    fn test_sync_space_try_from_repo() {
        let client = HFClientSync::from_inner(HFClient::builder().build().unwrap()).unwrap();
        let space_repo = client.repo(RepoType::Space, "owner", "space");
        assert!(HFSpaceSync::try_from(space_repo).is_ok());

        let model_repo = client.repo(RepoType::Model, "owner", "model");
        let error = HFSpaceSync::try_from(model_repo).unwrap_err();
        match error {
            HFError::InvalidRepoType { expected, actual } => {
                assert_eq!(expected, RepoType::Space);
                assert_eq!(actual, RepoType::Model);
            },
            _ => panic!("expected invalid repo type error"),
        }
    }
}
