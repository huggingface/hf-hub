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
/// Wraps an [`HFClient`] together with a dedicated single-threaded tokio
/// runtime so the async API can be used from synchronous code.
///
/// Xet uploads and downloads do not run on this runtime: hf-xet requires a
/// multi-threaded runtime with the IO and time drivers enabled, so the
/// single-threaded runtime here does not meet its requirements. When a Xet
/// transfer is triggered through any blocking handle, hf-xet spins up its
/// own multi-threaded thread pool to back the `XetSession`, separate from
/// the runtime owned by `HFClientSync`.
///
/// See [`HFClient`] for configuration and API semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFClientSync {
    pub(crate) inner: HFClient,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

/// Synchronous/blocking counterpart to [`HFRepository`].
///
/// Wraps an [`HFRepository`] and blocks on the corresponding async methods.
/// Derefs to [`HFRepository`], so repo accessors are available directly.
///
/// See [`HFRepository`] for method semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFRepositorySync {
    pub(crate) inner: Arc<HFRepository>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

/// Synchronous/blocking counterpart to [`HFSpace`].
///
/// Derefs to [`HFRepositorySync`], so blocking repository methods are
/// available directly.
///
/// See [`HFSpace`] for Space-specific behavior.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFSpaceSync {
    pub(crate) repo_sync: Arc<HFRepositorySync>,
    pub(crate) inner: Arc<HFSpace>,
}

impl std::ops::Deref for HFSpaceSync {
    type Target = HFRepositorySync;

    fn deref(&self) -> &Self::Target {
        &self.repo_sync
    }
}

/// Synchronous/blocking counterpart to [`crate::buckets::HFBucket`].
///
/// Wraps an [`crate::buckets::HFBucket`] and blocks on the corresponding async
/// methods.
///
/// See [`crate::buckets::HFBucket`] for method semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
#[derive(Clone)]
pub struct HFBucketSync {
    pub(crate) inner: Arc<crate::buckets::HFBucket>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

impl std::fmt::Debug for HFClientSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFClientSync").finish()
    }
}

impl std::fmt::Debug for HFRepositorySync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFRepositorySync").field("inner", &self.inner).finish()
    }
}

impl std::fmt::Debug for HFSpaceSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFSpaceSync")
            .field("inner", &self.inner)
            .field("repo_sync", &self.repo_sync)
            .finish()
    }
}

impl std::fmt::Debug for HFBucketSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFBucketSync").field("inner", &self.inner).finish()
    }
}

impl HFClientSync {
    /// Creates an `HFClientSync` using the default configuration from the environment.
    ///
    /// Reads the standard environment variables used by [`HFClient::new`].
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

    /// Creates an `HFClientSync` from an existing [`HFClient`].
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

    /// Creates a blocking repository handle.
    ///
    /// See [`HFClient::repo`] for details.
    pub fn repo(&self, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        HFRepositorySync::new(self.clone(), repo_type, owner, name)
    }

    /// Creates a blocking handle for a model repository.
    ///
    /// See [`HFClient::model`].
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        self.repo(RepoType::Model, owner, name)
    }

    /// Creates a blocking handle for a dataset repository.
    ///
    /// See [`HFClient::dataset`].
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync {
        self.repo(RepoType::Dataset, owner, name)
    }

    /// Creates a blocking handle for a Space repository.
    ///
    /// See [`HFClient::space`].
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFSpaceSync {
        HFSpaceSync::new(self.clone(), owner, name)
    }

    /// Creates a blocking handle for a bucket.
    ///
    /// See [`HFClient::bucket`].
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucketSync {
        HFBucketSync::new(self.clone(), owner, name)
    }
}

impl HFRepositorySync {
    /// Creates a blocking repository handle.
    ///
    /// See [`HFRepository::new`].
    pub fn new(client: HFClientSync, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(HFRepository::new(client.inner.clone(), repo_type, owner, name)),
            runtime: client.runtime.clone(),
        }
    }

    /// The repository owner (user or organization name).
    pub fn owner(&self) -> &str {
        self.inner.owner()
    }

    /// The repository name (without the owner prefix).
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// The full `"owner/name"` identifier used in Hub API calls.
    ///
    /// If no owner is set, returns just the name (for repos using short-form IDs like `"gpt2"`).
    pub fn repo_path(&self) -> String {
        self.inner.repo_path()
    }

    /// The type of this repository (model, dataset, or space).
    pub fn repo_type(&self) -> RepoType {
        self.inner.repo_type()
    }
}

impl HFSpaceSync {
    /// Creates a blocking Space handle.
    ///
    /// See [`HFSpace::new`].
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        let repo_sync = Arc::new(HFRepositorySync::new(client, RepoType::Space, owner, name));
        let inner = Arc::new(HFSpace {
            repo: repo_sync.inner.clone(),
        });
        Self { repo_sync, inner }
    }

    /// Returns the underlying blocking repository handle.
    ///
    /// See [`HFSpace::repo`].
    pub fn repo(&self) -> &HFRepositorySync {
        &self.repo_sync
    }
}

impl HFBucketSync {
    /// Creates a blocking bucket handle.
    ///
    /// See [`crate::buckets::HFBucket::new`].
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(crate::buckets::HFBucket::new(client.inner.clone(), owner, name)),
            runtime: client.runtime.clone(),
        }
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
