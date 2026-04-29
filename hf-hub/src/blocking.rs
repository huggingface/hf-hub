use std::marker::PhantomData;
use std::sync::Arc;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::repository::{HFRepository, RepoType, RepoTypeDataset, RepoTypeKernel, RepoTypeModel, RepoTypeSpace};

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

/// Synchronous/blocking counterpart to [`HFRepository`], parameterised by the repo kind via `T`.
///
/// Wraps an [`HFRepository<T>`] and blocks on the corresponding async methods.
///
/// See [`HFRepository`] for method semantics.
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub struct HFRepositorySync<T: RepoType> {
    pub(crate) inner: Arc<HFRepository<T>>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
    _ty: PhantomData<fn() -> T>,
}

impl<T: RepoType> Clone for HFRepositorySync<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            runtime: Arc::clone(&self.runtime),
            _ty: PhantomData,
        }
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

impl<T: RepoType> std::fmt::Debug for HFRepositorySync<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFRepositorySync").field("inner", &self.inner).finish()
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

    /// Creates a blocking handle for any repo kind via a turbofished generic.
    ///
    /// See [`HFClient::repository`].
    pub fn repository<T: RepoType>(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<T> {
        HFRepositorySync::new(self.clone(), owner, name)
    }

    /// Creates a blocking handle for a model repository.
    ///
    /// See [`HFClient::model`].
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeModel> {
        self.repository::<RepoTypeModel>(owner, name)
    }

    /// Creates a blocking handle for a dataset repository.
    ///
    /// See [`HFClient::dataset`].
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeDataset> {
        self.repository::<RepoTypeDataset>(owner, name)
    }

    /// Creates a blocking handle for a Space repository.
    ///
    /// See [`HFClient::space`].
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeSpace> {
        self.repository::<RepoTypeSpace>(owner, name)
    }

    /// Creates a blocking handle for a kernel repository.
    ///
    /// See [`HFClient::kernel`].
    pub fn kernel(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepositorySync<RepoTypeKernel> {
        self.repository::<RepoTypeKernel>(owner, name)
    }

    /// Creates a blocking handle for a bucket.
    ///
    /// See [`HFClient::bucket`].
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucketSync {
        HFBucketSync::new(self.clone(), owner, name)
    }
}

impl<T: RepoType> HFRepositorySync<T> {
    /// Creates a blocking repository handle.
    ///
    /// See [`HFRepository::new`].
    pub fn new(client: HFClientSync, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(HFRepository::new(client.inner.clone(), owner, name)),
            runtime: client.runtime.clone(),
            _ty: PhantomData,
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

    /// Lowercase singular name of this repo's kind, equivalent to `T::default().singular()`.
    pub fn repo_type(&self) -> &'static str {
        T::default().singular()
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
        assert_eq!(repo.repo_type(), "model");
        assert_eq!(space.repo_type(), "space");
    }
}
