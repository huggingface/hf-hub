use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::client::HFClient;
use crate::error::{HFError, HFResult};
use crate::types::{RepoInfo, RepoInfoParams, RepoType};

/// A handle for a single repository on the Hugging Face Hub.
///
/// `HFRepository` is created via [`HFClient::repo`], [`HFClient::model`], or
/// [`HFClient::dataset`] and binds together the client, owner, repo name, and repo type.
/// All repo-scoped API operations are methods on this type.
///
/// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::{HFClient, types::RepoType};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let repo = client.model("openai-community", "gpt2");
/// let info = repo.info(&Default::default()).await?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFRepository {
    pub(crate) hf_client: HFClient,
    owner: String,
    name: String,
    pub(crate) repo_type: RepoType,
}

/// Alias for [`HFRepository`].
pub type HFRepo = HFRepository;

/// A handle for a Space repository, providing Space-specific operations on top of [`HFRepository`].
///
/// `HFSpace` wraps an [`HFRepository`] fixed to [`RepoType::Space`] and exposes hardware,
/// secret, and variable management. It derefs to [`HFRepository`], so all general repo
/// methods (e.g. `exists`, `info`, `download_file`) are accessible directly.
///
/// Created via [`HFClient::space`] or [`TryFrom<HFRepository>`].
///
/// # Example
///
/// ```rust,no_run
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let space = client.space("huggingface", "diffusers-gallery");
/// // General repo methods are available via Deref:
/// let exists = space.exists().await?;
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFSpace {
    pub(crate) repo: Arc<HFRepository>,
}

impl fmt::Debug for HFRepository {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFRepository")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .field("repo_type", &self.repo_type)
            .finish()
    }
}

impl fmt::Debug for HFSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HFSpace").field("repo", &self.repo).finish()
    }
}

impl HFClient {
    /// Create an [`HFRepository`] handle for any repo type.
    pub fn repo(&self, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        HFRepository::new(self.clone(), repo_type, owner, name)
    }

    /// Create an [`HFRepository`] handle for a model repository.
    pub fn model(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        self.repo(RepoType::Model, owner, name)
    }

    /// Create an [`HFRepository`] handle for a dataset repository.
    pub fn dataset(&self, owner: impl Into<String>, name: impl Into<String>) -> HFRepository {
        self.repo(RepoType::Dataset, owner, name)
    }

    /// Create an [`HFSpace`] handle for a Space repository.
    pub fn space(&self, owner: impl Into<String>, name: impl Into<String>) -> HFSpace {
        HFSpace::new(self.clone(), owner, name)
    }
}

impl HFRepository {
    /// Construct a new repository handle. Prefer the factory methods on [`HFClient`] instead.
    pub fn new(client: HFClient, repo_type: RepoType, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
            repo_type,
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The repository owner (user or organization name).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The repository name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` identifier used in Hub API calls.
    ///
    /// If no owner is set, returns just the name (for repos using short-form IDs like `"gpt2"`).
    pub fn repo_path(&self) -> String {
        if self.owner.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.owner, self.name)
        }
    }

    /// The type of this repository (model, dataset, or space).
    pub fn repo_type(&self) -> RepoType {
        self.repo_type
    }

    /// Fetch repository metadata, returning the appropriate [`RepoInfo`] variant.
    pub async fn info(&self, params: &RepoInfoParams) -> HFResult<RepoInfo> {
        match self.repo_type {
            RepoType::Model => self
                .model_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Model),
            RepoType::Dataset => self
                .dataset_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Dataset),
            RepoType::Space => self
                .space_info(params.revision.clone(), params.expand.clone())
                .await
                .map(RepoInfo::Space),
            RepoType::Kernel => {
                Err(HFError::Other("Repository info is not implemented yet for kernel repositories".to_string()))
            },
        }
    }
}

impl HFSpace {
    /// Construct a new Space handle. Prefer [`HFClient::space`] in most cases.
    pub fn new(client: HFClient, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            repo: Arc::new(HFRepository::new(client, RepoType::Space, owner, name)),
        }
    }

    pub fn repo(&self) -> &HFRepository {
        &self.repo
    }
}

impl TryFrom<HFRepository> for HFSpace {
    type Error = HFError;

    fn try_from(repo: HFRepository) -> HFResult<Self> {
        if repo.repo_type() != RepoType::Space {
            return Err(HFError::InvalidRepoType {
                expected: RepoType::Space,
                actual: repo.repo_type(),
            });
        }
        Ok(Self { repo: Arc::new(repo) })
    }
}

impl From<HFSpace> for Arc<HFRepository> {
    fn from(space: HFSpace) -> Self {
        space.repo.clone()
    }
}

impl Deref for HFSpace {
    type Target = HFRepository;

    fn deref(&self) -> &Self::Target {
        &self.repo
    }
}

#[cfg(test)]
mod tests {
    use super::{HFRepository, HFSpace};
    use crate::types::RepoType;

    #[test]
    fn test_repo_path_and_accessors() {
        let client = crate::HFClient::builder().build().unwrap();
        let repo = HFRepository::new(client, RepoType::Model, "openai-community", "gpt2");

        assert_eq!(repo.owner(), "openai-community");
        assert_eq!(repo.name(), "gpt2");
        assert_eq!(repo.repo_path(), "openai-community/gpt2");
        assert_eq!(repo.repo_type(), RepoType::Model);
    }

    #[test]
    fn test_hfspace_constructor_and_deref() {
        let client = crate::HFClient::builder().build().unwrap();
        let space = HFSpace::new(client, "huggingface-projects", "diffusers-gallery");

        assert_eq!(space.repo_type(), RepoType::Space);
        assert_eq!(space.repo_path(), "huggingface-projects/diffusers-gallery");
    }

    #[test]
    fn test_hfspace_try_from_repo() {
        let client = crate::HFClient::builder().build().unwrap();
        let space_repo = HFRepository::new(client.clone(), RepoType::Space, "owner", "space");
        assert!(HFSpace::try_from(space_repo).is_ok());

        let model_repo = HFRepository::new(client, RepoType::Model, "owner", "model");
        let error = HFSpace::try_from(model_repo).unwrap_err();
        match error {
            crate::HFError::InvalidRepoType { expected, actual } => {
                assert_eq!(expected, RepoType::Space);
                assert_eq!(actual, RepoType::Model);
            },
            _ => panic!("expected invalid repo type error"),
        }
    }
}
