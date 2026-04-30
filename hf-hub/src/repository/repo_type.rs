//! Type-level markers for the four Hugging Face Hub repo kinds.
//!
//! The [`RepoType`] sealed trait and its four marker structs ([`RepoTypeModel`],
//! [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]) parameterise
//! [`HFRepository<T>`](super::HFRepository) and
//! [`HFRepositorySync<T>`](crate::HFRepositorySync) so the repo kind is encoded in the
//! type system. Methods that differ per repo kind (such as
//! [`info`](super::HFRepository::info)) dispatch at compile time.
//!
//! See [`RepoType`] for the full overview, including the table of strings each marker
//! returns and how to pass markers to client builders.
//!
//! All items here are re-exported flat at `hf_hub::repository::…` and at the crate root,
//! so the canonical paths are `hf_hub::RepoType` and `hf_hub::RepoTypeModel` etc.

mod sealed {
    pub trait Sealed {}
}

/// Type-level marker for a Hugging Face Hub repo kind (model, dataset, Space, or kernel).
///
/// `RepoType` is sealed — only the four marker structs in this module implement it:
/// [`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]. Each is
/// zero-sized and serves as the type parameter on [`HFRepository<T>`](super::HFRepository)
/// and [`HFRepositorySync<T>`](crate::HFRepositorySync), so the repo kind is encoded in the
/// type system and methods that differ per kind (such as
/// [`info`](super::HFRepository::info)) dispatch at compile time.
///
/// # Picking a marker
///
/// Two equivalent ways:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeDataset};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
///
/// // Typed shortcut — preferred when the kind is known at the call site:
/// let repo = client.dataset("rajpurkar", "squad");
///
/// // Explicit marker — useful when the kind comes from a generic context:
/// let repo = client.repository::<RepoTypeDataset>("rajpurkar", "squad");
/// # let _ = repo; Ok(()) }
/// ```
///
/// Methods like [`HFClient::create_repository`](crate::HFClient::create_repository) /
/// [`delete_repository`](crate::HFClient::delete_repository) /
/// [`move_repository`](crate::HFClient::move_repository) take the marker as a builder
/// value:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeSpace};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// # let client = HFClient::new()?;
/// client
///     .create_repository()
///     .repo_id("acme/my-space")
///     .repo_type(RepoTypeSpace)
///     .space_sdk("static")
///     .send()
///     .await?;
/// # Ok(()) }
/// ```
///
/// # Trait methods
///
/// Each method returns the same value regardless of the receiver — they're properties
/// of the type, not the instance. The receiver is `&self` so callers with a value
/// in hand can write `marker.singular()`. Inside generic code where only the type
/// is in scope (`impl<T: RepoType>`), materialise the marker on the fly with
/// `T::default().singular()` — markers are zero-sized so it's a no-op:
///
/// | trait method     | `RepoTypeModel` | `RepoTypeDataset` | `RepoTypeSpace` | `RepoTypeKernel` |
/// | ---------------- | --------------- | ----------------- | --------------- | ---------------- |
/// | [`singular`]     | `"model"`       | `"dataset"`       | `"space"`       | `"kernel"`       |
/// | [`plural`]       | `"models"`      | `"datasets"`      | `"spaces"`      | `"kernels"`      |
/// | [`url_prefix`]   | `""`            | `"datasets/"`     | `"spaces/"`     | `"kernels/"`     |
///
/// The markers also implement [`Display`](std::fmt::Display) (writing the singular
/// form), so `tracing` fields and `format!("{}", RepoTypeModel)` work directly.
///
/// [`singular`]: Self::singular
/// [`plural`]: Self::plural
/// [`url_prefix`]: Self::url_prefix
pub trait RepoType: sealed::Sealed + Default + Copy + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Lowercase singular form of the repo kind: `"model"`, `"dataset"`, `"space"`, or `"kernel"`.
    /// Used in logs, error messages, and the `"type"` field of repo create/delete/move bodies.
    fn singular(&self) -> &'static str;

    /// Lowercase plural form, also the API segment in `/api/{plural}/{repo_id}` URLs:
    /// `"models"`, `"datasets"`, `"spaces"`, or `"kernels"`.
    fn plural(&self) -> &'static str;

    /// Path prefix used in resolve URLs (`<endpoint>/{prefix}{repo_id}/resolve/...`):
    /// `""` for [`RepoTypeModel`] (the model namespace is unprefixed) and `"<plural>/"`
    /// for the other three kinds.
    fn url_prefix(&self) -> &'static str;
}

/// Marker for a model repository on the Hub.
///
/// Implements [`RepoType`] returning `"model"` / `"models"` / `""` for
/// [`singular`](RepoType::singular), [`plural`](RepoType::plural), and
/// [`url_prefix`](RepoType::url_prefix). API URLs for models are
/// `/api/models/{repo_id}`; resolve URLs are unprefixed (`<endpoint>/{repo_id}/...`).
///
/// Use [`HFClient::model`](crate::HFClient::model) for the typed shortcut, or pass this
/// marker explicitly:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeModel};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
/// let repo = client.repository::<RepoTypeModel>("openai-community", "gpt2");
///
/// client
///     .create_repository()
///     .repo_id("acme/my-model")
///     .repo_type(RepoTypeModel)
///     .send()
///     .await?;
/// # let _ = repo; Ok(()) }
/// ```
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeModel;

/// Marker for a dataset repository on the Hub.
///
/// Implements [`RepoType`] returning `"dataset"` / `"datasets"` / `"datasets/"` for
/// [`singular`](RepoType::singular), [`plural`](RepoType::plural), and
/// [`url_prefix`](RepoType::url_prefix). API URLs are `/api/datasets/{repo_id}`;
/// resolve URLs are `<endpoint>/datasets/{repo_id}/...`.
///
/// Use [`HFClient::dataset`](crate::HFClient::dataset) for the typed shortcut, or pass
/// this marker explicitly:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeDataset};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
/// let repo = client.repository::<RepoTypeDataset>("rajpurkar", "squad");
///
/// client
///     .create_repository()
///     .repo_id("acme/my-dataset")
///     .repo_type(RepoTypeDataset)
///     .send()
///     .await?;
/// # let _ = repo; Ok(()) }
/// ```
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeDataset;

/// Marker for a Space repository on the Hub.
///
/// Implements [`RepoType`] returning `"space"` / `"spaces"` / `"spaces/"` for
/// [`singular`](RepoType::singular), [`plural`](RepoType::plural), and
/// [`url_prefix`](RepoType::url_prefix). API URLs are `/api/spaces/{repo_id}`;
/// resolve URLs are `<endpoint>/spaces/{repo_id}/...`.
///
/// Selects this marker to access Space-only methods on
/// [`HFRepository`](super::HFRepository) such as
/// [`runtime`](super::HFRepository::runtime),
/// [`pause`](super::HFRepository::pause),
/// [`add_secret`](super::HFRepository::add_secret), and
/// [`duplicate`](super::HFRepository::duplicate). Use
/// [`HFClient::space`](crate::HFClient::space) for the typed shortcut, or pass this
/// marker explicitly:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeSpace};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
/// let space = client.repository::<RepoTypeSpace>("acme", "demo");
/// let runtime = space.runtime().send().await?;
///
/// // `space_sdk` is required when creating a Space repo.
/// client
///     .create_repository()
///     .repo_id("acme/another")
///     .repo_type(RepoTypeSpace)
///     .space_sdk("static")
///     .send()
///     .await?;
/// # let _ = runtime; Ok(()) }
/// ```
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeSpace;

/// Marker for a kernel repository on the Hub.
///
/// Implements [`RepoType`] returning `"kernel"` / `"kernels"` / `"kernels/"` for
/// [`singular`](RepoType::singular), [`plural`](RepoType::plural), and
/// [`url_prefix`](RepoType::url_prefix). API URLs are `/api/kernels/{repo_id}`;
/// resolve URLs are `<endpoint>/kernels/{repo_id}/...`.
///
/// Note that [`HFRepository::<RepoTypeKernel>::info`](super::HFRepository::info) hits
/// `/api/kernels/{repo_id}` which returns a slim shape (no `tags`, `cardData`, or
/// `siblings`). For full model-style metadata on a kernel repo, build a model handle
/// for the same id and call `info()` on that. Use
/// [`HFClient::kernel`](crate::HFClient::kernel) for the typed shortcut, or pass this
/// marker explicitly:
///
/// ```rust,no_run
/// use hf_hub::{HFClient, RepoTypeKernel};
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::new()?;
/// let repo = client.repository::<RepoTypeKernel>("kernels-community", "cutlass-mla");
/// # let _ = repo; Ok(()) }
/// ```
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeKernel;

impl sealed::Sealed for RepoTypeModel {}
impl sealed::Sealed for RepoTypeDataset {}
impl sealed::Sealed for RepoTypeSpace {}
impl sealed::Sealed for RepoTypeKernel {}

impl RepoType for RepoTypeModel {
    fn singular(&self) -> &'static str {
        "model"
    }
    fn plural(&self) -> &'static str {
        "models"
    }
    fn url_prefix(&self) -> &'static str {
        ""
    }
}

impl RepoType for RepoTypeDataset {
    fn singular(&self) -> &'static str {
        "dataset"
    }
    fn plural(&self) -> &'static str {
        "datasets"
    }
    fn url_prefix(&self) -> &'static str {
        "datasets/"
    }
}

impl RepoType for RepoTypeSpace {
    fn singular(&self) -> &'static str {
        "space"
    }
    fn plural(&self) -> &'static str {
        "spaces"
    }
    fn url_prefix(&self) -> &'static str {
        "spaces/"
    }
}

impl RepoType for RepoTypeKernel {
    fn singular(&self) -> &'static str {
        "kernel"
    }
    fn plural(&self) -> &'static str {
        "kernels"
    }
    fn url_prefix(&self) -> &'static str {
        "kernels/"
    }
}

impl std::fmt::Display for RepoTypeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}

impl std::fmt::Display for RepoTypeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.singular())
    }
}
