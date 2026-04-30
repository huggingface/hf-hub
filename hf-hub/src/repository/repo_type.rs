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

/// Model-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::model`](crate::HFClient::model) for the
/// typed shortcut.
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeModel;

/// Dataset-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::dataset`](crate::HFClient::dataset) for
/// the typed shortcut.
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeDataset;

/// Space-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::space`](crate::HFClient::space) for the
/// typed shortcut. This is the marker used to access Space-only methods on
/// [`HFRepository`](super::HFRepository) such as
/// [`runtime`](super::HFRepository::runtime) and
/// [`pause`](super::HFRepository::pause).
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeSpace;

/// Kernel-repository marker. See [`RepoType`] for the trait, the per-kind string
/// table, and usage examples; use [`HFClient::kernel`](crate::HFClient::kernel) for the
/// typed shortcut. Note that
/// [`HFRepository::<RepoTypeKernel>::info`](super::HFRepository::info) returns a slim
/// shape — to get full model-style metadata for a kernel repo, build a model handle
/// for the same id and call `info()` on that.
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
