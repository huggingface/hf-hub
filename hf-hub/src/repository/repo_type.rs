//! Type-level markers for the four Hugging Face Hub repo kinds.
//!
//! The [`RepoType`] sealed trait and its four marker structs ([`RepoTypeModel`],
//! [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]) parameterise
//! [`HFRepository<T>`](super::HFRepository) and
//! [`HFRepositorySync<T>`](crate::HFRepositorySync) so the repo kind is encoded in the
//! type system. Methods that differ per repo kind (such as `info`) dispatch at compile time.
//!
//! All items here are re-exported flat at `hf_hub::repository::…` and at the crate root.

mod sealed {
    pub trait Sealed {}
}

/// Type-level marker for a Hugging Face Hub repo kind (model, dataset, Space, or kernel).
///
/// `RepoType` is a sealed trait — implemented only by the four marker structs in this
/// module ([`RepoTypeModel`], [`RepoTypeDataset`], [`RepoTypeSpace`], [`RepoTypeKernel`]).
/// Pick the marker as the `T` parameter on [`HFRepository<T>`](super::HFRepository) /
/// [`HFRepositorySync<T>`](crate::HFRepositorySync) at construction time (e.g. via
/// [`HFClient::model`](crate::HFClient::model) / [`HFClient::dataset`](crate::HFClient::dataset) /
/// [`HFClient::space`](crate::HFClient::space) / [`HFClient::kernel`](crate::HFClient::kernel))
/// to lock the repo type into the type system; methods that differ per repo kind (such
/// as `info`) are then resolved at compile time.
///
/// The trait exposes the fragments used to build Hub URLs and identify the repo kind in
/// logs and error messages: [`singular`](Self::singular) (`"model"`, `"dataset"`, `"space"`,
/// `"kernel"`), [`plural`](Self::plural) (the API segment — `"models"`, `"datasets"`,
/// `"spaces"`, `"kernels"`), and [`url_prefix`](Self::url_prefix) (the path segment used by
/// resolve URLs — `""` for models, `"<plural>/"` for everything else).
///
/// All methods return the same value regardless of the receiver — they're properties of
/// the type itself. The marker structs are zero-sized and `Default`, so when only the
/// type is in scope (inside an `impl<T: RepoType>` block, for example), use
/// `T::default().singular()` to materialize an instance to call the method on.
pub trait RepoType: sealed::Sealed + Default + Copy + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Lowercase singular name used in logs, error messages, and request bodies — e.g. `"model"`.
    fn singular(&self) -> &'static str;
    /// Lowercase plural form, used as the `/api/{plural}/...` segment — e.g. `"models"`.
    fn plural(&self) -> &'static str;
    /// Path prefix used in `<endpoint>/{prefix}{repo_id}/resolve/...` URLs.
    /// Empty for models; `"<plural>/"` for all other repo kinds.
    fn url_prefix(&self) -> &'static str;
}

/// Marker for a model repository on the Hub. Implements [`RepoType`].
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeModel;

/// Marker for a dataset repository on the Hub. Implements [`RepoType`].
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeDataset;

/// Marker for a Space repository on the Hub. Implements [`RepoType`].
#[derive(Default, Copy, Clone, Debug)]
pub struct RepoTypeSpace;

/// Marker for a kernel repository on the Hub. Implements [`RepoType`].
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
