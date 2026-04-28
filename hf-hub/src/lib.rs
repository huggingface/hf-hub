//! # hf-hub
//!
//! Async Rust client for the [Hugging Face Hub API](https://huggingface.co/docs/hub/api) —
//! the Rust counterpart to the Python [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library.
//!
//! The crate exposes a high-level, ergonomic API built around a single entry point,
//! [`HFClient`], and a family of typed handles ([`HFRepository`], [`HFSpace`],
//! [`HFBucket`]) that scope operations to a specific resource. All network I/O is
//! async and driven by the [`reqwest`](https://docs.rs/reqwest) HTTP client with built-in retries on transient failures.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use hf_hub::HFClient;
//!
//! #[tokio::main]
//! async fn main() -> hf_hub::HFResult<()> {
//!     let client = HFClient::new()?;
//!     let info = client.model("openai-community", "gpt2").info().send().await?;
//!     println!("Repo: {:?}", info);
//!     Ok(())
//! }
//! ```
//!
//! ## Feature overview
//!
//! - **Repositories** — read info, list contents, create, delete, move, and update settings for models, datasets, and
//!   Spaces.
//! - **Files** — list, download (with optional local cache or `local_dir`), upload single files or whole folders, and
//!   build multi-operation commits.
//! - **Commits & refs** — paginate commit history, compute diffs between revisions, and manage branches and tags.
//! - **Users & orgs** — `whoami`, authentication checks, profile lookup, and follower/following lists.
//! - **Spaces** — runtime, hardware, secrets, variables, pause/restart.
//! - **Buckets** — namespaced storage buckets, tree listings, and bucket sync plans.
//! - **Xet transfers** — high-performance chunk-deduplicated uploads and downloads integrated transparently into the
//!   file APIs.
//! - **Optional blocking API** — synchronous counterparts to every async handle when the `blocking` feature is enabled.
//!
//! ## Creating a client
//!
//! [`HFClient::new()`] resolves configuration from the environment:
//!
//! | Variable | Purpose |
//! |---|---|
//! | `HF_TOKEN` | Authentication token (preferred source) |
//! | `HF_TOKEN_PATH` | Path to a file containing the token |
//! | `HF_ENDPOINT` | Override the Hub base URL |
//! | `HF_HOME` | Root for Hugging Face state (defaults to `~/.cache/huggingface`) |
//! | `HF_HUB_CACHE` | Cache directory for downloaded files |
//! | `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Ignore the ambient `HF_TOKEN`/token file |
//!
//! For explicit configuration use [`HFClient::builder()`]:
//!
//! ```rust,no_run
//! use hf_hub::HFClient;
//!
//! let client = HFClient::builder()
//!     .token("hf_xxx")
//!     .endpoint("https://huggingface.co")
//!     .cache_dir("/tmp/hf-cache")
//!     .build()?;
//! # Ok::<(), hf_hub::HFError>(())
//! ```
//!
//! [`HFClient`] wraps an `Arc<…>` internally, so cloning it is cheap and all clones
//! share the same connection pool, token, and cache configuration.
//!
//! ## Repository handles
//!
//! Rather than passing `(repo_type, owner, name)` to every method, bind them once
//! with a typed handle:
//!
//! ```rust,no_run
//! use hf_hub::{HFClient, RepoType};
//!
//! # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
//! let client = HFClient::new()?;
//!
//! let model = client.model("openai-community", "gpt2");
//! let dataset = client.dataset("HuggingFaceFW", "fineweb");
//! let space = client.space("huggingface", "diffusers-gallery");
//!
//! // Or a generic repo when the type is only known at runtime:
//! let repo = client.repo(RepoType::Model, "openai-community", "gpt2");
//!
//! let exists = model.exists().send().await?;
//! # let _ = (dataset, space, repo, exists); Ok(()) }
//! ```
//!
//! [`HFSpace`] dereferences to [`HFRepository`], so every generic repo method is
//! available on Space handles in addition to Space-specific operations like
//! hardware and secret management.
//!
//! ## File operations
//!
//! File APIs live on the repository handle. Downloads go through the local cache
//! by default, producing a path that is safe to read from even across concurrent
//! calls:
//!
//! ```rust,no_run
//! use hf_hub::HFClient;
//!
//! # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
//! let client = HFClient::new()?;
//! let path = client
//!     .model("openai-community", "gpt2")
//!     .download_file()
//!     .filename("config.json")
//!     .send()
//!     .await?;
//! println!("cached at {}", path.display());
//! # Ok(()) }
//! ```
//!
//! Uploads accept bytes, files, or entire folders, and can be batched into a single
//! commit via [`HFRepository::create_commit`] with [`repository::CommitOperation`]s.
//!
//! ## Pagination
//!
//! Endpoints that return a stream of results — commit history, repo listings,
//! recursive tree walks — return `impl Stream<Item = Result<T>>`. Use the
//! [`futures::StreamExt`](https://docs.rs/futures) adapters to iterate:
//!
//! ```rust,no_run
//! use futures::StreamExt;
//! use hf_hub::HFClient;
//!
//! # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
//! let client = HFClient::new()?;
//! let model = client.model("openai-community", "gpt2");
//! let stream = model.list_tree().recursive(true).send()?;
//! futures::pin_mut!(stream);
//! while let Some(entry) = stream.next().await {
//!     println!("{:?}", entry?);
//! }
//! # Ok(()) }
//! ```
//!
//! ## Blocking API
//!
//! Enable the `blocking` feature for synchronous wrappers that manage a dedicated
//! tokio runtime internally:
//!
//! ```toml
//! [dependencies]
//! hf-hub = { version = "1", features = ["blocking"] }
//! ```
//!
//! ```rust,no_run
//! #[cfg(feature = "blocking")]
//! fn main() -> Result<(), hf_hub::HFError> {
//!     use hf_hub::HFClientSync;
//!
//!     let client = HFClientSync::new()?;
//!     let _info = client.model("openai-community", "gpt2").info().send()?;
//!     Ok(())
//! }
//!
//! #[cfg(not(feature = "blocking"))]
//! fn main() {}
//! ```
//!
//! The blocking handles ([`HFClientSync`], [`HFRepositorySync`], [`HFSpaceSync`],
//! [`HFBucketSync`]) mirror their async counterparts method-for-method.
//!
//! ## Errors
//!
//! All fallible operations return [`Result<T>`][Result] = `Result<T, `[`HFError`]`>`.
//! `HFError` distinguishes common Hub conditions — [`RepoNotFound`][HFError::RepoNotFound],
//! [`EntryNotFound`][HFError::EntryNotFound], [`RevisionNotFound`][HFError::RevisionNotFound],
//! [`AuthRequired`][HFError::AuthRequired], [`Forbidden`][HFError::Forbidden], and
//! [`RateLimited`][HFError::RateLimited] — so you can match on them directly without
//! parsing HTTP status codes or response bodies.
//!
//! ## Caching
//!
//! Downloaded files are stored in a content-addressed cache under
//! [`HF_HUB_CACHE`][crate::resolve_cache_dir] (default:
//! `$HF_HOME/hub` → `~/.cache/huggingface/hub`). Concurrent downloads of the same
//! file are coordinated with an on-disk lock, and subsequent calls resolve to the
//! existing cached blob without re-downloading.
//!
//! Disable caching entirely with
//! [`HFClientBuilder::cache_enabled(false)`][HFClientBuilder::cache_enabled], or
//! bypass it for a single request by setting `.local_dir(...)` on the download
//! builder.
//!
//! To inspect the cache — list cached repos, revisions, and disk usage — use
//! [`HFClient::scan_cache`][crate::HFClient::scan_cache], which returns an
//! [`HFCacheInfo`][crate::cache::HFCacheInfo] tree.
//!
//! ## Cargo features
//!
//! - `blocking` — enables the synchronous `*Sync` handles.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod client;
pub(crate) mod constants;
mod error;
mod pagination;
mod retry;

#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
mod blocking;

pub mod buckets;
pub mod cache;
pub mod progress;
pub mod repository;
pub mod spaces;
pub mod users;
pub(crate) mod xet;

#[cfg(feature = "blocking")]
#[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
pub use blocking::{HFBucketSync, HFClientSync, HFRepositorySync, HFSpaceSync};
#[doc(inline)]
pub use buckets::_handle::HFBucket;
pub use client::{HFClient, HFClientBuilder};
#[doc(hidden)]
pub use constants::{hf_home, resolve_cache_dir};
pub use error::{HFError, HFResult};
#[doc(inline)]
pub use repository::_handle::HFRepository;
#[doc(inline)]
pub use repository::_kind::RepoType;
#[doc(inline)]
pub use spaces::_handle::HFSpace;
