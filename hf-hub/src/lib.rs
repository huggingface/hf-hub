//! # hf-hub
//!
//! Async Rust client for the [Hugging Face Hub API](https://huggingface.co/docs/hub/api).
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use hf_hub::{HFClient, RepoInfoParams};
//!
//! #[tokio::main]
//! async fn main() -> hf_hub::Result<()> {
//!     let api = HFClient::new()?;
//!     let info = api.model("openai-community", "gpt2").info(&RepoInfoParams::default()).await?;
//!     println!("Repo: {:?}", info);
//!     Ok(())
//! }
//! ```

#[macro_use]
mod macros;
pub mod api;
#[cfg(feature = "blocking")]
pub mod blocking;
#[cfg(feature = "buckets")]
pub mod bucket;
pub mod cache;
pub mod client;
pub(crate) mod constants;
pub mod diff;
pub mod error;
pub mod pagination;
pub mod repository;
pub mod types;
#[cfg(feature = "xet")]
pub mod xet;

pub mod test_utils;

#[cfg(all(feature = "blocking", feature = "buckets"))]
pub use blocking::HFBucketSync;
#[cfg(feature = "blocking")]
pub use blocking::{HFClientSync, HFRepoSync, HFRepositorySync, HFSpaceSync};
#[cfg(feature = "buckets")]
pub use bucket::*;
pub use client::{HFClient, HFClientBuilder};
#[cfg(feature = "cli")]
#[doc(hidden)]
pub use constants::{hf_home, resolve_cache_dir};
pub use error::{HFError, Result};
pub use repository::*;
pub use types::*;
