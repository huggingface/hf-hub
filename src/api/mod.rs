use std::time::Duration;

use serde::{Deserialize, Serialize};

/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
#[cfg(feature = "ureq")]
pub mod sync;

/// Siblings are simplified file descriptions of remote files on the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Siblings {
    /// The path within the repo.
    pub rfilename: String,
}

/// The description of the repo given by the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RepoInfo {
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,

    /// The commit sha of the repo.
    pub sha: String,
}

/// The state of a download progress
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    /// The resource to download
    pub url: String,

    /// The progress expressed as a value between 0 and 1
    pub percentage: f32,

    /// Time elapsed since the download as being started
    pub elapsed_time: Duration,

    /// Estimated time to complete the download
    pub remaining_time: Duration,
}
