use serde::Deserialize;

/// The asynchronous version of the API
#[cfg(any(feature = "tokio", feature = "tokio-rustls"))]
pub mod tokio;

/// The synchronous version of the API
#[cfg(feature = "online")]
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
