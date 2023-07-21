use serde::Deserialize;

/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
pub mod sync;

/// Siblings are simplified file descriptions of remote files on the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Siblings {
    /// The path within the repo.
    pub rfilename: String,
}

/// The description of a repo given by the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RepoInfo {
    /// Git commit sha
    pub sha: String,
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RepoSha {
    pub sha: String,
}
