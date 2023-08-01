use serde::Deserialize;

/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
pub mod sync;

// /// Lfs metadata of a file
// #[derive(Debug, Clone, Deserialize, PartialEq)]
// pub struct Lfs {
//     /// The file size in bytes.
//     pub size: u32,
// }

/// Siblings are simplified file descriptions of remote files on the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Siblings {
    /// The path within the repo.
    pub rfilename: String,
    // /// Git blob id
    // #[serde(rename = "blobId")]
    // pub blob_id: String,
    // /// The file size in bytes.
    // /// Note that this can be the LFS pointer size on some occasions
    // /// and not the actual file size. See `lfs`.
    // pub size: u32,
    // /// Lfs metadata
    // pub lfs: Option<Lfs>,
}

/// The description of the repo given by the hub
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RepoInfo {
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,

    /// The commit sha of the repo.
    pub sha: String,
}
