use std::time::{Duration, Instant};

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

/// The download progress event
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

/// Store the state of a download
struct DownloadState {
    start_time: Instant,
    len: usize,
    offset: usize,
    url: String,
}

impl DownloadState {
    fn new(len: usize, url: String) -> DownloadState {
        DownloadState {
            start_time: Instant::now(),
            len,
            offset: 0,
            url,
        }
    }

    fn update(&mut self, delta: usize) -> Option<ProgressEvent> {
        if delta == 0 {
            return None;
        }

        self.offset += delta;

        let elapsed_time = Instant::now() - self.start_time;

        let progress = self.offset as f32 / self.len as f32;
        let progress_100 = progress * 100.;

        let remaing_percentage = 100. - progress_100;
        let duration_unit = elapsed_time / progress_100 as u32;
        let remaining_time = duration_unit * remaing_percentage as u32;

        let event = ProgressEvent {
            url: self.url.clone(),
            percentage: progress,
            elapsed_time,
            remaining_time,
        };
        Some(event)
    }
}
