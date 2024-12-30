use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
#[cfg(feature = "ureq")]
pub mod sync;

const HF_ENDPOINT: &str = "HF_ENDPOINT";

/// This trait is used by users of the lib
/// to implement custom behavior during file downloads
pub trait Progress {
    /// At the start of the download
    /// The size is the total size in bytes of the file.
    fn init(&mut self, size: usize, filename: &str);
    /// This function is called whenever `size` bytes have been
    /// downloaded in the temporary file
    fn update(&mut self, size: usize);
    /// This is called at the end of the download
    fn finish(&mut self);
}

impl Progress for () {
    fn init(&mut self, _size: usize, _filename: &str) {}
    fn update(&mut self, _size: usize) {}
    fn finish(&mut self) {}
}

impl Progress for ProgressBar {
    fn init(&mut self, size: usize, filename: &str) {
        self.set_length(size as u64);
        self.set_style(
                ProgressStyle::with_template(
                    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec} ({eta})",
                )
                    .unwrap(), // .progress_chars("━ "),
            );
        let maxlength = 30;
        let message = if filename.len() > maxlength {
            format!("..{}", &filename[filename.len() - maxlength..])
        } else {
            filename.to_string()
        };
        self.set_message(message);
    }

    fn update(&mut self, size: usize) {
        self.inc(size as u64)
    }

    fn finish(&mut self) {
        ProgressBar::finish(self);
    }
}

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
