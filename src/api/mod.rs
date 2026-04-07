use std::{collections::VecDeque, io::Read, path::Path, time::Duration};

use crate::{Repo, RepoType};
use indicatif::{style::ProgressTracker, HumanBytes, ProgressBar, ProgressStyle};
use serde::{Deserialize, Deserializer};
use sha1::Sha1;
use sha2::{Digest, Sha256};

/// The asynchronous version of the API
#[cfg(feature = "tokio")]
pub mod tokio;

/// The synchronous version of the API
#[cfg(feature = "ureq")]
pub mod sync;

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

impl<T: Progress + ?Sized> Progress for &mut T {
    fn init(&mut self, size: usize, filename: &str) {
        (**self).init(size, filename);
    }

    fn update(&mut self, size: usize) {
        (**self).update(size);
    }

    fn finish(&mut self) {
        (**self).finish();
    }
}

impl Progress for ProgressBar {
    fn init(&mut self, size: usize, filename: &str) {
        self.set_length(size as u64);
        self.set_style(
                ProgressStyle::with_template(
                    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec_smoothed} ({eta})",
                ).unwrap().with_key("bytes_per_sec_smoothed", MovingAvgRate::default())
                    ,
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
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct RepoInfo {
    /// Canonical repo id such as `openai-community/gpt2`.
    #[serde(default)]
    pub id: Option<String>,
    /// Legacy model id field returned by some Hub endpoints.
    #[serde(default, rename = "modelId")]
    pub model_id: Option<String>,
    /// Owning user or organization when returned by the endpoint.
    #[serde(default)]
    pub author: Option<String>,
    /// See [`Siblings`]
    pub siblings: Vec<Siblings>,
    /// The commit sha of the repo.
    pub sha: String,
    /// Creation timestamp from the Hub API.
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    /// Last modified timestamp from the Hub API.
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    /// Number of downloads when available.
    #[serde(default)]
    pub downloads: Option<u64>,
    /// Number of likes when available.
    #[serde(default)]
    pub likes: Option<u64>,
    /// Whether the repo is private.
    #[serde(default)]
    pub private: bool,
    /// Whether the repo is gated.
    #[serde(default, deserialize_with = "deserialize_gated_flag")]
    pub gated: Option<bool>,
    /// Whether the repo is disabled.
    #[serde(default)]
    pub disabled: Option<bool>,
    /// Hub tags returned by the endpoint.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional repo description.
    #[serde(default)]
    pub description: Option<String>,
    /// Model pipeline tag when returned by the endpoint.
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    /// Model library name when returned by the endpoint.
    #[serde(default, rename = "library_name")]
    pub library_name: Option<String>,
    /// Space SDK when returned by the endpoint.
    #[serde(default)]
    pub sdk: Option<String>,
}

/// A summarized repository returned from Hub search endpoints.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RepoSummary {
    /// Canonical repo id such as `openai-community/gpt2`.
    pub id: String,
    /// Owning user or organization when returned by the endpoint.
    #[serde(default)]
    pub author: Option<String>,
    /// Commit hash when returned by the endpoint.
    #[serde(default)]
    pub sha: Option<String>,
    /// Creation timestamp from the Hub API.
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    /// Last modified timestamp from the Hub API.
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    /// Number of downloads when available.
    #[serde(default)]
    pub downloads: Option<u64>,
    /// Number of likes when available.
    #[serde(default)]
    pub likes: Option<u64>,
    /// Trending score when available.
    #[serde(default, rename = "trendingScore")]
    pub trending_score: Option<f64>,
    /// Whether the repo is private.
    #[serde(default)]
    pub private: bool,
    /// Whether the repo is gated.
    #[serde(default, deserialize_with = "deserialize_gated_flag")]
    pub gated: Option<bool>,
    /// Whether the repo is disabled.
    #[serde(default)]
    pub disabled: Option<bool>,
    /// Hub tags returned by the search endpoint.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional repo description.
    #[serde(default)]
    pub description: Option<String>,
    /// Model pipeline tag when searching models.
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: Option<String>,
    /// Model library name when searching models.
    #[serde(default, rename = "library_name")]
    pub library_name: Option<String>,
    /// Space SDK when searching spaces.
    #[serde(default)]
    pub sdk: Option<String>,
    #[serde(skip)]
    repo_type: Option<RepoType>,
}

impl RepoSummary {
    pub(crate) fn with_repo_type(mut self, repo_type: RepoType) -> Self {
        self.repo_type = Some(repo_type);
        self
    }

    /// The repo kind used for this search result.
    pub fn repo_type(&self) -> RepoType {
        self.repo_type.expect("repo_type is set by search results")
    }

    /// Convert the summary into a [`Repo`] targeting the default branch.
    pub fn repo(&self) -> Repo {
        Repo::new(self.id.clone(), self.repo_type())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum GatedFlag {
    Bool(bool),
    String(String),
}

fn deserialize_gated_flag<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let parsed = Option::<GatedFlag>::deserialize(deserializer)?;
    Ok(parsed.map(|value| match value {
        GatedFlag::Bool(flag) => flag,
        GatedFlag::String(raw) => {
            let normalized = raw.trim().to_ascii_lowercase();
            !normalized.is_empty()
                && normalized != "false"
                && normalized != "0"
                && normalized != "none"
                && normalized != "no"
        }
    }))
}

/// Query builder shared by the sync and async search APIs.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    repo_type: RepoType,
    query: Option<String>,
    author: Option<String>,
    filters: Vec<String>,
    limit: Option<usize>,
}

impl SearchQuery {
    pub(crate) fn new(repo_type: RepoType) -> Self {
        Self {
            repo_type,
            query: None,
            author: None,
            filters: Vec::new(),
            limit: None,
        }
    }

    pub(crate) fn repo_type(&self) -> RepoType {
        self.repo_type
    }

    pub(crate) fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    pub(crate) fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub(crate) fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filters.push(filter.into());
        self
    }

    pub(crate) fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub(crate) fn query_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();

        if let Some(query) = &self.query {
            pairs.push(("search", query.clone()));
        }
        if let Some(author) = &self.author {
            pairs.push(("author", author.clone()));
        }
        for filter in &self.filters {
            pairs.push(("filter", filter.clone()));
        }
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }

        pairs
    }
}

#[derive(Clone, Default)]
struct MovingAvgRate {
    samples: VecDeque<(std::time::Instant, u64)>,
}

impl ProgressTracker for MovingAvgRate {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, state: &indicatif::ProgressState, now: std::time::Instant) {
        // sample at most every 20ms
        if self
            .samples
            .back()
            .is_none_or(|(prev, _)| (now - *prev) > Duration::from_millis(20))
        {
            self.samples.push_back((now, state.pos()));
        }

        while let Some(first) = self.samples.front() {
            if now - first.0 > Duration::from_secs(1) {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn reset(&mut self, _state: &indicatif::ProgressState, _now: std::time::Instant) {
        self.samples = Default::default();
    }

    fn write(&self, _state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write) {
        match (self.samples.front(), self.samples.back()) {
            (Some((t0, p0)), Some((t1, p1))) if self.samples.len() > 1 => {
                let elapsed_ms = (*t1 - *t0).as_millis();
                let rate = ((p1 - p0) as f64 * 1000f64 / elapsed_ms as f64) as u64;
                write!(w, "{}/s", HumanBytes(rate)).unwrap()
            }
            _ => write!(w, "-").unwrap(),
        }
    }
}

#[derive(Clone)]
pub(crate) enum EtagDigest {
    GitSha1(String),
    Sha256(String),
}

impl EtagDigest {
    pub(crate) fn expected(&self) -> &str {
        match self {
            Self::GitSha1(expected) | Self::Sha256(expected) => expected,
        }
    }
}

pub(crate) fn expected_digest_for_etag(etag: &str) -> Option<EtagDigest> {
    if etag.chars().all(|ch| ch.is_ascii_hexdigit()) {
        match etag.len() {
            40 => Some(EtagDigest::GitSha1(etag.to_string())),
            64 => Some(EtagDigest::Sha256(etag.to_string())),
            _ => None,
        }
    } else {
        None
    }
}

pub(crate) fn file_digest_hex(path: &Path, digest: &EtagDigest) -> Result<String, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = [0u8; 1024 * 1024];
    match digest {
        EtagDigest::GitSha1(_) => {
            let mut hasher = Sha1::new();
            hasher.update(format!("blob {len}\0").as_bytes());
            loop {
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            Ok(format!("{:x}", hasher.finalize()))
        }
        EtagDigest::Sha256(_) => {
            let mut hasher = Sha256::new();
            loop {
                let read = reader.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            Ok(format!("{:x}", hasher.finalize()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RepoInfo, RepoSummary};

    #[test]
    fn repo_info_accepts_string_gated_flag() {
        let json = r#"{
            "id":"org/repo",
            "siblings":[],
            "sha":"abc",
            "private":false,
            "gated":"manual"
        }"#;
        let parsed: RepoInfo = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.gated, Some(true));
    }

    #[test]
    fn repo_summary_accepts_string_gated_flag() {
        let json = r#"{
            "id":"org/repo",
            "private":false,
            "gated":"manual"
        }"#;
        let parsed: RepoSummary = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.gated, Some(true));
    }

    #[test]
    fn repo_summary_accepts_floating_trending_score() {
        let json = r#"{
            "id":"org/repo",
            "private":false,
            "trendingScore":0.7000000000000001
        }"#;
        let parsed: RepoSummary = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.trending_score, Some(0.7000000000000001));
    }
}
