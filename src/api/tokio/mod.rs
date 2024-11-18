use super::RepoInfo;
use crate::{Cache, Repo, RepoType};
use http::StatusCode;
use rand::Rng;
use regex::Regex;
use reqwest::{
    header::{
        HeaderMap, HeaderName, HeaderValue, InvalidHeaderValue, ToStrError, AUTHORIZATION,
        CONTENT_RANGE, LOCATION, RANGE, USER_AGENT,
    },
    redirect::Policy,
    Client, Error as ReqwestError,
};
use std::{fmt::Display, num::ParseIntError};
use std::{
    future::Future,
    path::{Component, Path, PathBuf},
};
use thiserror::Error;
use tokio::sync::{AcquireError, TryAcquireError};

mod download;
mod repo_info;
mod upload;
pub use upload::{CommitError, UploadSource};

/// Current version (used in user-agent)
const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Current name (used in user-agent)
const NAME: &str = env!("CARGO_PKG_NAME");

/// A custom error type that combines a Reqwest error with the response body.
///
/// This struct wraps a [`reqwest::Error`] and includes the response body as a string,
/// which can be useful for debugging and error reporting when HTTP requests fail.
#[derive(Debug)]
pub struct ReqwestErrorWithBody {
    url: String,
    error: ReqwestError,
    body: Result<String, ReqwestError>,
}

impl Display for ReqwestErrorWithBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Request error: {}", self.url)?;
        writeln!(f, "{}", self.error)?;
        match &self.body {
            Ok(body) => {
                writeln!(f, "Response body:")?;
                writeln!(f, "{body}")?;
            }
            Err(err) => {
                writeln!(f, "Failed to fetch body:")?;
                writeln!(f, "{err}")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ReqwestErrorWithBody {}

// Extension trait for `reqwest::Response` that provides error handling with response body capture.
///
/// This trait adds the ability to check for HTTP error status codes while preserving the response body
/// in case of an error, which is useful for debugging and error reporting.
///
/// # Examples
///
/// ```
/// use hf_hub::api::tokio::HfBadResponse;
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let response = reqwest::get("https://api.example.com/data").await?;
///
///     // Will return Err with both the error and response body if status code is not successful
///     let response = response.maybe_err().await?;
///     
///     // Process successful response...
///     Ok(())
/// }
/// ```
///
/// # Error Handling
///
/// - If the response status is successful (2xx), returns `Ok(Response)`
/// - If the response status indicates an error (4xx, 5xx), returns `Err(ApiError)`
///   containing both the original error and the response body text
pub trait HfBadResponse {
    /// Checks if the response status code indicates an error, and if so, captures the response body
    /// along with the error details.
    ///
    /// Returns a Future that resolves to:
    /// - `Ok(Response)` if the status code is successful
    /// - `Err(ApiError)` if the status code indicates an error
    fn maybe_hf_err(self) -> impl Future<Output = Result<Self, ApiError>>
    where
        Self: Sized;
}

lazy_static::lazy_static! {
    static ref REPO_API_REGEX: Regex = Regex::new(
        r#"(?x)
            # staging or production endpoint
            ^https://[^/]+
            (
                # on /api/repo_type/repo_id
                /api/(models|datasets|spaces)/(.+)
                |
                # or /repo_id/resolve/revision/...
                /(.+)/resolve/(.+)
            )
        "#,
    ).unwrap();
}

impl HfBadResponse for reqwest::Response {
    async fn maybe_hf_err(self) -> Result<Self, ApiError>
    where
        Self: Sized,
    {
        let error = self.error_for_status_ref();
        if let Err(error) = error {
            let hf_error_code = self
                .headers()
                .get("X-Error-Code")
                .and_then(|v| v.to_str().ok());
            let hf_error_message = self
                .headers()
                .get("X-Error-Message")
                .and_then(|v| v.to_str().ok());
            let url = self.url().to_string();
            Err(match (hf_error_code, hf_error_message) {
                (Some("RevisionNotFound"), _) => ApiError::RevisionNotFoundError(url),
                (Some("EntryNotFound"), _) => ApiError::EntryNotFoundError(url),
                (Some("GatedRepo"), _) => ApiError::GatedRepoError(url),
                (_, Some("Access to this resource is disabled.")) => {
                    ApiError::DisabledRepoError(url)
                }
                // 401 is misleading as it is returned for:
                //    - private and gated repos if user is not authenticated
                //    - missing repos
                // => for now, we process them as `RepoNotFound` anyway.
                // See https://gist.github.com/Wauplin/46c27ad266b15998ce56a6603796f0b9
                (Some("RepoNotFound"), _)
                    if self.status() == StatusCode::UNAUTHORIZED
                        && REPO_API_REGEX.is_match(&url) =>
                {
                    ApiError::RepositoryNotFoundError(url)
                }
                (_, _) => {
                    let body = self.text().await;
                    ApiError::RequestErrorWithBody(ReqwestErrorWithBody { url, body, error })
                }
            })
        } else {
            Ok(self)
        }
    }
}

#[derive(Debug, Error)]
/// All errors the API can throw
pub enum ApiError {
    /// Api expects certain header to be present in the results to derive some information
    #[error("Header {0} is missing")]
    MissingHeader(HeaderName),

    /// The header exists, but the value does not conform to what the Api expects.
    #[error("Header {0} is invalid")]
    InvalidHeader(HeaderName),

    /// The value cannot be used as a header during request header construction
    #[error("Invalid header value {0}")]
    InvalidHeaderValue(#[from] InvalidHeaderValue),

    /// The header value is not valid utf-8
    #[error("header value is not a string")]
    ToStr(#[from] ToStrError),

    /// Error in the request
    #[error("request error: {0}")]
    RequestError(#[from] ReqwestError),

    /// Error in the request
    #[error("request error: {0}")]
    RequestErrorWithBody(#[from] ReqwestErrorWithBody),

    /// Error parsing some range value
    #[error("Cannot parse int")]
    ParseIntError(#[from] ParseIntError),

    /// I/O Error
    #[error("I/O error {0}")]
    IoError(#[from] std::io::Error),

    /// We tried to download chunk too many times
    #[error("Too many retries: {0}")]
    TooManyRetries(Box<ApiError>),

    /// Semaphore cannot be acquired
    #[error("Try acquire: {0}")]
    TryAcquireError(#[from] TryAcquireError),

    /// Semaphore cannot be acquired
    #[error("Acquire: {0}")]
    AcquireError(#[from] AcquireError),

    /// Bad data from the API
    #[error("Invalid Response: {0}")]
    InvalidResponse(String),

    /// Repo exists, but the revision / oid doesn't exist.
    #[error("Revision Not Found for url: {0}")]
    RevisionNotFoundError(String),

    /// todo what is this?
    #[error("Entry Not Found for url: {0}")]
    EntryNotFoundError(String),

    /// Repo is gated
    #[error("Cannot access gated repo for url: {0}")]
    GatedRepoError(String),

    /// Repo is disabled
    #[error("Cannot access repo - access to resource is disabled for url: {0}")]
    DisabledRepoError(String),

    /// Repo does not exist for the caller (could be private)
    #[error("Repository Not Found for url: {0}")]
    RepositoryNotFoundError(String),
}

/// Helper to create [`Api`] with all the options.
#[derive(Debug)]
pub struct ApiBuilder {
    endpoint: String,
    cache: Cache,
    token: Option<String>,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

impl Default for ApiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBuilder {
    /// Default api builder
    /// ```
    /// use hf_hub::api::tokio::ApiBuilder;
    /// let api = ApiBuilder::new().build().unwrap();
    /// ```
    pub fn new() -> Self {
        let cache = Cache::default();
        Self::from_cache(cache)
    }

    /// From a given cache
    /// ```
    /// use hf_hub::{api::tokio::ApiBuilder, Cache};
    /// let path = std::path::PathBuf::from("/tmp");
    /// let cache = Cache::new(path);
    /// let api = ApiBuilder::from_cache(cache).build().unwrap();
    /// ```
    pub fn from_cache(cache: Cache) -> Self {
        let token = cache.token();

        let progress = true;

        Self {
            endpoint: "https://huggingface.co".to_string(),
            cache,
            token,
            max_files: num_cpus::get(),
            chunk_size: 10_000_000,
            parallel_failures: 0,
            max_retries: 0,
            progress,
        }
    }

    /// Wether to show a progressbar
    pub fn with_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    /// Changes the location of the cache directory. Defaults is `~/.cache/huggingface/`.
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache = Cache::new(cache_dir);
        self
    }

    /// Sets the t to be used in the API
    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    fn build_headers(&self) -> Result<HeaderMap, ApiError> {
        let mut headers = HeaderMap::new();
        let user_agent = format!("unkown/None; {NAME}/{VERSION}; rust/unknown");
        headers.insert(USER_AGENT, HeaderValue::from_str(&user_agent)?);
        if let Some(token) = &self.token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}"))?,
            );
        }
        Ok(headers)
    }

    /// Consumes the builder and builds the final [`Api`]
    pub fn build(self) -> Result<Api, ApiError> {
        let headers = self.build_headers()?;
        let client = Client::builder().default_headers(headers.clone()).build()?;

        // Policy: only follow relative redirects
        // See: https://github.com/huggingface/huggingface_hub/blob/9c6af39cdce45b570f0b7f8fad2b311c96019804/src/huggingface_hub/file_download.py#L411
        let relative_redirect_policy = Policy::custom(|attempt| {
            // Follow redirects up to a maximum of 10.
            if attempt.previous().len() > 10 {
                return attempt.error("too many redirects");
            }

            if let Some(last) = attempt.previous().last() {
                // If the url is not relative
                if last.make_relative(attempt.url()).is_none() {
                    return attempt.stop();
                }
            }

            // Follow redirect
            attempt.follow()
        });

        let relative_redirect_client = Client::builder()
            .redirect(relative_redirect_policy)
            .default_headers(headers)
            .build()?;
        Ok(Api {
            endpoint: self.endpoint,
            cache: self.cache,
            client,
            relative_redirect_client,
            max_files: self.max_files,
            chunk_size: self.chunk_size,
            parallel_failures: self.parallel_failures,
            max_retries: self.max_retries,
            progress: self.progress,
        })
    }
}

#[derive(Debug)]
struct Metadata {
    commit_hash: String,
    etag: String,
    size: usize,
}

/// The actual Api used to interact with the hub.
/// You can inspect repos with [`Api::info`]
/// or download files with [`Api::download`]
#[derive(Clone, Debug)]
pub struct Api {
    endpoint: String,
    cache: Cache,
    client: Client,
    relative_redirect_client: Client,
    max_files: usize,
    chunk_size: usize,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
}

fn make_relative(src: &Path, dst: &Path) -> PathBuf {
    let path = src;
    let base = dst;

    assert_eq!(
        path.is_absolute(),
        base.is_absolute(),
        "This function is made to look at absolute paths only"
    );
    let mut ita = path.components();
    let mut itb = base.components();

    loop {
        match (ita.next(), itb.next()) {
            (Some(a), Some(b)) if a == b => (),
            (some_a, _) => {
                // Ignoring b, because 1 component is the filename
                // for which we don't need to go back up for relative
                // filename to work.
                let mut new_path = PathBuf::new();
                for _ in itb {
                    new_path.push(Component::ParentDir);
                }
                if let Some(a) = some_a {
                    new_path.push(a);
                    for comp in ita {
                        new_path.push(comp);
                    }
                }
                return new_path;
            }
        }
    }
}

fn symlink_or_rename(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    if dst.exists() {
        return Ok(());
    }

    let rel_src = make_relative(src, dst);
    #[cfg(target_os = "windows")]
    {
        if std::os::windows::fs::symlink_file(rel_src, dst).is_err() {
            std::fs::rename(src, dst)?;
        }
    }

    #[cfg(target_family = "unix")]
    std::os::unix::fs::symlink(rel_src, dst)?;

    Ok(())
}

fn jitter() -> usize {
    rand::thread_rng().gen_range(0..=500)
}

fn exponential_backoff(base_wait_time: usize, n: usize, max: usize) -> usize {
    (base_wait_time + n.pow(2) + jitter()).min(max)
}

impl Api {
    /// Creates a default Api, for Api options See [`ApiBuilder`]
    pub fn new() -> Result<Self, ApiError> {
        ApiBuilder::new().build()
    }

    /// Get the underlying api client
    /// Allows for lower level access
    pub fn client(&self) -> &Client {
        &self.client
    }

    async fn metadata(&self, url: &str) -> Result<Metadata, ApiError> {
        let response = self
            .relative_redirect_client
            .get(url)
            .header(RANGE, "bytes=0-0")
            .send()
            .await?;
        let response = response.error_for_status()?;
        let headers = response.headers();
        let header_commit = HeaderName::from_static("x-repo-commit");
        let header_linked_etag = HeaderName::from_static("x-linked-etag");
        let header_etag = HeaderName::from_static("etag");

        let etag = match headers.get(&header_linked_etag) {
            Some(etag) => etag,
            None => headers
                .get(&header_etag)
                .ok_or(ApiError::MissingHeader(header_etag))?,
        };
        // Cleaning extra quotes
        let etag = etag.to_str()?.to_string().replace('"', "");
        let commit_hash = headers
            .get(&header_commit)
            .ok_or(ApiError::MissingHeader(header_commit))?
            .to_str()?
            .to_string();

        // The response was redirected o S3 most likely which will
        // know about the size of the file
        let response = if response.status().is_redirection() {
            self.client
                .get(headers.get(LOCATION).unwrap().to_str()?.to_string())
                .header(RANGE, "bytes=0-0")
                .send()
                .await?
        } else {
            response
        };
        let headers = response.headers();
        let content_range = headers
            .get(CONTENT_RANGE)
            .ok_or(ApiError::MissingHeader(CONTENT_RANGE))?
            .to_str()?;

        let size = content_range
            .split('/')
            .last()
            .ok_or(ApiError::InvalidHeader(CONTENT_RANGE))?
            .parse()?;
        Ok(Metadata {
            commit_hash,
            etag,
            size,
        })
    }

    /// Creates a new handle [`ApiRepo`] which contains operations
    /// on a particular [`Repo`]
    pub fn repo(&self, repo: Repo) -> ApiRepo {
        ApiRepo::new(self.clone(), repo)
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::tokio::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Model));
    /// ```
    pub fn model(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Model))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::tokio::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Dataset));
    /// ```
    pub fn dataset(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Dataset))
    }

    /// Simple wrapper over
    /// ```
    /// # use hf_hub::{api::tokio::Api, Repo, RepoType};
    /// # let model_id = "gpt2".to_string();
    /// let api = Api::new().unwrap();
    /// let api = api.repo(Repo::new(model_id, RepoType::Space));
    /// ```
    pub fn space(&self, model_id: String) -> ApiRepo {
        self.repo(Repo::new(model_id, RepoType::Space))
    }
}

/// Shorthand for accessing things within a particular repo
#[derive(Debug)]
pub struct ApiRepo {
    api: Api,
    repo: Repo,
}

impl ApiRepo {
    fn new(api: Api, repo: Repo) -> Self {
        Self { api, repo }
    }
}
