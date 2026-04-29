use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT};
use tracing::debug;

use crate::constants;
use crate::error::{HFError, HFResult, HttpErrorContext, NotFoundContext};
use crate::retry::{self, RetryConfig};

/// Async client for the Hugging Face Hub API.
///
/// `HFClient` wraps an `Arc<HFClientInner>` so it is cheap to clone — all clones
/// share the same underlying HTTP connection pool, token, and configuration.
///
/// # Creating a client
///
/// ```rust,no_run
/// use hf_hub::HFClient;
///
/// // Reads token and settings from the environment (HF_TOKEN, HF_ENDPOINT, …).
/// let client = HFClient::new()?;
///
/// // Or configure explicitly:
/// let client = HFClient::builder().token("hf_…").endpoint("https://huggingface.co").build()?;
/// # Ok::<(), hf_hub::HFError>(())
/// ```
pub struct HFClient {
    pub(crate) inner: Arc<HFClientInner>,
}

impl Clone for HFClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for HFClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("HFClient");
        s.field("endpoint", &self.inner.endpoint);
        s.field("token_set", &self.inner.token.is_some());
        s.field("cache_enabled", &self.inner.cache_enabled);
        if self.inner.cache_enabled {
            s.field("cache_dir", &self.inner.cache_dir);
        }
        s.finish()
    }
}

pub(crate) struct HFClientInner {
    pub(crate) client: reqwest::Client,
    pub(crate) no_redirect_client: reqwest::Client,
    pub(crate) retry_config: RetryConfig,
    pub(crate) endpoint: String,
    pub(crate) token: Option<String>,
    pub(crate) cache_dir: std::path::PathBuf,
    pub(crate) cache_enabled: bool,
    pub(crate) xet_state: std::sync::Mutex<crate::xet::XetState>,
}

/// Builder for [`HFClient`].
///
/// Get one via [`HFClient::builder()`] or [`HFClientBuilder::new()`].
/// Call [`build()`](HFClientBuilder::build) when all options are set.
pub struct HFClientBuilder {
    endpoint: Option<String>,
    token: Option<String>,
    user_agent: Option<String>,
    headers: Option<HeaderMap>,
    client: Option<reqwest::Client>,
    cache_dir: Option<std::path::PathBuf>,
    cache_enabled: Option<bool>,
    retry_max_attempts: Option<usize>,
    retry_base_delay: Option<Duration>,
}

impl HFClientBuilder {
    /// Creates a builder with all options unset; defaults are applied at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            endpoint: None,
            token: None,
            user_agent: None,
            headers: None,
            client: None,
            cache_dir: None,
            cache_enabled: None,
            retry_max_attempts: None,
            retry_base_delay: None,
        }
    }

    /// Overrides the Hub base URL (default: `https://huggingface.co`, or `HF_ENDPOINT` env var).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Sets the authentication token. Without this, the client falls back to the `HF_TOKEN` env
    /// var and the cached token file written by `huggingface-cli login`.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Overrides the `User-Agent` header sent with every request.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Merges additional default headers into every request. Ignored when a custom
    /// [`client`](Self::client) is supplied (configure headers on that client directly).
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers = Some(headers);
        self
    }

    /// Supplies a pre-configured `reqwest::Client`. Retry middleware is still applied on top.
    /// The caller is responsible for any default headers (including `User-Agent`) on this client;
    /// the [`headers`](Self::headers) and [`user_agent`](Self::user_agent) options are ignored.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the local cache directory. Defaults to `HF_HUB_CACHE` → `$HF_HOME/hub` →
    /// `~/.cache/huggingface/hub`.
    pub fn cache_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Enables or disables the local file cache. Caching is on by default.
    pub fn cache_enabled(mut self, enabled: bool) -> Self {
        self.cache_enabled = Some(enabled);
        self
    }

    /// Overrides the maximum number of retry attempts after an initial failure (default: 3).
    pub fn retry_max_attempts(mut self, n: usize) -> Self {
        self.retry_max_attempts = Some(n);
        self
    }

    /// Overrides the base delay for exponential backoff between retries (default: 100ms).
    pub fn retry_base_delay(mut self, delay: Duration) -> Self {
        self.retry_base_delay = Some(delay);
        self
    }

    /// Builds the [`HFClient`].
    ///
    /// # Errors
    ///
    /// Returns an error if the endpoint URL is not a valid URL or if the `reqwest` client
    /// cannot be constructed (e.g., an invalid `User-Agent` string was provided).
    pub fn build(self) -> HFResult<HFClient> {
        let endpoint = self
            .endpoint
            .or_else(|| std::env::var(constants::HF_ENDPOINT).ok())
            .unwrap_or_else(|| constants::DEFAULT_HF_ENDPOINT.to_string());

        let _ = url::Url::parse(&endpoint)?;

        let token = self.token.or_else(resolve_token);

        let cache_dir = self.cache_dir.unwrap_or_else(constants::resolve_cache_dir);

        let mut default_headers = self.headers.unwrap_or_default();

        let user_agent = self.user_agent.unwrap_or_else(|| {
            let ua_origin = std::env::var(constants::HF_HUB_USER_AGENT_ORIGIN).ok();
            match ua_origin {
                Some(origin) => format!("hf-hub/{}; {origin}", env!("CARGO_PKG_VERSION")),
                None => format!("hf-hub/{}", env!("CARGO_PKG_VERSION")),
            }
        });
        default_headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&user_agent)
                .map_err(|e| HFError::InvalidParameter(format!("invalid user agent {user_agent:?}: {e}")))?,
        );

        let client = match self.client {
            Some(c) => c,
            None => reqwest::Client::builder().default_headers(default_headers.clone()).build()?,
        };

        let no_redirect_client = reqwest::Client::builder()
            .default_headers(default_headers)
            .redirect(reqwest::redirect::Policy::none())
            .build()?;

        let retry_config = RetryConfig {
            max_attempts: self.retry_max_attempts.unwrap_or(retry::DEFAULT_MAX_ATTEMPTS),
            base_delay: self.retry_base_delay.unwrap_or(retry::DEFAULT_BASE_DELAY),
        };

        Ok(HFClient {
            inner: Arc::new(HFClientInner {
                client,
                no_redirect_client,
                retry_config,
                endpoint: endpoint.trim_end_matches('/').to_string(),
                token,
                cache_dir,
                cache_enabled: self.cache_enabled.unwrap_or(true),
                xet_state: std::sync::Mutex::new(crate::xet::XetState::default()),
            }),
        })
    }

    /// Builds the [`crate::blocking::HFClientSync`].
    ///
    /// Requires the `blocking` feature and is equivalent to calling
    /// [`build`](Self::build) and then [`HFClientSync::from_inner`](crate::HFClientSync::from_inner).
    ///
    /// # Errors
    ///
    /// Returns an error if the endpoint URL is not a valid URL or if the `reqwest` client
    /// cannot be constructed (e.g., an invalid `User-Agent` string was provided), or if the
    /// tokio runtime handle could not be correctly created for the blocking client.
    #[cfg(feature = "blocking")]
    #[cfg_attr(docsrs, doc(cfg(feature = "blocking")))]
    pub fn build_sync(self) -> HFResult<crate::blocking::HFClientSync> {
        let async_client = self.build()?;
        let client = crate::blocking::HFClientSync::from_inner(async_client)?;
        Ok(client)
    }
}

impl Default for HFClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HFClient {
    /// Creates a client with default settings, reading the token and endpoint from the
    /// environment. Equivalent to `HFClient::builder().build()`.
    ///
    /// # Errors
    ///
    /// Fails if the resolved endpoint URL is invalid or the HTTP client cannot be built.
    pub fn new() -> HFResult<Self> {
        HFClientBuilder::new().build()
    }

    /// Returns an [`HFClientBuilder`] for fine-grained configuration.
    pub fn builder() -> HFClientBuilder {
        HFClientBuilder::new()
    }

    pub(crate) fn http_client(&self) -> &reqwest::Client {
        &self.inner.client
    }

    pub(crate) fn no_redirect_client(&self) -> &reqwest::Client {
        &self.inner.no_redirect_client
    }

    pub(crate) fn retry_config(&self) -> &RetryConfig {
        &self.inner.retry_config
    }

    /// Hub base URL this client targets, with any trailing slash trimmed.
    ///
    /// Resolved at [`build`](HFClientBuilder::build) time from
    /// [`HFClientBuilder::endpoint`] → `HF_ENDPOINT` → the default
    /// (`https://huggingface.co`).
    pub fn endpoint(&self) -> &str {
        &self.inner.endpoint
    }

    /// Local cache directory used for downloaded files.
    ///
    /// Resolved at [`build`](HFClientBuilder::build) time from
    /// [`HFClientBuilder::cache_dir`] → `HF_HUB_CACHE` → `$HF_HOME/hub` →
    /// `~/.cache/huggingface/hub`. Returned even when caching is disabled —
    /// see [`cache_enabled`](Self::cache_enabled) to check that.
    pub fn cache_dir(&self) -> &std::path::Path {
        &self.inner.cache_dir
    }

    /// Whether the local file cache is enabled. Set via
    /// [`HFClientBuilder::cache_enabled`]; defaults to `true`.
    pub fn cache_enabled(&self) -> bool {
        self.inner.cache_enabled
    }

    /// Build authorization headers for requests
    pub(crate) fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(ref token) = self.inner.token
            && let Ok(val) = HeaderValue::from_str(&format!("Bearer {token}"))
        {
            headers.insert(AUTHORIZATION, val);
        }
        headers
    }

    /// Build a URL for the API: {endpoint}/api/{segment}/{repo_id}
    pub(crate) fn api_url(&self, repo_type: Option<crate::repository::RepoType>, repo_id: &str) -> String {
        let segment = constants::repo_type_api_segment(repo_type);
        format!("{}/api/{}/{}", self.endpoint(), segment, repo_id)
    }

    /// Build a download URL: {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filename}
    pub(crate) fn download_url(
        &self,
        repo_type: Option<crate::repository::RepoType>,
        repo_id: &str,
        revision: &str,
        filename: &str,
    ) -> String {
        let prefix = constants::repo_type_url_prefix(repo_type);
        format!("{}/{}{}/resolve/{}/{}", self.endpoint(), prefix, repo_id, revision, filename)
    }

    /// Build a bucket API URL: `{endpoint}/api/buckets/{bucket_id}`
    pub(crate) fn bucket_api_url(&self, bucket_id: &str) -> String {
        format!("{}/api/buckets/{}", self.endpoint(), bucket_id)
    }

    /// Check an HTTP response and map error status codes to HFError variants.
    /// Returns the response on success (2xx).
    ///
    /// `repo_id` and `not_found_ctx` control how 404s are mapped:
    /// - `NotFoundContext::Repo` → `HFError::RepoNotFound`
    /// - `NotFoundContext::Entry { path }` → `HFError::EntryNotFound`
    /// - `NotFoundContext::Revision { revision }` → `HFError::RevisionNotFound`
    /// - `NotFoundContext::Generic` → `HFError::Http`
    pub(crate) async fn check_response(
        &self,
        response: reqwest::Response,
        repo_id: Option<&str>,
        not_found_ctx: NotFoundContext,
    ) -> HFResult<reqwest::Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let retry_after = if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            crate::error::parse_retry_after(response.headers())
        } else {
            None
        };
        let context = Box::new(HttpErrorContext::from_response(response).await);
        let repo_id_str = repo_id.unwrap_or("").to_string();

        match status.as_u16() {
            401 => Err(HFError::AuthRequired { context }),
            403 => Err(HFError::Forbidden { context }),
            404 => match not_found_ctx {
                NotFoundContext::Repo => Err(HFError::RepoNotFound {
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Bucket => Err(HFError::BucketNotFound {
                    bucket_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Entry { path } => Err(HFError::EntryNotFound {
                    path,
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Revision { revision } => Err(HFError::RevisionNotFound {
                    revision,
                    repo_id: repo_id_str,
                    context: Some(context),
                }),
                NotFoundContext::Generic => Err(HFError::Http { context }),
            },
            409 => Err(HFError::Conflict { context }),
            429 => Err(HFError::RateLimited { retry_after, context }),
            _ => Err(HFError::Http { context }),
        }
    }

    /// Get or lazily create the cached XetSession.
    ///
    /// Returns `(session, generation)`. The generation is an opaque counter
    /// that identifies which session instance this is. Pass it to
    /// [`replace_xet_session`](Self::replace_xet_session) so that only the
    /// caller that observed the error triggers a replacement — concurrent
    /// callers that already got a session won't clobber it.
    pub(crate) fn xet_session(&self) -> HFResult<(xet::xet_session::XetSession, u64)> {
        let mut guard = self
            .inner
            .xet_state
            .lock()
            .map_err(|e| HFError::Other(format!("xet session mutex poisoned: {e}")))?;

        if let Some(ref session) = guard.session {
            return Ok((session.clone(), guard.generation));
        }

        let session = xet::xet_session::XetSessionBuilder::new()
            .build()
            .map_err(|e| HFError::xet(crate::error::XetOperation::Session, e))?;
        guard.session = Some(session.clone());
        guard.generation += 1;
        Ok((session, guard.generation))
    }

    /// Replace the cached XetSession only if the generation matches.
    ///
    /// Called by xet call sites when a factory method returns an error.
    /// The generation check ensures that if another thread already replaced
    /// the session, this call is a no-op rather than discarding the fresh one.
    pub(crate) fn replace_xet_session(&self, generation: u64, err: &xet::error::XetError) {
        tracing::warn!(error = %err, generation, "replacing cached XetSession");
        let Ok(mut guard) = self.inner.xet_state.lock() else {
            return;
        };
        if guard.generation == generation {
            guard.session = None;
        }
    }
}

/// Resolve token from environment or a token file.
/// Priority: HF_TOKEN env → HF_TOKEN_PATH file → $HF_HOME/token file.
fn resolve_token() -> Option<String> {
    if let Ok(val) = std::env::var(constants::HF_HUB_DISABLE_IMPLICIT_TOKEN)
        && !val.is_empty()
    {
        debug!("implicit token disabled via HF_HUB_DISABLE_IMPLICIT_TOKEN");
        return None;
    }

    if let Ok(token) = std::env::var(constants::HF_TOKEN)
        && !token.is_empty()
    {
        debug!("resolved token from HF_TOKEN env var");
        return Some(token);
    }

    if let Ok(path) = std::env::var(constants::HF_TOKEN_PATH)
        && let Ok(token) = std::fs::read_to_string(&path)
    {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("resolved token from HF_TOKEN_PATH file");
            return Some(token);
        }
    }

    let hf_home = constants::hf_home();
    let token_path = hf_home.join(constants::TOKEN_FILENAME);
    if let Ok(token) = std::fs::read_to_string(&token_path) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            debug!("resolved token from stored token file");
            return Some(token);
        }
    }

    debug!("no token found");
    None
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::HFClientBuilder;

    #[test]
    fn test_builder_cache_dir_explicit() {
        let client = HFClientBuilder::new().cache_dir("/tmp/my-cache").build().unwrap();
        assert_eq!(client.cache_dir(), std::path::Path::new("/tmp/my-cache"));
    }

    // `#[serial]` because the precedence tests below mutate `HF_HOME`, and the default
    // cache dir is derived from it.
    #[test]
    #[serial]
    fn test_builder_cache_dir_default() {
        let client = HFClientBuilder::new().build().unwrap();
        let path_str = client.cache_dir().to_string_lossy();
        assert!(path_str.contains("huggingface") && path_str.ends_with("hub"));
    }

    #[test]
    fn test_xet_session_lazy_creation() {
        let client = HFClientBuilder::new().build().unwrap();
        assert!(client.inner.xet_state.lock().unwrap().session.is_none());
        let (_s1, _gen) = client.xet_session().unwrap();
        assert!(client.inner.xet_state.lock().unwrap().session.is_some());
    }

    #[test]
    fn test_xet_session_shared_across_clones() {
        let client = HFClientBuilder::new().build().unwrap();
        let clone = client.clone();
        let (_s1, _gen) = client.xet_session().unwrap();
        assert!(clone.inner.xet_state.lock().unwrap().session.is_some());
    }

    #[test]
    fn test_xet_session_recovers_after_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        session.abort().unwrap();

        match session.new_file_download_group() {
            Ok(_) => panic!("expected error after abort"),
            Err(e) => client.replace_xet_session(generation, &e),
        }

        let (recovered, _) = client.xet_session().unwrap();
        assert!(recovered.new_file_download_group().is_ok());
    }

    #[test]
    fn test_xet_session_recovers_after_sigint_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        session.sigint_abort().unwrap();

        client.replace_xet_session(generation, &xet::error::XetError::KeyboardInterrupt);

        let (recovered, _) = client.xet_session().unwrap();
        assert!(recovered.new_file_download_group().is_ok());
    }

    /// Simulates the call-site retry pattern used in xet.rs:
    /// 1. Get session + generation, factory call fails
    /// 2. Call replace_xet_session(generation) to drop the bad session
    /// 3. Get a fresh session, factory call succeeds
    #[test]
    fn test_replace_and_retry_after_abort() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, generation) = client.xet_session().unwrap();
        assert!(session.new_file_download_group().is_ok());

        session.abort().unwrap();

        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                client.replace_xet_session(generation, &e);
                client
                    .xet_session()
                    .unwrap()
                    .0
                    .new_file_download_group()
                    .expect("fresh session factory call should succeed")
            },
        };
        drop(group);
    }

    /// Verifies that replace_xet_session with a stale generation is a no-op.
    #[test]
    fn test_replace_with_stale_generation_is_noop() {
        let client = HFClientBuilder::new().build().unwrap();

        let (session, gen1) = client.xet_session().unwrap();
        session.abort().unwrap();

        // First replace succeeds
        client.replace_xet_session(gen1, &xet::error::XetError::KeyboardInterrupt);

        // Get the fresh session with a new generation
        let (_fresh, gen2) = client.xet_session().unwrap();
        assert_ne!(gen1, gen2);

        // Attempting to replace with the old generation is a no-op
        client.replace_xet_session(gen1, &xet::error::XetError::KeyboardInterrupt);

        // The fresh session is still cached
        let (still_fresh, gen3) = client.xet_session().unwrap();
        assert_eq!(gen2, gen3);
        assert!(still_fresh.new_file_download_group().is_ok());
    }

    #[test]
    fn test_xet_session_reuse_without_replacement() {
        let client = HFClientBuilder::new().build().unwrap();

        let (s1, g1) = client.xet_session().unwrap();
        let (s2, g2) = client.xet_session().unwrap();

        assert_eq!(g1, g2);
        assert!(s1.new_file_download_group().is_ok());
        assert!(s2.new_file_download_group().is_ok());
    }

    /// Token resolution precedence tests.
    ///
    /// These tests mutate process-wide environment variables, so they must run
    /// serially. Each test isolates `HF_HOME` to a tempdir so the developer's
    /// real `~/.cache/huggingface/token` cannot leak into the result.
    mod token_precedence {
        use std::io::Write;

        use serial_test::serial;
        use tempfile::TempDir;

        use super::HFClientBuilder;
        use crate::constants::{HF_HOME, HF_HUB_DISABLE_IMPLICIT_TOKEN, HF_TOKEN, HF_TOKEN_PATH, TOKEN_FILENAME};

        struct EnvGuard {
            saved: Vec<(&'static str, Option<String>)>,
            _hf_home: TempDir,
        }

        impl EnvGuard {
            fn new() -> Self {
                let hf_home = tempfile::tempdir().expect("tempdir for HF_HOME");
                let keys = [HF_TOKEN, HF_TOKEN_PATH, HF_HOME, HF_HUB_DISABLE_IMPLICIT_TOKEN];
                let saved = keys.iter().map(|k| (*k, std::env::var(*k).ok())).collect();
                for k in keys {
                    unsafe { std::env::remove_var(k) };
                }
                unsafe { std::env::set_var(HF_HOME, hf_home.path()) };
                Self {
                    saved,
                    _hf_home: hf_home,
                }
            }
        }

        impl Drop for EnvGuard {
            fn drop(&mut self) {
                for (k, v) in &self.saved {
                    match v {
                        Some(val) => unsafe { std::env::set_var(k, val) },
                        None => unsafe { std::env::remove_var(k) },
                    }
                }
            }
        }

        fn write_token_file(dir: &std::path::Path, name: &str, contents: &str) -> std::path::PathBuf {
            let path = dir.join(name);
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(contents.as_bytes()).unwrap();
            path
        }

        #[test]
        #[serial]
        fn test_explicit_token_overrides_env() {
            let _g = EnvGuard::new();
            unsafe { std::env::set_var(HF_TOKEN, "env-token") };

            let client = HFClientBuilder::new().token("explicit-token").build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("explicit-token"));
        }

        #[test]
        #[serial]
        fn test_env_token_used_when_no_explicit() {
            let _g = EnvGuard::new();
            unsafe { std::env::set_var(HF_TOKEN, "env-token") };

            let client = HFClientBuilder::new().build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("env-token"));
        }

        #[test]
        #[serial]
        fn test_env_token_overrides_token_path_file() {
            let g = EnvGuard::new();
            let dir = tempfile::tempdir().unwrap();
            let token_file = write_token_file(dir.path(), "tok", "file-token");
            unsafe { std::env::set_var(HF_TOKEN, "env-token") };
            unsafe { std::env::set_var(HF_TOKEN_PATH, &token_file) };

            let client = HFClientBuilder::new().build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("env-token"));
            drop(g);
        }

        #[test]
        #[serial]
        fn test_token_path_file_used_when_no_env() {
            let _g = EnvGuard::new();
            let dir = tempfile::tempdir().unwrap();
            let token_file = write_token_file(dir.path(), "tok", "file-token\n");
            unsafe { std::env::set_var(HF_TOKEN_PATH, &token_file) };

            let client = HFClientBuilder::new().build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("file-token"));
        }

        #[test]
        #[serial]
        fn test_token_from_hf_home_file() {
            let g = EnvGuard::new();
            // HF_HOME was set to a tempdir by EnvGuard. Place a token file inside it.
            let hf_home = std::env::var(HF_HOME).unwrap();
            write_token_file(std::path::Path::new(&hf_home), TOKEN_FILENAME, "home-token");

            let client = HFClientBuilder::new().build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("home-token"));
            drop(g);
        }

        #[test]
        #[serial]
        fn test_disable_implicit_token_returns_none() {
            let _g = EnvGuard::new();
            unsafe { std::env::set_var(HF_TOKEN, "env-token") };
            unsafe { std::env::set_var(HF_HUB_DISABLE_IMPLICIT_TOKEN, "1") };

            let client = HFClientBuilder::new().build().unwrap();
            assert!(client.inner.token.is_none());
        }

        #[test]
        #[serial]
        fn test_disable_implicit_token_does_not_block_explicit() {
            let _g = EnvGuard::new();
            unsafe { std::env::set_var(HF_HUB_DISABLE_IMPLICIT_TOKEN, "1") };

            let client = HFClientBuilder::new().token("explicit-token").build().unwrap();
            assert_eq!(client.inner.token.as_deref(), Some("explicit-token"));
        }

        #[test]
        #[serial]
        fn test_no_token_anywhere_is_none() {
            let _g = EnvGuard::new();
            // EnvGuard isolates HF_HOME to a fresh tempdir with no token file inside.
            let client = HFClientBuilder::new().build().unwrap();
            assert!(client.inner.token.is_none());
        }
    }
}
