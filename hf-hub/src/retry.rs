use std::future::Future;
use std::time::Duration;

use reqwest::{Error as ReqwestError, Response, StatusCode};
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use tracing::{debug, error};

pub(crate) const DEFAULT_MAX_ATTEMPTS: usize = 5;
pub(crate) const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    pub max_attempts: usize,
    pub base_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay: DEFAULT_BASE_DELAY,
        }
    }
}

fn is_transient_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn is_transient_reqwest_error(err: &ReqwestError) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    if err.is_body() || err.is_decode() || err.is_builder() || err.is_redirect() || err.is_status() {
        return false;
    }
    if err.is_request()
        && let Some(hyper_err) = find_source::<hyper::Error>(err)
        && (hyper_err.is_incomplete_message()
            || hyper_err.is_canceled()
            || find_source::<std::io::Error>(hyper_err).is_some())
    {
        return true;
    }
    false
}

fn find_source<T: std::error::Error + 'static>(err: &dyn std::error::Error) -> Option<&T> {
    let mut source = err.source();
    while let Some(e) = source {
        if let Some(t) = e.downcast_ref::<T>() {
            return Some(t);
        }
        source = e.source();
    }
    None
}

fn err_url(err: &ReqwestError) -> String {
    err.url().map(|u| u.to_string()).unwrap_or_else(|| "<unknown url>".to_string())
}

fn is_transient(result: &Result<Response, ReqwestError>) -> bool {
    match result {
        Ok(resp) => is_transient_status(resp.status()),
        Err(e) => is_transient_reqwest_error(e),
    }
}

fn log_attempt(attempt: usize, transient: bool, result: &Result<Response, ReqwestError>) {
    match (transient, result) {
        (true, Ok(resp)) => {
            debug!(attempt, url = %resp.url(), status = %resp.status(), "retrying request");
        },
        (true, Err(e)) => {
            debug!(attempt, url = %err_url(e), error = %e, "retrying request");
        },
        (false, Ok(resp)) => {
            debug!(attempt, url = %resp.url(), status = %resp.status(), "request succeeded");
        },
        (false, Err(_)) => {},
    }
}

fn log_exhausted(max_attempts: usize, result: &Result<Response, ReqwestError>) {
    let url = match result {
        Ok(resp) => resp.url().to_string(),
        Err(e) => err_url(e),
    };
    error!(url = %url, max_attempts, "retry exhausted");
}

/// Retry the provided async request factory using the given config.
/// On each attempt the closure is invoked to build a fresh `send()` future.
/// Returns the final `Response` (including non-retryable error statuses) or
/// a final transport error.
pub(crate) async fn retry<F, Fut>(config: &RetryConfig, mut f: F) -> Result<Response, ReqwestError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, ReqwestError>>,
{
    let base_ms = config.base_delay.as_millis().min(u64::MAX as u128) as u64;
    let mut delays = ExponentialBackoff::from_millis(2)
        .factor(base_ms)
        .map(jitter)
        .take(config.max_attempts);

    let mut attempt = 0usize;
    loop {
        let result = f().await;
        let transient = is_transient(&result);
        log_attempt(attempt, transient, &result);

        if !transient {
            return result;
        }

        match delays.next() {
            Some(delay) => tokio::time::sleep(delay).await,
            None => {
                log_exhausted(config.max_attempts, &result);
                return result;
            },
        }

        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;

    use super::*;

    #[test]
    fn status_classification() {
        for s in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(is_transient_status(s), "{s} should be transient");
        }

        for s in [
            StatusCode::OK,
            StatusCode::CREATED,
            StatusCode::MOVED_PERMANENTLY,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            assert!(!is_transient_status(s), "{s} should be fatal");
        }
    }

    #[tokio::test]
    async fn connect_error_is_transient() {
        let err = reqwest::Client::new().get("http://127.0.0.1:1").send().await.unwrap_err();
        assert!(err.is_connect(), "expected connect error, got {err:?}");
        assert!(is_transient_reqwest_error(&err));
    }

    #[tokio::test]
    async fn builder_error_is_fatal() {
        let err = reqwest::Client::new().get("not-a-url").send().await.unwrap_err();
        assert!(err.is_builder(), "expected builder error, got {err:?}");
        assert!(!is_transient_reqwest_error(&err));
    }

    /// Regression: confirm the backoff schedule is `base_delay * 2^n`, not `base_delay^n`.
    /// With max_attempts=4 and base_delay=10ms, the total sleep budget must stay under
    /// a few hundred ms. A buggy `ExponentialBackoff::from_millis(10)` would yield
    /// 10ms + 100ms + 1000ms + 10000ms = 11s+ for the same inputs.
    #[tokio::test]
    async fn retry_delay_budget_is_bounded() {
        let config = RetryConfig {
            max_attempts: 4,
            base_delay: std::time::Duration::from_millis(10),
        };
        let client = reqwest::Client::new();
        let start = std::time::Instant::now();
        let result = retry(&config, || client.get("http://127.0.0.1:1").send()).await;
        let elapsed = start.elapsed();
        assert!(result.is_err());
        assert!(elapsed < std::time::Duration::from_secs(1), "retry loop took {elapsed:?}, expected <1s");
    }
}
