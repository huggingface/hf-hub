use std::future::Future;
use std::time::Duration;

use reqwest::{Error as ReqwestError, Response, StatusCode};
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use tracing::{debug, error};

use crate::error::parse_retry_after;

pub(crate) const DEFAULT_MAX_ATTEMPTS: usize = 5;
pub(crate) const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(100);
/// Cap on a server-requested `Retry-After` wait, so one response can't stall a call for minutes.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

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
    // The wasm reqwest backend exposes no `is_connect`/`is_body`/`is_decode`/
    // `is_redirect` and no `hyper::Error` source, so wasm retries only on
    // timeout — strictly less generous than native.
    if err.is_timeout() {
        return true;
    }
    if err.is_builder() || err.is_status() {
        return false;
    }
    #[cfg(target_family = "wasm")]
    {
        false
    }
    #[cfg(not(target_family = "wasm"))]
    {
        if err.is_connect() {
            return true;
        }
        if err.is_body() || err.is_decode() || err.is_redirect() {
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

/// The wait a 429/503 response asks for via `Retry-After`, capped at [`MAX_RETRY_AFTER`].
fn retry_after(result: &Result<Response, ReqwestError>) -> Option<Duration> {
    let resp = result.as_ref().ok()?;
    if !matches!(resp.status(), StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE) {
        return None;
    }
    parse_retry_after(resp.headers()).map(|wait| wait.min(MAX_RETRY_AFTER))
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

/// Delay iterator used between retry attempts.
///
/// Yields at most `config.max_attempts` durations. With `base_delay = B` and `max_attempts = N`
/// the pre-jitter schedule is `2B, 4B, 8B, ..., 2^N * B`; `jitter` multiplies each by a random
/// factor in `[0, 1)`, so the total sleep budget is bounded above by `B * (2^(N+1) - 2)`.
fn delay_strategy(config: &RetryConfig) -> impl Iterator<Item = Duration> {
    let base_ms = config.base_delay.as_millis().min(u64::MAX as u128) as u64;
    ExponentialBackoff::from_millis(2)
        .factor(base_ms)
        .map(jitter)
        .take(config.max_attempts)
}

/// Retry the provided async request factory using the given config.
/// On each attempt the closure is invoked to build a fresh `send()` future.
/// Between attempts it waits the backoff delay, or longer when a 429/503
/// carries a `Retry-After`.
/// Returns the final `Response` (including non-retryable error statuses) or
/// a final transport error.
pub(crate) async fn retry<F, Fut>(config: &RetryConfig, mut f: F) -> Result<Response, ReqwestError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, ReqwestError>>,
{
    let mut delays = delay_strategy(config);

    let mut attempt = 0usize;
    loop {
        let result = f().await;
        let transient = is_transient(&result);
        log_attempt(attempt, transient, &result);

        if !transient {
            return result;
        }

        match delays.next() {
            Some(delay) => tokio::time::sleep(delay.max(retry_after(&result).unwrap_or_default())).await,
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
    /// With max_attempts=4 and base_delay=10ms, the pre-jitter schedule is
    /// 20 + 40 + 80 + 160 = 300ms; jitter only shortens delays, so the total must stay
    /// well under that. A buggy `ExponentialBackoff::from_millis(10)` would yield
    /// 10 + 100 + 1000 + 10000 = 11110ms.
    #[test]
    fn retry_delay_budget_is_bounded() {
        let config = RetryConfig {
            max_attempts: 4,
            base_delay: Duration::from_millis(10),
        };
        let total: Duration = delay_strategy(&config).sum();
        assert!(total < Duration::from_millis(500), "total sleep budget {total:?} exceeds 500ms");
    }

    fn response(status: u16, retry_after: Option<&str>) -> Result<Response, ReqwestError> {
        let mut builder = http::Response::builder().status(status);
        if let Some(value) = retry_after {
            builder = builder.header(reqwest::header::RETRY_AFTER, value);
        }
        Ok(Response::from(builder.body("").unwrap()))
    }

    #[test]
    fn retry_after_applies_to_429_and_503_only() {
        assert_eq!(retry_after(&response(429, Some("3"))), Some(Duration::from_secs(3)));
        assert_eq!(retry_after(&response(503, Some("3"))), Some(Duration::from_secs(3)));
        assert_eq!(retry_after(&response(500, Some("3"))), None);
        assert_eq!(retry_after(&response(429, None)), None);
    }

    #[test]
    fn retry_after_is_capped() {
        assert_eq!(retry_after(&response(429, Some("3600"))), Some(MAX_RETRY_AFTER));
    }

    #[tokio::test]
    async fn retry_waits_for_retry_after() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let served = requests.clone();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let reply = if served.fetch_add(1, Ordering::SeqCst) == 0 {
                    "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                } else {
                    "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                };
                let _ = socket.write_all(reply.as_bytes()).await;
            }
        });

        let client = reqwest::Client::new();
        let config = RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
        };
        let started = std::time::Instant::now();
        let response = retry(&config, || client.get(&url).send()).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert!(started.elapsed() >= Duration::from_secs(1), "retried before Retry-After elapsed");
    }
}
