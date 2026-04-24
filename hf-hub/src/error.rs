use std::time::Duration;

use reqwest::header::HeaderMap;
use thiserror::Error;

/// Context captured from a failed HTTP response.
///
/// Carried by every HTTP-derived variant of [`HFError`]. Populated from the
/// response headers and body by [`HttpErrorContext::from_response`]. The
/// request id, error code, and server message come from the Hub's
/// `X-Request-Id`, `X-Error-Code`, and `X-Error-Message` headers when
/// present; `server_message` falls back to the JSON body's `"error"` field.
#[derive(Debug, Clone)]
pub struct HttpErrorContext {
    pub status: reqwest::StatusCode,
    pub url: String,
    pub request_id: Option<String>,
    pub error_code: Option<String>,
    pub server_message: Option<String>,
    pub body: String,
}

impl HttpErrorContext {
    /// Build context from a non-success response. Drains the response body.
    pub(crate) async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status();
        let url = response.url().to_string();
        let request_id = header_string(response.headers(), "x-request-id");
        let error_code = header_string(response.headers(), "x-error-code");
        let header_message = header_string(response.headers(), "x-error-message");
        let body = response.text().await.unwrap_or_default();
        let server_message = header_message.or_else(|| extract_json_error(&body));
        Self {
            status,
            url,
            request_id,
            error_code,
            server_message,
            body,
        }
    }
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name)?.to_str().ok().map(|s| s.to_string())
}

fn extract_json_error(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value.get("error")?.as_str().map(|s| s.to_string())
}

/// Parse an integer-seconds `Retry-After` header. HTTP-date form is not supported.
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    raw.trim().parse::<u64>().ok().map(Duration::from_secs)
}

fn format_http_suffix(ctx: &HttpErrorContext) -> String {
    match (&ctx.request_id, &ctx.error_code) {
        (Some(rid), Some(code)) => format!(" (request_id={rid}, error_code={code})"),
        (Some(rid), None) => format!(" (request_id={rid})"),
        (None, Some(code)) => format!(" (error_code={code})"),
        (None, None) => String::new(),
    }
}

#[derive(Error, Debug)]
pub enum HFError {
    #[error("HTTP error: {} {}{}", .context.status, .context.url, format_http_suffix(.context))]
    Http { context: Box<HttpErrorContext> },

    #[error("Authentication required: {}{}", .context.url, format_http_suffix(.context))]
    AuthRequired { context: Box<HttpErrorContext> },

    #[error("Repository not found: {repo_id}")]
    RepoNotFound {
        repo_id: String,
        context: Option<Box<HttpErrorContext>>,
    },

    #[error("Revision not found: {revision} in {repo_id}")]
    RevisionNotFound {
        repo_id: String,
        revision: String,
        context: Option<Box<HttpErrorContext>>,
    },

    #[error("Entry not found: {path} in {repo_id}")]
    EntryNotFound {
        path: String,
        repo_id: String,
        context: Option<Box<HttpErrorContext>>,
    },

    #[error("Bucket not found: {bucket_id}")]
    BucketNotFound {
        bucket_id: String,
        context: Option<Box<HttpErrorContext>>,
    },

    #[error("Invalid repository type: expected {expected}, got {actual}")]
    InvalidRepoType {
        expected: crate::repository::RepoType,
        actual: crate::repository::RepoType,
    },

    #[error("Forbidden: {}{}", .context.url, format_http_suffix(.context))]
    Forbidden { context: Box<HttpErrorContext> },

    #[error("Conflict: {}{}", .context.body, format_http_suffix(.context))]
    Conflict { context: Box<HttpErrorContext> },

    #[error("Rate limited: {}{}", .context.url, format_http_suffix(.context))]
    RateLimited {
        retry_after: Option<Duration>,
        context: Box<HttpErrorContext>,
    },

    #[error("File not found in local cache: {path}")]
    LocalEntryNotFound { path: String },

    #[error(
        "Cache is not enabled — set cache_enabled(true) on HFClientBuilder, or provide local_dir in download params"
    )]
    CacheNotEnabled,

    #[error("Cache lock timed out: {}", path.display())]
    CacheLockTimeout { path: std::path::PathBuf },

    #[error("HTTP request error: {source}{}", .url.as_deref().map(|u| format!(" ({u})")).unwrap_or_default())]
    Request {
        #[source]
        source: reqwest::Error,
        url: Option<String>,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Url(#[from] url::ParseError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error(transparent)]
    DiffParse(#[from] crate::repository::HFDiffParseError),

    #[error("{0}")]
    Other(String),
}

impl From<reqwest::Error> for HFError {
    fn from(source: reqwest::Error) -> Self {
        let url = source.url().map(|u| u.to_string());
        HFError::Request { source, url }
    }
}

impl HFError {
    /// Returns true for errors that indicate transient network/server issues
    /// where falling back to a cached version is appropriate.
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            HFError::Request { source, .. } => source.is_connect() || source.is_timeout(),
            HFError::Http { context } => {
                matches!(context.status.as_u16(), 500 | 502 | 503 | 504)
            },
            _ => false,
        }
    }
}

pub type HFResult<T> = std::result::Result<T, HFError>;

/// Context for mapping HTTP 404 errors to specific HFError variants.
pub(crate) enum NotFoundContext {
    /// 404 means the repository does not exist
    Repo,
    /// 404 means the bucket does not exist
    Bucket,
    /// 404 means a file/path does not exist within the repo
    Entry { path: String },
    /// 404 means the revision does not exist
    Revision { revision: String },
    /// No special mapping — use generic Http error
    Generic,
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;
    use reqwest::header::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn retry_after_parses_integer_seconds() {
        let mut h = HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, HeaderValue::from_static("42"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(42)));
    }

    #[test]
    fn retry_after_rejects_http_date() {
        let mut h = HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"));
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn retry_after_absent() {
        let h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn header_string_case_insensitive() {
        let mut h = HeaderMap::new();
        h.insert("X-Request-Id", HeaderValue::from_static("abc123"));
        assert_eq!(header_string(&h, "x-request-id"), Some("abc123".to_string()));
    }

    #[test]
    fn json_error_extraction() {
        let body = r#"{"error": "gated repository"}"#;
        assert_eq!(extract_json_error(body), Some("gated repository".to_string()));

        assert_eq!(extract_json_error("not json"), None);
        assert_eq!(extract_json_error(r#"{"other": "x"}"#), None);
    }

    fn ctx(status: u16) -> HttpErrorContext {
        HttpErrorContext {
            status: StatusCode::from_u16(status).unwrap(),
            url: "https://example".to_string(),
            request_id: None,
            error_code: None,
            server_message: None,
            body: String::new(),
        }
    }

    #[test]
    fn is_transient_classifies_http_statuses() {
        for s in [500u16, 502, 503, 504] {
            assert!(
                HFError::Http {
                    context: Box::new(ctx(s))
                }
                .is_transient(),
                "{s} should be transient"
            );
        }
        for s in [400u16, 401, 403, 404, 409, 429] {
            assert!(
                !HFError::Http {
                    context: Box::new(ctx(s))
                }
                .is_transient(),
                "{s} should not be transient"
            );
        }
    }

    #[test]
    fn display_includes_request_id_when_present() {
        let mut c = ctx(500);
        c.request_id = Some("req-xyz".to_string());
        let msg = HFError::Http { context: Box::new(c) }.to_string();
        assert!(msg.contains("req-xyz"), "display missing request_id: {msg}");
    }

    #[tokio::test]
    async fn from_response_extracts_headers_and_body() {
        // Spin up a tiny server that returns 500 with custom headers + body.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
            let body = r#"{"error":"server said no"}"#;
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 X-Request-Id: req-123\r\n\
                 X-Error-Code: GatedRepo\r\n\
                 X-Error-Message: acceptance required\r\n\
                 \r\n{}",
                body.len(),
                body
            );
            tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes()).await.unwrap();
        });

        let response = reqwest::get(format!("http://{addr}/test")).await.unwrap();
        let ctx = HttpErrorContext::from_response(response).await;

        assert_eq!(ctx.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(ctx.url.ends_with("/test"));
        assert_eq!(ctx.request_id.as_deref(), Some("req-123"));
        assert_eq!(ctx.error_code.as_deref(), Some("GatedRepo"));
        assert_eq!(ctx.server_message.as_deref(), Some("acceptance required"));
        assert!(ctx.body.contains("server said no"));
    }

    #[tokio::test]
    async fn from_response_falls_back_to_json_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
            let body = r#"{"error":"repo is gated"}"#;
            let resp = format!(
                "HTTP/1.1 403 Forbidden\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 \r\n{}",
                body.len(),
                body
            );
            tokio::io::AsyncWriteExt::write_all(&mut sock, resp.as_bytes()).await.unwrap();
        });

        let response = reqwest::get(format!("http://{addr}/")).await.unwrap();
        let ctx = HttpErrorContext::from_response(response).await;

        assert_eq!(ctx.request_id, None);
        assert_eq!(ctx.server_message.as_deref(), Some("repo is gated"));
    }
}
