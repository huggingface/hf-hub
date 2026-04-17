# Retry mechanism design

**Date:** 2026-04-17
**Status:** Approved for implementation

## Goal

Replace `reqwest_retry` + `reqwest_middleware` with a thin retry wrapper built on `tokio_retry`. Follow the pattern established in `xet-core`'s `RetryWrapper` (`../xet-core/xet_client/src/cas_client/retry_wrapper.rs`), minus the per-call special-case overrides (no `with_429_no_retry`, `with_retry_on_403`, `with_expected_416`). One logical way to decide retry vs. fatal, tuned once via `HFClientBuilder`.

## Motivation

- `reqwest_middleware` is pulled in solely to carry the retry middleware. Dropping it removes a layer of indirection and an error type (`HFError::Middleware`) that leaks into our public error surface.
- `reqwest_retry`'s retry policy is not easy to customize without middleware gymnastics. A hand-written loop over `tokio_retry` is small, clear, and under our control.
- `xet-core` already demonstrates the pattern works well against the Hub and its CDN.

## Non-goals

- Per-call-site retry policy overrides. If a future call site needs different behavior, that's a separate design.
- Retrying body-read failures (truncated JSON, dropped stream mid-read). `xet-core`'s `run_and_extract_json`/`run_and_extract_bytes` handle this; we defer. If download paths show regressions, we add targeted helpers then.
- `api_tag` parameters threaded through call sites. We rely on the `tracing` span context already established at the API layer; the retry wrapper logs `url` and `attempt` as structured fields.

## Design

### Module layout

New module `hf-hub/src/retry.rs`, mounted from `lib.rs`. All items `pub(crate)`.

```rust
pub(crate) struct RetryConfig {
    pub max_attempts: usize,    // default: 3
    pub base_delay: Duration,   // default: 100ms
}

fn is_transient_reqwest_error(err: &reqwest::Error) -> bool;
fn is_transient_status(status: StatusCode) -> bool;

pub(crate) async fn retry<F, Fut>(
    config: &RetryConfig,
    f: F,
) -> Result<Response, reqwest::Error>
where
    F: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = Result<Response, reqwest::Error>> + Send;
```

New method on `HFClient`:

```rust
impl HFClient {
    pub(crate) async fn retry<F, Fut>(&self, f: F) -> Result<Response, reqwest::Error>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = Result<Response, reqwest::Error>> + Send,
    {
        retry::retry(&self.inner.retry_config, f).await
    }
}
```

The retry loop uses `tokio_retry::RetryIf` with `ExponentialBackoff::from_millis(base_delay.as_millis()).map(jitter).take(max_attempts)` — same shape as xet-core's wrapper.

### Retry decision logic

Single classifier. No per-call overrides.

**Transport errors — `is_transient_reqwest_error(&reqwest::Error) -> bool`:**

- **Transient (retry):**
  - `err.is_timeout()`
  - `err.is_connect()`
  - `err.is_request()` AND the source chain contains a `hyper::Error` with `is_incomplete_message()`, `is_canceled()`, or wraps a `std::io::Error` (server dropped mid-response)
- **Fatal (do not retry):**
  - `err.is_body()`, `err.is_decode()`, `err.is_builder()`, `err.is_redirect()`
  - `err.is_status()` (we see the status through the `Ok(Response)` path instead)
  - Anything else we can't classify — fail closed

The hyper source walk uses a private `get_source_error_type::<hyper::Error>(..)` helper copied from xet-core (which itself copied it from `reqwest_middleware`).

**Status codes — `is_transient_status(StatusCode) -> bool`:**

```rust
matches!(status,
    StatusCode::REQUEST_TIMEOUT          // 408
    | StatusCode::TOO_MANY_REQUESTS      // 429
    | StatusCode::INTERNAL_SERVER_ERROR  // 500
    | StatusCode::BAD_GATEWAY            // 502
    | StatusCode::SERVICE_UNAVAILABLE    // 503
    | StatusCode::GATEWAY_TIMEOUT        // 504
)
```

Everything else is fatal, including 501 (server said "not implemented" — won't change on retry) and 425 (explicitly excluded).

**Retry loop flow:**

1. Call `f().await`.
2. `Err(e)`: if `is_transient_reqwest_error(&e)` → retry; else → return `Err(e)`.
3. `Ok(resp)`: if `is_transient_status(resp.status())` → retry (the response body from this attempt is discarded — the next attempt opens its own); else → return `Ok(resp)`. This includes fatal HTTP statuses like 404 — those come back as `Ok(Response)` and the caller's existing `check_response(..)` maps them to typed `HFError` variants.
4. When retries exhaust: return the last `Err(e)` or `Ok(resp)` as-is.

**Logging:**

- `debug!(attempt = N, url = %url, "retrying request")` before each retry
- `debug!(attempt = N, url = %url, status = %s, "request succeeded")` on terminal success
- `error!(url = %url, attempts = N, "retry exhausted")` on give-up

URL is taken from the `Response` when available; from `err.url()` on transport errors. When `err.url()` is `None` (builder-stage errors and similar edge cases — which are all classified fatal and never retried), logs show `<unknown url>`.

### Client changes

`HFClientInner` drops `ClientWithMiddleware` for raw `reqwest::Client`:

```rust
pub(crate) struct HFClientInner {
    pub(crate) client: reqwest::Client,
    pub(crate) no_redirect_client: reqwest::Client,
    pub(crate) retry_config: RetryConfig,
    // ... rest unchanged
}
```

`HFClientBuilder` gains two setters:

```rust
pub fn retry_max_attempts(mut self, n: usize) -> Self { ... }
pub fn retry_base_delay(mut self, d: Duration) -> Self { ... }
```

Defaults: `max_attempts = 3`, `base_delay = 100ms` (matches current `reqwest_retry` configuration).

`build()` drops the `reqwest_middleware::ClientBuilder` / `RetryTransientMiddleware` wiring. Both `client` and `no_redirect_client` become plain `reqwest::Client`; both flow through `HFClient::retry(..)` — no-redirect retries still happen.

**`HFClient` getter signatures:**

- `http_client()` → `&reqwest::Client` (was `&ClientWithMiddleware`)
- `no_redirect_client()` → `&reqwest::Client`

### `HFError` changes

- Remove the `Middleware(#[from] reqwest_middleware::Error)` variant.
- Simplify `is_transient()`: drop the `Middleware` arm. The `Request(e)` branch stays (it's the post-retry transport error path). The `Http { status, .. }` 5xx branch stays (cache fallback uses it).

### Cargo changes

Remove from `hf-hub/Cargo.toml`:

- `reqwest-retry`
- `reqwest-middleware`

Add:

- `tokio-retry`
- `hyper` (only for the source-type downcast in the transport classifier)

### Call-site migration pattern

Every direct `.send()` call wraps in `HFClient::retry(..)`. The closure captures URL/headers/body and returns the `send()` future.

**Before:**

```rust
let url = self.hf_client.api_url(Some(self.repo_type), &self.repo_id);
let response = self
    .hf_client
    .http_client()
    .get(&url)
    .headers(self.hf_client.auth_headers())
    .send()
    .await?;
self.hf_client
    .check_response(response, Some(&self.repo_id), NotFoundContext::Repo)
    .await?;
```

**After:**

```rust
let url = self.hf_client.api_url(Some(self.repo_type), &self.repo_id);
let headers = self.hf_client.auth_headers();
let response = self
    .hf_client
    .retry(|| self.hf_client.http_client().get(&url).headers(headers.clone()).send())
    .await?;
self.hf_client
    .check_response(response, Some(&self.repo_id), NotFoundContext::Repo)
    .await?;
```

**Pattern notes:**

- `HeaderMap: Clone` — cheap, cloned per attempt.
- `url: &str` — captured by reference; closure outlives only the `retry()` call.
- `http_client()` returns `&reqwest::Client`, which is `Clone` (Arc-internal) — `.get()` on `&Client` is fine.
- JSON bodies (`.json(&payload)`): `payload` must be `Clone`-movable into the closure. Existing request payloads are `Serialize + Clone` — no issue.
- Streaming upload bodies (file readers, multipart): the closure rebuilds the body each attempt, which is the correct behavior. A consumed `Body` can't be replayed.

**Files affected:**

- `hf-hub/src/client.rs` — builder + struct changes
- `hf-hub/src/error.rs` — `Middleware` variant removal, `is_transient` simplification
- `hf-hub/src/retry.rs` — new module
- `hf-hub/src/lib.rs` — mount the new module
- `hf-hub/src/pagination.rs` — one call site
- `hf-hub/src/xet.rs` — one call site
- `hf-hub/src/api/repo.rs`
- `hf-hub/src/api/files.rs`
- `hf-hub/src/api/commits.rs`
- `hf-hub/src/api/users.rs`
- `hf-hub/src/api/spaces.rs`
- `hf-hub/src/api/buckets/mod.rs`
- `hf-hub/Cargo.toml` — dependency swap

## Testing

Test the classification functions directly; rely on existing integration tests for end-to-end sanity.

**`is_transient_status`** — pure function, exhaustive table-style test covering all retryable statuses (408, 429, 500, 502, 503, 504) and representative non-retryable ones (200, 201, 301, 400, 401, 403, 404, 409, 501).

**`is_transient_reqwest_error`** — `reqwest::Error` has no public constructor, so we produce real values via the natural API:

```rust
#[tokio::test]
async fn connect_error_is_transient() {
    let err = reqwest::Client::new()
        .get("http://127.0.0.1:1")
        .send()
        .await
        .unwrap_err();
    assert!(err.is_connect());
    assert!(is_transient_reqwest_error(&err));
}

#[tokio::test]
async fn builder_error_is_fatal() {
    let err = reqwest::Client::new()
        .get("not-a-url")
        .send()
        .await
        .unwrap_err();
    assert!(err.is_builder());
    assert!(!is_transient_reqwest_error(&err));
}
```

**Not directly tested:** timeout errors, hyper incomplete-message detection, retry-loop glue. These require either a mock server or exotic setup. Coverage comes from:

- Code review on the hyper source-walk (copied almost verbatim from xet-core and `reqwest_middleware`).
- Existing integration tests in `tests/integration_test.rs` exercising retries against the real Hub.

## Verification checklist

Run before considering the change complete:

1. `cargo test -p hf-hub` — unit tests pass.
2. `cargo clippy -p hf-hub --all-features -- -D warnings` — clean.
3. `cargo +nightly fmt` — clean.
4. `HF_TOKEN=... cargo test -p hf-hub --test integration_test` — read-only integration tests still pass.
5. `cargo tree -p hf-hub | grep -E 'reqwest-retry|reqwest-middleware'` — no hits.
6. Workspace grep for `ClientWithMiddleware`, `reqwest_middleware`, `reqwest_retry` — zero hits.
