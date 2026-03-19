# Xet Upload & Download Support for hf-hub

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Add xet-aware file downloads and full upload support (with xet for large files) to the hf-hub Rust crate's tokio async API.

## Context

hf-hub is a download-only Rust crate for interacting with Hugging Face Hub. It supports sync (ureq) and async (tokio/reqwest) APIs with local caching, resumable chunked downloads, and file locking.

The xet-core monorepo publishes two relevant crates:
- `hf-xet` (lib name `xet`, path `xet_pkg/`) — pure Rust library providing `XetSession`, `UploadCommit`, `DownloadGroup` for content-addressed, deduplicated file uploads and downloads against a CAS server.
- `xet-client` (path `xet_client/`) — lower-level client library, exports the `TokenRefresher` trait needed for auth integration.

Note: there is also an `hf_xet` PyO3 crate in the same monorepo — that is the Python binding and is **not** used here.

This design adds two capabilities to hf-hub's tokio API:
1. **Xet-aware downloads** — detect xet-stored files via HEAD response headers and download via `XetSession` instead of HTTP.
2. **File uploads** — `upload_file`, `upload_folder`, `create_commit` with server-negotiated routing: small files via HTTP, large files via `XetSession`.

## Dependencies & Feature Gate

### Cargo.toml additions

```toml
[dependencies]
hf-xet = { git = "https://github.com/huggingface/xet-core", rev = "c0f798061658259275557df6e124f64973fdcf85", optional = true }
xet-client = { git = "https://github.com/huggingface/xet-core", rev = "c0f798061658259275557df6e124f64973fdcf85", optional = true }
[features]
xet = ["dep:hf-xet", "dep:xet-client", "tokio"]
```

- `xet` feature is **off by default**. It implies `tokio`.
- `xet-client` is needed for the `TokenRefresher` trait.

## File Structure

```
src/
  lib.rs          — Cache, Repo, RepoType (unchanged)
  api/
    mod.rs        — Progress, RepoInfo, Siblings, shared types (unchanged, plus new module declarations)
    sync.rs       — sync API (unchanged)
    tokio.rs      — Api, ApiBuilder, ApiRepo (extended with upload/commit methods + xet download branch)
    commit.rs     — CommitOperation types, create_commit HTTP request, preupload API
    lfs.rs        — UploadInfo (sha256/size), LFS batch negotiation
    xet.rs        — XetAuth, XetSession wrapper, xet download/upload helpers
```

### Module visibility

- `commit.rs`, `lfs.rs` — `pub(crate)` types and functions, consumed by `tokio.rs`.
- `xet.rs` — `pub(crate)`, consumed by `tokio.rs` for both download and upload paths.
- `tokio.rs` — the only public API surface.

### mod.rs additions

```rust
#[cfg(feature = "tokio")]
mod commit;
#[cfg(feature = "tokio")]
mod lfs;
#[cfg(feature = "xet")]
mod xet;
```

## Commit Operations & Types (`commit.rs`)

### Core types

`CommitOperation` and its variants are **public** since they appear in `CreateCommitParams`.

```rust
/// What to do in a commit.
pub enum CommitOperation {
    Add(CommitOperationAdd),
    Delete(CommitOperationDelete),
    Copy(CommitOperationCopy),
}

pub struct CommitOperationAdd {
    pub path_in_repo: String,
    pub local_path: PathBuf,
    // Internal fields set during the upload pipeline
    pub(crate) upload_info: Option<UploadInfo>,
    pub(crate) upload_mode: Option<UploadMode>,
    pub(crate) should_ignore: bool,
    pub(crate) remote_oid: Option<String>,
}

pub struct CommitOperationDelete {
    pub path_in_repo: String,
}

pub struct CommitOperationCopy {
    pub src_path_in_repo: String,
    pub dest_path_in_repo: String,
    pub src_revision: Option<String>,
}

pub(crate) enum UploadMode {
    Regular,
    Lfs,
}
```

**Note on `should_ignore`**: The preupload response includes a `shouldIgnore` flag per file, indicating the file hasn't changed. Files with `should_ignore: true` are skipped during upload to avoid unnecessary transfers. The `remote_oid` field carries the server-provided OID for deduplication.

### Preupload check

```rust
/// POST /api/{repo_type}s/{repo_id}/preupload/{revision}
///
/// Sends file paths, sizes, and SHA256 samples in chunks of 256.
/// Server returns "regular" or "lfs" per file.
/// Mutates each CommitOperationAdd with its upload_mode.
pub(crate) async fn fetch_upload_modes(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    additions: &mut [CommitOperationAdd],
    create_pr: bool,
) -> Result<(), ApiError>
```

Request payload per chunk:
```json
{
  "files": [
    { "path": "model.safetensors", "size": 1234567, "sample": "<base64 first 512 bytes>" }
  ]
}
```

Response:
```json
{
  "files": [
    { "path": "model.safetensors", "uploadMode": "lfs", "shouldIgnore": false, "oid": null }
  ]
}
```

### Create commit request

```rust
/// POST /api/{repo_type}s/{repo_id}/commit/{revision}
///
/// Sends the final commit payload with:
/// - regular files inline (base64 content)
/// - LFS files as pointer metadata (after xet upload)
/// - deletions and copies as operations
pub(crate) async fn create_commit_request(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    commit_message: &str,
    operations: Vec<CommitOperation>,
    create_pr: bool,
) -> Result<CommitResponse, ApiError>

pub(crate) struct CommitResponse {
    pub commit_id: String,
    pub commit_url: String,
}
```

## LFS Batch Negotiation (`lfs.rs`)

### UploadInfo

```rust
pub(crate) struct UploadInfo {
    pub sha256: [u8; 32],
    pub size: u64,
    /// First 512 bytes for content-type detection by the server.
    pub sample: Vec<u8>,
}

impl UploadInfo {
    pub async fn from_path(path: &Path) -> Result<UploadInfo, ApiError>
}
```

### LFS batch endpoint

```rust
/// POST /{repo_url}.git/info/lfs/objects/batch
///
/// Request:
/// {
///   "operation": "upload",
///   "transfers": ["basic", "multipart", "xet"],
///   "objects": [{ "oid": hex(sha256), "size": u64 }],
///   "hash_algo": "sha256"
/// }
///
/// Response includes per-object actions and a top-level "transfer" field.
pub(crate) async fn post_lfs_batch_info(
    client: &Client,
    endpoint: &str,
    repo_url: &str,
    headers: &HeaderMap,
    upload_infos: &[UploadInfo],
) -> Result<LfsBatchResponse, ApiError>

pub(crate) struct LfsBatchResponse {
    pub transfer: String,         // "xet", "basic", or "multipart"
    pub objects: Vec<LfsObject>,
}

pub(crate) struct LfsObject {
    pub oid: String,
    pub size: u64,
}
```

We always offer `["basic", "multipart", "xet"]` in the transfers list. If the server doesn't choose `"xet"`, we return `ApiError::UnsupportedTransfer`. Actual LFS upload (basic/multipart) is not implemented in this iteration — see Known Limitations.

## Xet Integration (`xet.rs`)

Entire file behind `#[cfg(feature = "xet")]`.

### Auth (following latest spec — JSON response)

```rust
pub(crate) struct XetConnectionInfo {
    pub access_token: String,
    pub expiration_unix_epoch: u64,
    pub cas_url: String,
}

pub(crate) enum XetTokenType {
    Read,
    Write,
}

/// GET /api/{repo_type}s/{repo_id}/xet-{token_type}-token/{revision}
///
/// Returns JSON: { "accessToken": str, "exp": u64, "casUrl": str }
pub(crate) async fn fetch_xet_token(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    token_type: XetTokenType,
) -> Result<XetConnectionInfo, ApiError>
```

### Token refresher

Implements `xet_client::cas_client::auth::TokenRefresher` so `XetSession` can refresh tokens autonomously during long transfers.

The trait requires `#[async_trait]` and returns `Result<TokenInfo, AuthError>` where `TokenInfo = (String, u64)`.

```rust
use xet_client::cas_client::auth::{AuthError, TokenInfo, TokenRefresher};

struct HfTokenRefresher {
    client: Client,
    endpoint: String,
    repo_type: String,
    repo_id: String,
    revision: String,
    headers: HeaderMap,
    token_type: XetTokenType,
}

#[async_trait::async_trait]
impl TokenRefresher for HfTokenRefresher {
    async fn refresh(&self) -> Result<TokenInfo, AuthError> {
        let info = fetch_xet_token(
            &self.client, &self.endpoint, &self.repo_type,
            &self.repo_id, &self.revision, &self.headers, self.token_type,
        ).await.map_err(|e| AuthError::TokenRefreshFailure(e.to_string()))?;
        Ok((info.access_token, info.expiration_unix_epoch))
    }
}
```

### Xet file detection (for downloads)

When the Hub serves a file stored in xet, the HEAD response on the resolve URL includes an `X-Xet-Hash` header containing the xet content hash. We only need the hash — token acquisition uses `fetch_xet_token` with repo info directly (not a refresh route).

```rust
/// Parsed from HEAD response headers on the resolve URL.
pub(crate) struct XetFileData {
    pub file_hash: String,
}

/// Checks for X-Xet-Hash header in the response.
/// Returns None if not a xet file.
pub(crate) fn parse_xet_file_data(headers: &HeaderMap) -> Option<XetFileData>
```

### Session lifecycle

A single `XetSession` is stored on the `Api` struct as `Option<XetSession>`. Since `XetSession` is cheaply cloneable (all clones share state via `Arc`), it is cloned into each `ApiRepo`.

The session is initialized on the first xet operation. Once set, it is reused for all subsequent operations across all `ApiRepo` instances from the same `Api`. Initialization requires a CAS endpoint and initial token, obtained via `fetch_xet_token`. The session's `TokenRefresher` handles subsequent token refreshes.

**Initialization** (in `xet.rs`):
```rust
/// Create a new XetSession.
/// Fetches initial token, builds session with token refresher.
pub(crate) async fn create_session(
    client: &Client,
    endpoint: &str,
    repo_type: &str,
    repo_id: &str,
    revision: &str,
    headers: &HeaderMap,
    token_type: XetTokenType,
) -> Result<XetSession, ApiError>
```

### Download helper

```rust
/// Downloads a single xet file to the given path.
/// Uses the shared XetSession from Api.
/// 1. Creates DownloadGroup from session
/// 2. Queues file, calls finish()
pub(crate) async fn xet_download(
    session: &XetSession,
    file_data: &XetFileData,
    file_size: u64,
    dest_path: &Path,
) -> Result<(), ApiError>
```

### Upload helper

```rust
/// Uploads multiple files via the shared XetSession.
/// 1. Creates UploadCommit from session
/// 2. Queues all files, calls commit()
/// 3. Returns per-file xet hashes for the commit payload
pub(crate) async fn xet_upload(
    session: &XetSession,
    files: &[CommitOperationAdd],
) -> Result<Vec<XetUploadResult>, ApiError>

pub(crate) struct XetUploadResult {
    pub path_in_repo: String,
    pub xet_hash: String,
    pub file_size: u64,
    pub sha256: Option<String>,
}
```

The `XetSession` is long-lived and reused across all operations. Each upload creates an `UploadCommit` and each download creates a `DownloadGroup` from the same session — these are the short-lived, per-operation objects.

## Download Flow Changes (`tokio.rs`)

### Extended Metadata

The `size` field is changed to `u64` to correctly handle files >4GB on 32-bit platforms. A new xet field is added behind the `xet` feature gate.

```rust
pub struct Metadata {
    commit_hash: String,
    etag: String,
    size: u64,
    #[cfg(feature = "xet")]
    xet_file_data: Option<XetFileData>,
}
```

Accessor methods (`commit_hash()`, `etag()`, `size()`) are updated accordingly. The `size` type change from `usize` to `u64` is a breaking change.

The `metadata()` method is updated to parse xet headers from the HEAD response (before following redirects).

### Changes to `Api` and `ApiRepo`

```rust
#[derive(Clone, Debug)]
pub struct Api {
    endpoint: String,
    cache: Cache,
    client: Client,
    relative_redirect_client: Client,
    max_files: usize,
    chunk_size: Option<usize>,
    parallel_failures: usize,
    max_retries: usize,
    progress: bool,
    #[cfg(feature = "xet")]
    xet_session: Arc<RwLock<Option<XetSession>>>,  // written once, read many
}

pub struct ApiRepo {
    api: Api,
    repo: Repo,
}
```

`Api` exposes a `xet_session()` method using double-checked locking via `RwLock`:

```rust
#[cfg(feature = "xet")]
impl Api {
    /// Returns a cloned XetSession, initializing it on first call.
    /// Uses double-checked locking: acquires read lock first, falls back
    /// to write lock only if session is uninitialized.
    pub(crate) async fn xet_session(
        &self,
        repo_type: &str,
        repo_id: &str,
        revision: &str,
    ) -> Result<XetSession, ApiError> {
        // Fast path: read lock, return clone if already initialized
        {
            let guard = self.xet_session.read().await;
            if let Some(ref session) = *guard {
                return Ok(session.clone());
            }
        }
        // Slow path: write lock, double-check, then create
        {
            let mut guard = self.xet_session.write().await;
            // Another task may have initialized while we waited for the write lock
            if let Some(ref session) = *guard {
                return Ok(session.clone());
            }
            let session = xet::create_session(
                &self.client,
                &self.endpoint,
                repo_type,
                repo_id,
                revision,
                /* headers */
            ).await?;
            *guard = Some(session.clone());
            Ok(session)
        }
    }
}
```

`ApiRepo` calls `self.api.xet_session(...)` to get a session clone. Since `XetSession` is cheaply cloneable (Arc-based internally), the returned clone shares all state with the original.

### Download branching in `download_with_progress`

```rust
pub async fn download_with_progress<P: Progress + Clone + Send + Sync + 'static>(
    &self, filename: &str, progress: P,
) -> Result<PathBuf, ApiError> {
    let url = self.url(filename);
    let metadata = self.api.metadata(&url).await?;
    let cache = self.api.cache.repo(self.repo.clone());

    let blob_path = cache.blob_path(&metadata.etag);
    std::fs::create_dir_all(blob_path.parent().unwrap())?;
    let lock = lock_file(blob_path.clone()).await?;

    #[cfg(feature = "xet")]
    if let Some(ref xet_data) = metadata.xet_file_data {
        let session = self.api.xet_session(/* repo_type, repo_id, revision */).await?;
        xet::xet_download(&session, xet_data, metadata.size, &blob_path).await?;
    } else {
        self.http_download(&url, &metadata, &blob_path, progress).await?;
    }

    #[cfg(not(feature = "xet"))]
    self.http_download(&url, &metadata, &blob_path, progress).await?;

    drop(lock);

    // symlink + ref creation (unchanged)
    let mut pointer_path = cache.pointer_path(&metadata.commit_hash);
    pointer_path.push(filename);
    std::fs::create_dir_all(pointer_path.parent().unwrap()).ok();
    symlink_or_rename(&blob_path, &pointer_path)?;
    cache.create_ref(&metadata.commit_hash)?;
    Ok(pointer_path)
}
```

Existing HTTP download logic is extracted into a private `http_download` helper method.

## Upload Flow (`tokio.rs` public API)

### Params structs

All params structs have public fields and implement `Default` for optional fields.

```rust
#[derive(Default)]
pub struct UploadFileParams<'a> {
    pub local_path: &'a Path,
    pub path_in_repo: &'a str,
    pub commit_message: &'a str,
    pub commit_description: Option<&'a str>,
    pub parent_commit: Option<&'a str>,
    pub create_pr: bool,
}

#[derive(Default)]
pub struct UploadFolderParams<'a> {
    pub local_folder: &'a Path,
    pub path_in_repo: &'a str,
    pub commit_message: &'a str,
    pub commit_description: Option<&'a str>,
    pub parent_commit: Option<&'a str>,
    pub create_pr: bool,
}

#[derive(Default)]
pub struct CreateCommitParams<'a> {
    pub operations: Vec<CommitOperation>,
    pub commit_message: &'a str,
    pub commit_description: Option<&'a str>,
    pub parent_commit: Option<&'a str>,
    pub create_pr: bool,
}
```

- `commit_description`: Optional extended description (separate from the short message).
- `parent_commit`: Optional commit SHA for optimistic locking — the server rejects the commit if HEAD has moved.

### Public methods

```rust
impl ApiRepo {
    pub async fn upload_file(&self, params: UploadFileParams<'_>) -> Result<CommitResponse, ApiError>
    pub async fn upload_folder(&self, params: UploadFolderParams<'_>) -> Result<CommitResponse, ApiError>
    pub async fn create_commit(&self, params: CreateCommitParams<'_>) -> Result<CommitResponse, ApiError>
}
```

### `create_commit` orchestration

```
create_commit(params)
  │
  ├── 1. Compute UploadInfo (sha256, size, sample) for each Add
  │
  ├── 2. fetch_upload_modes() → server labels each Add as Regular or Lfs
  │
  ├── 3. Partition additions: regular_files, lfs_files
  │
  ├── 4. For lfs_files:
  │     ├── post_lfs_batch_info() → server confirms xet transfer
  │     ├── self.api.xet_session() → get/init shared session
  │     └── xet_upload(&session, lfs_files) → returns xet hashes per file
  │
  └── 5. create_commit_request()
        ├── regular files: inline base64 content
        ├── lfs files: xet pointer metadata (hash, size, sha256)
        ├── deletes: path references
        └── copies: src/dest path references
```

### Usage example

```rust
use hf_hub::api::tokio::{Api, UploadFileParams, CreateCommitParams, CommitOperation};

let api = Api::new()?;
let repo = api.model("user/my-model".into());

// Simple file upload
repo.upload_file(UploadFileParams {
    local_path: Path::new("model.safetensors"),
    path_in_repo: "model.safetensors",
    commit_message: "Upload model weights",
    ..Default::default()
}).await?;

// Complex commit with mixed operations
repo.create_commit(CreateCommitParams {
    operations: vec![
        CommitOperation::Add(CommitOperationAdd { ... }),
        CommitOperation::Delete(CommitOperationDelete { path_in_repo: "old_model.bin".into() }),
        CommitOperation::Copy(CommitOperationCopy { ... }),
    ],
    commit_message: "Reorganize model files",
    create_pr: true,
    ..Default::default()
}).await?;
```

## Error Handling

Extended `ApiError` enum:

```rust
#[derive(Debug, Error)]
pub enum ApiError {
    // --- existing variants (unchanged) ---
    MissingHeader(HeaderName),
    InvalidHeader(HeaderName),
    InvalidHeaderValue(InvalidHeaderValue),
    ToStr(ToStrError),
    RequestError(ReqwestError),
    ParseIntError(ParseIntError),
    IoError(std::io::Error),
    TooManyRetries(Box<ApiError>),
    TryAcquireError(TryAcquireError),
    AcquireError(AcquireError),
    Join(JoinError),
    LockAcquisition(PathBuf),

    // --- new variants ---

    /// Server returned an error in a JSON response body.
    #[error("API error ({status}): {message}")]
    HubApiError { status: u16, message: String },

    /// Preupload or commit response was malformed.
    #[error("Invalid API response: {0}")]
    InvalidApiResponse(String),

    /// Serialization/deserialization failure.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Xet session or transfer error.
    #[cfg(feature = "xet")]
    #[error("Xet error: {0}")]
    XetError(#[from] xet::XetError),

    /// Server chose a transfer protocol we can't handle.
    #[error("Unsupported transfer protocol: {0}")]
    UnsupportedTransfer(String),
}
```

## Testing Strategy

### Unit tests
- `commit.rs` — serialization of commit operations, preupload request/response parsing, commit request payload construction (mock JSON)
- `lfs.rs` — `UploadInfo::from_path` SHA256 computation, LFS batch request/response parsing
- `xet.rs` — `parse_xet_file_data` header extraction, `fetch_xet_token` JSON response parsing

### Integration tests (gated behind `#[ignore]` or env var)
- Download a known xet-stored file from a public repo, verify SHA256
- Upload a file via `upload_file`, download it back, verify round-trip integrity
- `create_commit` with mixed operations (add + delete + copy), verify commit on Hub
- Token refresh: upload large enough file that initial token expires mid-transfer

### Feature flag testing
- `cargo test` (default features, no xet) — existing download tests pass, no xet code compiled
- `cargo test --features xet` — full suite including xet paths

## Known Limitations

1. **No LFS upload fallback**: If the LFS batch server chooses `basic` or `multipart` instead of `xet`, the upload fails with `UnsupportedTransfer`. This means repos without xet enabled cannot use the upload API for large files. Basic LFS upload can be added in a future iteration.

2. **Path-only uploads**: `CommitOperationAdd` only accepts `PathBuf` (file on disk). In-memory `Vec<u8>` / streaming uploads are not supported in this iteration. The type can be extended to an enum (`Source::Path(PathBuf)` / `Source::Bytes(Vec<u8>)`) later.

3. **Xet download progress**: The xet download path does not integrate with hf-hub's `Progress` trait. `XetSession` has its own internal progress tracking, but it is not surfaced through the existing `download_with_progress` callback. This can be bridged in a future iteration by implementing `TrackingProgressUpdater` as an adapter.

4. **No retry logic for xet operations**: Unlike the HTTP download path (which has `max_retries` and exponential backoff), xet upload/download failures are not retried at the hf-hub level. The xet-core library has its own internal retry mechanisms.

5. **Auth format**: The xet token endpoint (`/api/{repo_type}s/{repo_id}/xet-{token_type}-token/{revision}`) returns a JSON response body per the latest spec at https://huggingface.co/docs/xet/auth. The Python `huggingface_hub` library currently reads from HTTP response headers (`X-Xet-*`) — this is a legacy format. Our implementation follows the documented JSON spec.
