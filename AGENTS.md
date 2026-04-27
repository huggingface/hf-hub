# hf-hub

Async Rust client library for the [Hugging Face Hub API](https://huggingface.co/docs/hub/api). This is the Rust equivalent of the Python [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library.

The primary entry point is the `HFClient` struct, which wraps an `Arc<HFClientInner>` for cheap cloning. All methods are async and use `reqwest` as the HTTP client. Paginated endpoints return `impl Stream<Item = Result<T>>` via `futures::stream::try_unfold`. Methods take parameters via [`bon`](https://docs.rs/bon) per-method builders finished with `.send().await` (mirroring reqwest / aws-sdk / octocrab style).

Key capabilities:

- Repository info, listing, creation, deletion, and settings updates
- File upload, download, listing, and deletion
- Commit creation, commit history, diffs between revisions
- Branch and tag management
- User and organization info
- Xet high-performance transfers

## Code Standards

These rules apply to ALL code written or modified in this repo:

### Style

- NO trivial comments — do not add comments that restate what the code does
- Descriptive variable and function names
- No wildcard imports (e.g., `use foo::*`), except `pub use` re-exports in `lib.rs`
- All imports are at the top of the file or top of module
- Latest stable Rust features are allowed

### Error Handling

- Use `Result<T, E>` with explicit error handling — never panic
- Define custom error types using `thiserror` for domain-specific errors
- Provide helpful, actionable error messages

### Performance

- Be mindful of allocations in hot paths
- Prefer structured logging (tracing/log macros with fields, not string formatting)

### Dependencies

- Add all dependencies to `Cargo.toml` (workspace root) or `hf-hub/Cargo.toml` (crate-level)
- Prefer well-maintained crates from crates.io
- Shared dependencies belong in the workspace `[dependencies]` table, not per-crate

### Testing

#### Unit Tests

- Place in the same file using `#[cfg(test)]` modules
- Run: `cargo test -p hf-hub`

#### Integration Tests

- Located in the `integration-tests` workspace crate, under `integration-tests/tests/` (`integration_test.rs`, `blocking_test.rs`, `bucket_sync_test.rs`, `bucket_xet_transfer_test.rs`, `cache_test.rs`, `download_test.rs`, `xet_transfer_test.rs`). Shared helpers live in `integration-tests/src/test_utils.rs`.
- Require a valid `HF_TOKEN` environment variable and internet access
- Tests skip gracefully when `HF_TOKEN` is not set (no failures)
- Run read-only tests: `HF_TOKEN=HF_xxx cargo test -p integration-tests`
- Write operation tests (create/delete repos, upload files) require `HF_TEST_WRITE=1`
- Run all tests including writes: `HF_TOKEN=HF_xxx HF_TEST_WRITE=1 cargo test -p integration-tests`

### Formatting and Linting

- Format: `cargo +nightly fmt`
- Lint: `cargo clippy -p hf-hub --all-features -- -D warnings`
- ALWAYS run both after making changes — do not skip this step

### Minimal Changes

- Verify that every change is minimal and necessary — do not include unrelated modifications

### Method builders (bon)

All public methods on `HFClient`, `HFRepository`, `HFSpace`, and `HFBucket` use `bon`'s
per-method `#[builder(finish_fn = send)]` pattern. Do NOT introduce `*Params` wrapper structs —
parameters live directly on the method. Every method (including parameterless ones like `info`,
`exists`, `pause`, `whoami`) is finished with `.send()` for uniformity.

```rust
use bon::bon;

#[bon]
impl HFRepository {
    #[builder(finish_fn = send)]
    pub async fn my_method(
        &self,
        #[builder(into)] name: String,
        revision: Option<String>,
    ) -> HFResult<MyOutput> { ... }
}
```

Translation rules from `typed-builder` attributes (now removed):

| typed-builder | bon (as method param) |
|---|---|
| `#[builder(setter(into))] x: String` | `#[builder(into)] x: String` |
| `#[builder(default, setter(strip_option))] x: Option<T>` | `x: Option<T>` (implicit) |
| `#[builder(default, setter(into, strip_option))] x: Option<String>` | `#[builder(into)] x: Option<String>` |
| `#[builder(default)] x: bool` | `#[builder(default)] x: bool` |
| `#[builder(default = expr)] x: T` | `#[builder(default = expr)] x: T` |
| required field, no attribute | `x: T` |

bon also generates a `maybe_<field>(opt)` setter alongside each `Option<T>` parameter — use it
when forwarding an existing `Option<T>` value (e.g. inside a sync mirror, or when porting a
helper that already has the field as an `Option<T>` local).

#### Mandatory blocking counterpart

Every async method MUST have a manually-written blocking counterpart on the corresponding
`*Sync` struct under `#[cfg(feature = "blocking")]`. Mirror the same `#[builder]` parameter list
verbatim. The body calls the inner async builder and blocks on the runtime:

```rust
#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFRepositorySync {
    #[builder(finish_fn = send)]
    pub fn my_method(
        &self,
        #[builder(into)] name: String,
        revision: Option<String>,
    ) -> HFResult<MyOutput> {
        self.runtime.block_on(
            self.inner
                .my_method()
                .name(name)
                .maybe_revision(revision)
                .send(),
        )
    }
}
```

For stream-returning methods, the sync counterpart collects into `Vec<T>`: pin the stream and
drain with `while let Some` (see existing examples in `repository/listing.rs` and
`repository/mod.rs`). On `HFSpaceSync`, the runtime is reached via `&self.repo_sync.runtime`
because the struct has no direct `runtime` field. Do NOT reintroduce the legacy `sync_api!` /
`sync_api_stream!` / `sync_api_async_stream!` macros — they were removed deliberately when the
crate moved to bon.

When parameters need a backing struct for internal helpers (e.g. `RepoDownloadFileParams` is
plumbed through several private functions in `repository/download.rs`), keep that struct
**private** to the module with no `#[derive(TypedBuilder)]` and have the bon method assemble it.
Never expose it publicly.

## Project Layout

> **Agents MUST update this section when adding new crates or large modules.**

Types and API methods live together per component. Each component is either a single file (for smaller components) or a folder (when the impl block needs splitting). Public types are reached via `hf_hub::<component>::…`; no `hf_hub::types` module exists.

```txt
hf-hub/
├── Cargo.toml                      # Workspace root
├── AGENTS.md                       # This file (CLAUDE.md is a symlink to it)
├── .gitignore
├── hf-hub/                         # Main library crate (package: hf-hub)
│   ├── Cargo.toml                  # Crate manifest, dependencies, features
│   ├── src/
│   │   ├── lib.rs                  # Module declarations, public re-exports, crate docs
│   │   ├── client.rs               # HFClient, HFClientBuilder, HFClientInner, auth headers, URL builders
│   │   ├── blocking.rs             # Synchronous *Sync handles (behind "blocking" feature)
│   │   ├── constants.rs            # Env var names, default URLs, repo type helpers
│   │   ├── error.rs                # HFError enum, HFResult alias, NotFoundContext
│   │   ├── pagination.rs           # Generic paginate<T>() with Link header parsing
│   │   ├── retry.rs                # Retry logic for transient HTTP failures
│   │   ├── progress.rs             # ProgressEvent, ProgressHandler, Upload/DownloadEvent, FileProgress
│   │   ├── repository/             # Repo component: all HFRepository-bound APIs live here,
│   │   │   │                       #   flat public surface via `hf_hub::repository::…`
│   │   │   ├── mod.rs              # HFRepository handle, RepoType, RepoInfo, ModelInfo/DatasetInfo/
│   │   │   │                       #   SpaceInfo, list/create/delete/move/update + flat re-exports
│   │   │   ├── commits.rs          # Git history types/params + list_commits/list_refs/diff/branches/tags
│   │   │   ├── diff.rs             # HFFileDiff, GitStatus, HFDiffParseError, parse_raw_diff, stream_raw_diff
│   │   │   ├── files.rs            # File types/params: BlobLfsInfo, RepoTreeEntry, FileMetadataInfo,
│   │   │   │                       #   CommitOperation/AddSource/CommitInfo, all file-op params
│   │   │   ├── listing.rs          # list_files, list_tree, get_paths_info, get_file_metadata
│   │   │   ├── download.rs         # download_file, download_file_stream, download_file_to_bytes,
│   │   │   │                       #   snapshot_download
│   │   │   └── upload.rs           # upload_file, upload_folder, create_commit, delete_file/folder
│   │   ├── spaces.rs               # Spaces component: HFSpace handle, SpaceRuntime/SpaceVariable,
│   │   │                           #   Space*Params, runtime/hardware/secrets/variables/duplicate
│   │   ├── users.rs                # Users component: User/Organization/OrgMembership, whoami,
│   │   │                           #   user+org lookup, followers/following
│   │   ├── xet.rs                  # Xet component (pub(crate)): XetConnectionInfo, xet transfer plumbing
│   │   │                           #   used by repositories and buckets
│   │   ├── buckets/
│   │   │   ├── mod.rs              # HFBucket handle, bucket types/params, create/list/tree/batch/download
│   │   │   └── sync.rs             # BucketSync* types, HFBucket::sync — plan computation and execution
│   │   └── cache/
│   │       ├── mod.rs              # CachedFileInfo/CachedRepoInfo/HFCacheInfo + scan_cache API
│   │       └── storage.rs          # pub(crate) on-disk plumbing: scan, locking, ref read/write, symlinks
│   └── (unit tests live next to their modules in #[cfg(test)] blocks)
├── integration-tests/              # Integration tests crate (package: integration-tests)
│   ├── Cargo.toml                  # Depends on hf-hub with "blocking" feature
│   ├── src/
│   │   ├── lib.rs                  # Re-exports test_utils
│   │   └── test_utils.rs           # Shared helpers: env-var names, token resolution, sha256_hex
│   └── tests/
│       ├── integration_test.rs        # Read-only integration tests against live Hub API
│       ├── blocking_test.rs           # Blocking wrapper parity tests
│       ├── bucket_sync_test.rs        # Bucket sync plan/execution tests
│       ├── bucket_xet_transfer_test.rs # Bucket xet upload/download tests
│       ├── cache_test.rs              # Cache scanning/locking tests
│       ├── download_test.rs           # Download path tests
│       └── xet_transfer_test.rs       # Xet transfer tests
├── examples/                       # Example programs crate (package: examples)
│   ├── Cargo.toml                  # Crate manifest; each example declared with explicit path
│   └── *.rs                        # Flat example source files (repo, files, commits, buckets, users,
│                                   #   spaces, progress, diff, download_upload, blocking_*, ...)
├── benches/                        # Benchmark crate (package: hf-hub-benches)
│   ├── Cargo.toml                  # Criterion benchmark setup
│   └── sync_api.rs                 # Download/info benches for the sync API
└── hfrs/                           # CLI crate (package: hfrs)
    ├── Cargo.toml                  # Crate manifest, binary dependencies
    ├── src/
    │   ├── main.rs                 # Binary entry point
    │   ├── cli.rs                  # Clap definitions, Command enum, OutputFormat
    │   ├── output.rs               # JSON/table rendering for command results
    │   ├── progress.rs             # CliProgressHandler for upload/download progress
    │   ├── util/                   # Token management helpers
    │   └── commands/               # Subcommand implementations (auth, buckets, cache,
    │                               #   datasets, download, env, models, repos, spaces,
    │                               #   upload, version)
    └── tests/
        ├── cli_comparison.rs       # CLI behavioral parity tests
        └── helpers.rs              # Test harness for invoking hfrs binary
```

## Feature Development

Before writing any code:

1. **Branch:** Confirm you are on a feature branch, not `main`. If on `main`, create a branch named `<username>/<short-description>`.
2. **Plan:** Write an implementation plan that includes testing strategy (unit tests, integration tests, manual verification steps). Add this plan as a comment on the PR.

When changing public interfaces or adding user-facing capabilities:

- ALWAYS update the relevant examples in `examples/` and any affected README snippets so they match the current public API.
- ALWAYS add at least one example for new functionality unless an existing example already demonstrates that exact workflow clearly.
- Prefer examples that show the intended high-level interface, not just the lowest-level parameter structs, especially for new ergonomic APIs like repo handles.

## Code Review

When reviewing a pull request, follow these rules:

### Tone

- Collegiate and constructive — write as a peer, not an authority
- Use phrases like "consider...", "what do you think about...", "we might want to..."
- Acknowledge good decisions and clean patterns, not just problems
- When unsure, ask a clarifying question instead of assuming something is wrong

### What to Review

- **Correctness** — logic errors, edge cases, off-by-one errors
- **Readability** — naming consistency, code clarity, helpful error messages
- **Maintainability** — temporary workarounds tracked, types in the right crate, clean abstractions
- **Testability** — missing tests for new endpoints/logic, weakened assertions, coverage gaps
- **Performance** — unnecessary allocations in hot paths, unbounded response sizes, missing concurrency limits
- **Security** — auth checks on new routes, input validation, error message information leakage

### How to Structure Feedback

- Post a summary comment on the PR: overview of the changes, key observations, cross-cutting concerns
- Add inline comments at specific diff locations for targeted feedback
- Prefix minor style suggestions with `nit:` — these are optional and the author may skip them
- Do NOT prefix substantive feedback (public API changes, correctness issues, missing tests) — these require attention
