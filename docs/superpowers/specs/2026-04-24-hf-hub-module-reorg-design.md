# hf-hub module reorganization — design

**Date:** 2026-04-24
**Author:** Assaf Vayner
**Status:** Draft

## Goal

Collapse the current `src/types/` and `src/api/` directories into per-component modules so that every component's data types, parameter structs, and `impl` blocks live together in one place. Reshape the public surface so callers import from component modules instead of a monolithic `hf_hub::types` namespace.

## Motivation

Today a "component" like buckets or commits is scattered across at least two files:

- `src/types/<component>.rs` — response types and optional params.
- `src/types/params.rs` / `src/types/repo_params.rs` — mixed params for several components in one place.
- `src/api/<component>.rs` — the `impl` blocks that consume those types.

To follow a single feature end-to-end (e.g. "how does `list_commits` work?") a reader jumps between three files. New types are also easy to misplace because the split between `params.rs` and `repo_params.rs` is fuzzy, and `types/commit.rs` mixes "commit-creation" types (`CommitOperation`, `CommitInfo`) with "commit-history" types (`GitCommitInfo`, `GitRefs`).

Co-locating each component in a single module (one file when small, one folder when large) gives a clean "one home per concept" invariant and matches the pattern already adopted for `buckets/`.

## Non-goals

- No changes to runtime behavior. The reorganization is structural only; every public method keeps its current signature and semantics.
- No changes to the blocking wrappers, `retry`, `pagination`, `constants`, `macros`, or `error` modules beyond import-path updates.
- No renames of public types, methods, or params beyond the import paths they live under.
- No introduction of new abstractions, traits, or helpers.

## Final directory layout

```
hf-hub/src/
├── lib.rs                     # module decls + public re-exports
├── macros.rs                  # (internal) unchanged
├── constants.rs               # (internal) unchanged
├── pagination.rs              # (internal) unchanged
├── retry.rs                   # (internal) unchanged
├── error.rs                   # public: HFError, HFResult, NotFoundContext
├── client.rs                  # public: HFClient, HFClientBuilder (HFClientInner stays pub(crate))
├── test_utils.rs              # public, unchanged
│
├── repo.rs                    # repo component + HFRepository / HFRepo handles
├── commits/
│   ├── mod.rs                 # commit-history types, refs, branch/tag, + impl HFRepository
│   └── diff.rs                # HFFileDiff, GitStatus, raw diff parsing (from src/diff.rs)
├── files/
│   ├── mod.rs                 # all file-related types and params
│   ├── listing.rs             # list_files, list_tree, get_paths_info, get_file_metadata
│   ├── download.rs            # download_file, download_file_stream, download_file_to_bytes, snapshot_download
│   └── upload.rs              # upload_file, upload_folder, create_commit, delete_file, delete_folder
├── buckets/
│   ├── mod.rs                 # HFBucket handle + bucket types/params + impls
│   └── sync.rs                # existing sync plan + execution (unchanged split)
├── spaces.rs                  # spaces component + HFSpace handle
├── users.rs                   # users component
├── cache/
│   ├── mod.rs                 # CachedFileInfo, HFCacheInfo, … + impl HFClient::scan_cache
│   └── storage.rs             # pub(crate) on-disk plumbing (from src/cache.rs)
├── xet.rs                     # XetTokenType, GetXetTokenParams, impl HFClient::get_xet_token, transfer plumbing
├── progress.rs                # Progress event model (from src/types/progress.rs)
│
└── blocking/                  # unchanged
    └── mod.rs
```

### Files removed

`src/types/` (the entire directory), `src/api/` (the entire directory), `src/bucket.rs`, `src/repository.rs`, `src/diff.rs`, and `src/cache.rs` are all removed. Their contents are redistributed into the layout above.

## Component contents

Each subsection lists exactly which items move where. All moves are verbatim — no type renames, no behavior changes.

### `repo.rs`

**From `src/repository.rs`:** `HFRepository` struct, `HFRepo` type alias, `HFRepository::{new, client, owner, name, repo_path, repo_type, info}`, `fmt::Debug for HFRepository`, and the `HFClient::{repo, model, dataset}` factory methods. The associated test cases for `HFRepository` move with it.

**From `src/types/repo.rs`:** `RepoType`, `RepoInfo`, `ModelInfo`, `DatasetInfo`, `SpaceInfo`, `RepoSibling`, `RepoUrl`, `GatedApprovalMode`, `GatedNotificationsMode`.

**From `src/types/params.rs`:** `ListModelsParams`, `ListDatasetsParams`, `ListSpacesParams`, `CreateRepoParams`, `DeleteRepoParams`, `MoveRepoParams`.

**From `src/types/repo_params.rs`:** `RepoInfoParams`, `RepoRevisionExistsParams`, `RepoFileExistsParams`, `RepoUpdateSettingsParams`.

**From `src/api/repo.rs`:** the whole file — both the `impl HFRepository { info, revision_exists, file_exists, exists, update_settings }` block and the `impl HFClient { list_models, list_datasets, list_spaces, create_repo, delete_repo, move_repo }` block. Note: `get_xet_token` lives in `api/repo.rs` today but moves to `xet.rs` instead.

### `commits/mod.rs`

**From `src/types/commit.rs`:** `GitCommitInfo`, `CommitAuthor`, `GitRefs`, `GitRefInfo`, `DiffEntry`. (`CommitInfo`, `CommitOperation`, `AddSource` move to `files/mod.rs` instead — see "borderline types".)

**From `src/types/repo_params.rs`:** `RepoListCommitsParams`, `RepoListRefsParams`, `RepoGetCommitDiffParams`, `RepoGetRawDiffParams`, `RepoCreateBranchParams`, `RepoDeleteBranchParams`, `RepoCreateTagParams`, `RepoDeleteTagParams`.

**From `src/api/commits.rs`:** the whole file — `impl HFRepository { list_commits, list_refs, get_commit_diff, get_raw_diff, create_branch, delete_branch, create_tag, delete_tag }`.

### `commits/diff.rs`

**From `src/diff.rs`:** the whole file — `HFFileDiff`, `GitStatus`, `parse_raw_diff`, `stream_raw_diff`. Public: `pub` items remain `pub`, re-exported from `commits/mod.rs` so existing call sites keep working by module path.

### `files/mod.rs`

Types and params only — the impls live in the sub-files.

**From `src/types/repo.rs`:** `BlobLfsInfo`, `LastCommitInfo`, `RepoTreeEntry`, `FileMetadataInfo`.

**From `src/types/commit.rs`:** `CommitInfo`, `CommitOperation`, `AddSource`.

**From `src/types/repo_params.rs`:** `RepoListFilesParams`, `RepoListTreeParams`, `RepoGetFileMetadataParams`, `RepoGetPathsInfoParams`, `RepoDownloadFileParams`, `RepoDownloadFileStreamParams` (plus the `RepoDownloadFileToBytesParams` and `RepoDownloadFileToBytesParamsBuilder` aliases), `RepoSnapshotDownloadParams`, `RepoUploadFileParams`, `RepoUploadFolderParams`, `RepoDeleteFileParams`, `RepoDeleteFolderParams`, `RepoCreateCommitParams`.

### `files/listing.rs`, `files/download.rs`, `files/upload.rs`

`src/api/files.rs` (1709 LOC) splits by verb family. The boundary is by method:

- `listing.rs` — `list_files`, `list_tree`, `get_paths_info`, `get_file_metadata`, plus any private helpers used only by listing.
- `download.rs` — `download_file`, `download_file_stream`, `download_file_to_bytes`, `snapshot_download`, plus download-specific helpers.
- `upload.rs` — `upload_file`, `upload_folder`, `create_commit`, `delete_file`, `delete_folder`, plus upload-specific helpers.

Helpers shared across two or more sub-files move up into `files/mod.rs` as `pub(super)` items.

### `buckets/mod.rs`

**From `src/bucket.rs`:** the `HFBucket` handle struct + associated factory/URL helpers — folded into `buckets/mod.rs` so the handle and its impls live together.

**From `src/types/buckets.rs`:** unchanged — all bucket types and params already co-located.

**From `src/api/buckets/mod.rs`:** unchanged — both `impl HFClient { create_bucket, list_buckets, delete_bucket, move_bucket, … }` and `impl HFBucket { info, list_tree, get_paths_info, batch, download_files, … }` stay here.

### `buckets/sync.rs`

**From `src/api/buckets/sync.rs`:** unchanged — sync types (`BucketSyncDirection`, `BucketSyncAction`, `BucketSyncOperation`, `BucketSyncPlan`, etc.) already live in `types/buckets.rs` alongside their params; `impl HFBucket { sync }` stays here. If any sync-specific types can be made private to `buckets::sync`, that is permitted but not required.

### `spaces.rs`

**From `src/repository.rs`:** `HFSpace` struct, `fmt::Debug for HFSpace`, `HFSpace::{new, repo}`, `Deref<Target = HFRepository> for HFSpace`, `TryFrom<HFRepository> for HFSpace`, `From<HFSpace> for Arc<HFRepository>`, and the `HFClient::space` factory method. The `HFSpace` test cases move with it. `HFRepository` is imported from `crate::repo` — one-way dependency.

**From `src/types/spaces.rs`:** `SpaceRuntime`, `SpaceVariable`.

**From `src/types/params.rs`:** `DuplicateSpaceParams`.

**From `src/types/repo_params.rs`:** `SpaceHardwareRequestParams`, `SpaceSleepTimeParams`, `SpaceSecretParams`, `SpaceSecretDeleteParams`, `SpaceVariableParams`, `SpaceVariableDeleteParams`.

**From `src/api/spaces.rs`:** the whole file — `impl HFSpace { runtime, request_hardware, set_sleep_time, add_secret, delete_secret, add_variable, delete_variable, duplicate, pause, restart, … }`.

The `#[cfg(feature = "spaces")]` gating (if any) moves with the module declaration in `lib.rs`, not the module's contents.

### `users.rs`

**From `src/types/user.rs`:** `User`, `OrgMembership`, `Organization`.

**From `src/api/users.rs`:** the whole file — `impl HFClient { whoami, auth_check, get_user_overview, get_organization, list_followers, list_following, … }`.

### `cache/mod.rs`

**From `src/types/cache.rs`:** `CachedFileInfo`, `CachedRevisionInfo`, `CachedRepoInfo`, `HFCacheInfo`, `DeleteCacheRevision`.

**From `src/api/cache.rs`:** `impl HFClient { scan_cache, delete_cache_revisions }`.

### `cache/storage.rs`

**From `src/cache.rs`:** the whole file — path computation, on-disk locking, ref read/write, symlink creation, scanning primitives. Everything is `pub(crate)`. Callers: `cache::scan_cache` (public entry) and `files::download` (cached writes).

### `xet.rs`

**From `src/xet.rs`:** the whole file — Xet transfer plumbing, most items remain `pub(crate)`.

**From `src/types/params.rs`:** `XetTokenType`, `GetXetTokenParams` (public).

**From `src/api/repo.rs`:** the `get_xet_token` method — extracted from the `impl HFClient` block and re-stated as `impl HFClient { pub async fn get_xet_token(…) }` inside `xet.rs`.

### `progress.rs`

**From `src/types/progress.rs`:** the whole file — `ProgressEvent`, `ProgressHandler`, `Progress`, `UploadEvent`, `DownloadEvent`, `FileProgress`, `FileStatus`, `AggregateProgress`, `EmitEvent`, and everything else in that module. Public surface unchanged.

## Public export strategy

`src/lib.rs`:

```rust
// Private infrastructure
mod client;
mod constants;
mod error;
mod pagination;
mod retry;
#[macro_use]
mod macros;

#[cfg(feature = "blocking")]
mod blocking;

// Component modules (public)
pub mod buckets;
pub mod cache;
pub mod commits;
pub mod files;
pub mod progress;
pub mod repo;
pub mod spaces;
pub mod users;
pub mod xet;

pub mod test_utils;

// Crate-root re-exports — the minimal set for ergonomics
pub use buckets::HFBucket;
pub use client::{HFClient, HFClientBuilder};
pub use error::{HFError, HFResult};
pub use progress::ProgressHandler;
pub use repo::{HFRepo, HFRepository, RepoType};
pub use spaces::HFSpace;

#[cfg(feature = "blocking")]
pub use blocking::{HFBucketSync, HFClientSync, HFRepoSync, HFRepositorySync, HFSpaceSync};

#[doc(hidden)]
pub use constants::{hf_home, resolve_cache_dir};
```

### No `hf_hub::types`

The `types` module is removed. All public types are reached through their component module:

| Today | After |
| --- | --- |
| `hf_hub::types::{RepoType, RepoInfo, RepoInfoParams}` | `hf_hub::RepoType` + `hf_hub::repo::{RepoInfo, RepoInfoParams}` |
| `hf_hub::types::{RepoDownloadFileParams, CommitOperation, AddSource}` | `hf_hub::files::{RepoDownloadFileParams, CommitOperation, AddSource}` |
| `hf_hub::types::{GitCommitInfo, GitRefs, RepoCreateBranchParams}` | `hf_hub::commits::{GitCommitInfo, GitRefs, RepoCreateBranchParams}` |
| `hf_hub::types::{BucketInfo, CreateBucketParams, ListBucketTreeParams}` | `hf_hub::buckets::{BucketInfo, CreateBucketParams, ListBucketTreeParams}` |
| `hf_hub::types::{SpaceRuntime, SpaceSecretParams}` | `hf_hub::spaces::{SpaceRuntime, SpaceSecretParams}` |
| `hf_hub::types::{User, Organization}` | `hf_hub::users::{User, Organization}` |
| `hf_hub::types::{DownloadEvent, ProgressEvent, FileStatus}` | `hf_hub::progress::{DownloadEvent, ProgressEvent, FileStatus}` |
| `hf_hub::types::{XetTokenType, GetXetTokenParams}` | `hf_hub::xet::{XetTokenType, GetXetTokenParams}` |
| `hf_hub::types::HFCacheInfo` | `hf_hub::cache::HFCacheInfo` |

`ProgressHandler` is also available unqualified as `hf_hub::ProgressHandler` for ergonomics; all other progress items require the module path.

## Borderline type placements (rationale)

These calls were made explicitly during design:

- **`CommitOperation`, `AddSource`, `CommitInfo` → `files/mod.rs`.** They're consumed by `create_commit`, `upload_file`, `upload_folder`, and `delete_file/folder` — all of which live in `files/upload.rs`. The `commits` module stays focused on git history and refs.
- **`RepoType` → `repo.rs`, re-exported at crate root.** Canonical home is the repo component; too common to require a module path at call sites.
- **`RepoUrl` → `repo.rs`.** Used by `create_repo` and `move_repo`. Buckets already defines a parallel `BucketUrl` in `buckets/`; the duplication is tolerable.

## In-tree consumer updates

Three consumers live in-tree and must be updated as part of the reorg:

1. **`hf-hub/src/lib.rs` doc comments.** Every doctest that does `use hf_hub::types::…` updates to the new module path. Every intra-doc link that points to `types::…` re-targets.
2. **`examples/*.rs`.** Mechanical `use` rewrites. Confirmed touched files: `repo.rs`, `repo_handles.rs`, `files.rs`, `commits.rs`, `diff.rs`, `spaces.rs`, `buckets.rs`, `download_upload.rs`, `progress.rs`, `progress_logger.rs`, `blocking_read.rs`, `blocking_write.rs`, `blocking_repo_handles.rs`, `blocking_spaces.rs`.
3. **`hfrs/` (the CLI crate).** Mechanical `use` rewrites in `cli.rs`, `progress.rs`, `util/mod.rs`, `commands/upload.rs`, `commands/buckets/cp.rs`, `commands/repos/branch.rs`, and any others that import from `hf_hub::types`. A grep over `hfrs/` during implementation will catch any we missed.

Integration tests under `hf-hub/tests/integration_test.rs` update their imports but do not change semantics.

## Tests

Unit tests embedded in `#[cfg(test)] mod tests { … }` move with their types (e.g. the `RepoType::from_str` tests stay at the bottom of `repo.rs`). No test is rewritten; only paths in `use super::…` or `use crate::…` lines change.

After the reorg:

- `cargo test -p hf-hub` passes (unit tests).
- `cargo test -p hf-hub --all-features` passes (including blocking).
- `HF_TOKEN=… cargo test -p hf-hub --test integration_test` passes on read-only tests (and with `HF_TEST_WRITE=1` on writes) — this exercises the public API.
- `cargo clippy -p hf-hub --all-features -- -D warnings` is clean.
- `cargo +nightly fmt` is clean.
- `cargo build --examples -p examples` and `cargo build -p hfrs` succeed with the rewritten imports.

## Migration approach

The reorg is invasive (touches most files in the crate) but shallow (no logic changes). It is executed as one atomic branch rather than an incremental series:

1. Physically move files and rename contents in one commit per component, in dependency order (leaves first, then crate-root re-exports last). This produces a branch where every intermediate commit compiles.
2. Update `src/lib.rs` module declarations and re-exports.
3. Update in-tree consumers (doc comments in `lib.rs`, `examples/`, `hfrs/`, `tests/`) in a final commit.
4. Run the test and lint suite listed above as the gate.

An alternative "temporary `types` re-export for one release cycle" approach was considered and rejected — the crate is pre-1.0 and all known consumers are in-tree, so a clean cut is preferable to dual maintenance.

## Out of scope / follow-ups

- Splitting `files/mod.rs` further if its types section grows beyond ~400 LOC after the move. Deferred until we see the post-reorg size.
- Making internal `cache::storage` items `pub(crate)` more aggressively where `files::download` doesn't need them. Minor cleanup after the structural move.
- A `prelude` module (e.g. `hf_hub::prelude::*`) for commonly-used imports. Not part of this reorg; can be added later if demand emerges.
