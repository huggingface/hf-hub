# hf-hub Module Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `src/types/` and `src/api/` into per-component modules so each component's types, params, and impls live together. Replace the `hf_hub::types` namespace with per-component public modules.

**Architecture:** Three-phase refactor designed for maximum parallelism. Phase 1 authors the new component modules at their final paths (or staging paths where they would collide) entirely in parallel — the orphan files aren't module-declared so the crate keeps compiling. Phase 2 is a single atomic "cutover" commit that flips `lib.rs`, deletes the old `types/` and `api/` directories, and finishes the handful of in-place renames. Phase 3 updates consumers and Phase 4 validates.

**Tech Stack:** Rust 1.x, `cargo`, `rustfmt` (nightly), `clippy`, existing `typed_builder`, `reqwest`, `serde`.

**Design spec:** `docs/superpowers/specs/2026-04-24-hf-hub-module-reorg-design.md` — consult for the detailed type-to-module mapping and rationale.

---

## Key invariants

- **The crate compiles at every commit boundary** — intermediate Phase 1 commits add orphan `.rs` files that are not declared in `lib.rs`, so cargo ignores them.
- **No behavior changes** — every public method keeps its signature and body; `use` paths inside moved code are rewritten to the new module layout but logic is verbatim.
- **Tests move with the types they cover** — `#[cfg(test)] mod tests { … }` blocks inside moved files travel with their parent.
- **Validation includes format and lint.** After every task that modifies Rust source, run:
  - `cargo +nightly fmt`
  - `cargo clippy -p hf-hub --all-features -- -D warnings`
  - `cargo build -p hf-hub --all-features`
  - `cargo test -p hf-hub` (unit tests)

A dedicated Phase 4 task runs the full validation including examples, `hfrs`, and integration tests.

---

## Target file structure

```
hf-hub/src/
├── lib.rs
├── macros.rs            (unchanged)
├── constants.rs         (unchanged)
├── pagination.rs        (unchanged)
├── retry.rs             (unchanged)
├── error.rs             (unchanged)
├── client.rs            (unchanged; HFClientInner stays pub(crate))
├── test_utils.rs        (unchanged)
├── repo.rs              NEW — absorbs parts of src/repository.rs + src/types/repo.rs + src/types/params.rs + src/types/repo_params.rs + src/api/repo.rs
├── commits/
│   ├── mod.rs           NEW — absorbs parts of src/types/commit.rs + src/types/repo_params.rs + src/api/commits.rs
│   └── diff.rs          NEW — absorbs src/diff.rs
├── files/
│   ├── mod.rs           NEW — types/params only
│   ├── listing.rs       NEW — list_files, list_tree, get_paths_info, get_file_metadata
│   ├── download.rs      NEW — download_file, download_file_stream, download_file_to_bytes, snapshot_download
│   └── upload.rs        NEW — upload_file, upload_folder, create_commit, delete_file, delete_folder
├── buckets/
│   ├── mod.rs           NEW — absorbs src/bucket.rs + src/types/buckets.rs + src/api/buckets/mod.rs (non-sync)
│   └── sync.rs          NEW — absorbs src/api/buckets/sync.rs + sync-related parts of src/types/buckets.rs
├── spaces.rs            NEW — absorbs HFSpace part of src/repository.rs + src/types/spaces.rs + parts of src/types/params.rs + src/types/repo_params.rs + src/api/spaces.rs
├── users.rs             NEW — absorbs src/types/user.rs + src/api/users.rs
├── cache/
│   ├── mod.rs           NEW — absorbs src/types/cache.rs + src/api/cache.rs
│   └── storage.rs       NEW — absorbs src/cache.rs (pub(crate) plumbing)
├── xet.rs               REWRITTEN — absorbs src/xet.rs + xet params from src/types/params.rs + get_xet_token from src/api/repo.rs
├── progress.rs          NEW — absorbs src/types/progress.rs
└── blocking/            (unchanged)
    └── mod.rs
```

Deleted after cutover: `src/types/` (all), `src/api/` (all), `src/repository.rs`, `src/bucket.rs`, `src/diff.rs`, `src/cache.rs` (moved into `cache/storage.rs`).

---

## Phase 1 — Author new component modules (PARALLEL)

All tasks in this phase run concurrently — each task writes one component's new files to disk, and the files are **not** yet declared in `lib.rs`. Cargo therefore ignores them and the crate compiles unchanged at each commit. Each task commits independently.

**Parallelism groups:**
- Tasks 1A, 1B, 1C, 1D, 1E, 1F, 1G all run in parallel. They write to disjoint new paths.
- Tasks 1H (xet) and 1I (cache) write to **staging** paths because their final paths collide with existing files — those collisions are resolved in Phase 2. They also run in parallel with the others.

For every task in this phase, apply these rules when copying content:
- **Copy content verbatim** from the source locations listed. Do not refactor; do not rename types, methods, or fields.
- **Rewrite `use` statements** in the new file so they reference the post-reorg module paths. Concrete translation table:
  | Old path | New path |
  |---|---|
  | `crate::types::RepoType` | `crate::repo::RepoType` |
  | `crate::types::{RepoInfo, RepoInfoParams, ...}` (repo items) | `crate::repo::{…}` |
  | `crate::types::{BlobLfsInfo, RepoTreeEntry, FileMetadataInfo, LastCommitInfo}` | `crate::files::{…}` |
  | `crate::types::{CommitOperation, AddSource, CommitInfo}` | `crate::files::{…}` |
  | `crate::types::{GitCommitInfo, GitRefs, GitRefInfo, CommitAuthor, DiffEntry}` | `crate::commits::{…}` |
  | `crate::types::{BucketInfo, …, BucketSync…}` | `crate::buckets::{…}` or `crate::buckets::sync::{…}` |
  | `crate::types::{SpaceRuntime, SpaceVariable, Space*Params, DuplicateSpaceParams}` | `crate::spaces::{…}` |
  | `crate::types::{User, Organization, OrgMembership}` | `crate::users::{…}` |
  | `crate::types::cache::{…}` and `crate::types::HFCacheInfo` | `crate::cache::{…}` |
  | `crate::types::progress::{…}` | `crate::progress::{…}` |
  | `crate::types::{XetTokenType, GetXetTokenParams}` | `crate::xet::{…}` |
  | `crate::repository::{HFRepository, HFRepo}` | `crate::repo::{HFRepository, HFRepo}` |
  | `crate::repository::HFSpace` | `crate::spaces::HFSpace` |
  | `crate::diff::{HFFileDiff, GitStatus, parse_raw_diff, stream_raw_diff}` | `crate::commits::diff::{…}` |
  | `crate::bucket::HFBucket` | `crate::buckets::HFBucket` |
- **Preserve `#[cfg(test)] mod tests { … }` blocks** at the bottom of the new file. The tests travel with their types.
- **Preserve the `sync_api! { … }` macro blocks** verbatim — they generate the blocking counterparts.

### Task 1A: Author `src/repo.rs`

**Files:**
- Create: `hf-hub/src/repo.rs`

**Content sources (all copied verbatim, then imports rewritten per the table above):**

1. **Handle struct and factories** from `src/repository.rs`:
   - `HFRepository` struct (incl. `#[derive(Clone)]`, all four fields).
   - `HFRepo` type alias.
   - `impl fmt::Debug for HFRepository`.
   - `impl HFClient { pub fn repo, pub fn model, pub fn dataset }` (three factory methods — NOT `pub fn space`).
   - `impl HFRepository { pub fn new, pub fn client, pub fn owner, pub fn name, pub fn repo_path, pub fn repo_type, pub async fn info }`.
   - The `#[cfg(test)] mod tests` block, keeping only `test_repo_path_and_accessors` (remove the `HFSpace` tests — those go to `spaces.rs`).

2. **Response and enum types** from `src/types/repo.rs`:
   - `RepoType` (enum + `Display` + `FromStr`) and its tests (`test_repo_type_from_str`, `test_repo_type_display`).
   - `RepoSibling`.
   - `ModelInfo`, `DatasetInfo`, `SpaceInfo`, `RepoInfo` (enum + `repo_type` method).
   - `RepoUrl`.
   - `GatedApprovalMode` (enum + `Serialize` + `FromStr`).
   - `GatedNotificationsMode`.
   - **Do NOT move:** `BlobLfsInfo`, `LastCommitInfo`, `RepoTreeEntry`, `FileMetadataInfo`, `test_repo_tree_entry_deserialize_*` — these go to `files/mod.rs`.

3. **List/manage params** from `src/types/params.rs`:
   - `ListModelsParams`, `ListDatasetsParams`, `ListSpacesParams`.
   - `CreateRepoParams`, `DeleteRepoParams`, `MoveRepoParams`.
   - **Do NOT move:** `XetTokenType`, `GetXetTokenParams`, `DuplicateSpaceParams`.

4. **Repo-scoped params** from `src/types/repo_params.rs`:
   - `RepoInfoParams`, `RepoRevisionExistsParams`, `RepoFileExistsParams`, `RepoUpdateSettingsParams`.

5. **API impls** from `src/api/repo.rs`:
   - The full `impl HFRepository { fetch_repo_info, model_info, dataset_info, space_info, exists, revision_exists, file_exists, update_settings }` block.
   - The full `impl HFClient { list_models, list_datasets, list_spaces, create_repo, delete_repo, move_repo }` block.
   - **Do NOT move:** `get_xet_token` — that goes to `xet.rs`.
   - The `sync_api! { … }` blocks for the listed HFClient and HFRepository methods (preserve verbatim, but drop `get_xet_token` from them — moves to xet).
   - The `#[cfg(test)] mod tests` block (the three `test_list_*_limit_zero_returns_empty` tests) at the bottom, preserved verbatim.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/repo.rs`.** Assemble the content per the sources above, with imports rewritten per the translation table. The file starts with the file-level doc comment, then all `use` statements grouped and sorted, then module items in this order: type aliases, enums, structs, `impl fmt::Debug`, `impl HFClient { /* factories */ }`, `impl HFRepository { /* accessors + info + exists/update */ }`, `impl HFClient { /* list/create/delete/move */ }`, `sync_api!` blocks, `#[cfg(test)] mod tests`.

- [ ] **Step 2: Validate the orphan file compiles standalone.** Because `lib.rs` does not yet declare `mod repo;`, cargo will not compile the new file. Confirm this is the case:
  ```bash
  cargo build -p hf-hub --all-features
  ```
  Expected: PASS (nothing changed from cargo's perspective).

- [ ] **Step 3: Run format and lint on the whole crate.** This touches only the new file via rustfmt file-discovery.
  ```bash
  cargo +nightly fmt -p hf-hub
  cargo clippy -p hf-hub --all-features -- -D warnings
  ```
  Expected: both pass. Clippy will not inspect `repo.rs` (not in module tree) but should stay green globally.

- [ ] **Step 4: Commit.**
  ```bash
  git add hf-hub/src/repo.rs
  git commit -m "refactor(hf-hub): add repo.rs component module (not yet wired)"
  ```

### Task 1B: Author `src/commits/mod.rs` and `src/commits/diff.rs`

**Files:**
- Create: `hf-hub/src/commits/mod.rs`
- Create: `hf-hub/src/commits/diff.rs`

**Content sources:**

1. **`src/commits/diff.rs`** — the full contents of `src/diff.rs` (369 LOC). Update its internal `use` statements per the translation table (it uses `crate::types::*` for a few items — rewrite). Keep all `pub` items `pub` so `commits/mod.rs` can re-export them.

2. **`src/commits/mod.rs`** — composed of:
   - **Module declaration and re-export for diff:**
     ```rust
     pub mod diff;
     pub use diff::{HFFileDiff, GitStatus};
     ```
     (Add any other `pub` symbols currently used externally; grep for `use crate::diff::` across the codebase before Phase 2.)
   - **History/refs types** from `src/types/commit.rs`:
     - `CommitAuthor`, `GitCommitInfo`, `GitRefInfo`, `GitRefs`, `DiffEntry`.
     - **Do NOT move:** `CommitInfo`, `CommitOperation`, `AddSource` — those go to `files/mod.rs`.
   - **Commit/branch/tag/ref params** from `src/types/repo_params.rs`:
     - `RepoListCommitsParams`, `RepoListRefsParams`.
     - `RepoGetCommitDiffParams`, `RepoGetRawDiffParams`.
     - `RepoCreateBranchParams`, `RepoDeleteBranchParams`.
     - `RepoCreateTagParams`, `RepoDeleteTagParams`.
   - **API impls** — full contents of `src/api/commits.rs`:
     - `impl HFRepository { list_commits, list_refs, get_commit_diff, get_raw_diff, get_raw_diff_stream, create_branch, delete_branch, create_tag, delete_tag }`.
     - The three `sync_api! { impl HFRepository -> HFRepositorySync { … } }` blocks verbatim.
     - Any inner `#[cfg(test)] mod tests` block verbatim.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/commits/diff.rs`** — copy all of `src/diff.rs` verbatim, then rewrite imports.
- [ ] **Step 2: Create `hf-hub/src/commits/mod.rs`** — assemble per the source list above.
- [ ] **Step 3: Validate compilation:** `cargo build -p hf-hub --all-features` (orphan files, should still pass).
- [ ] **Step 4: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 5: Commit:**
  ```bash
  git add hf-hub/src/commits/
  git commit -m "refactor(hf-hub): add commits/ component module (not yet wired)"
  ```

### Task 1C: Author `src/files/` (mod.rs + listing.rs + download.rs + upload.rs)

**Files:**
- Create: `hf-hub/src/files/mod.rs`
- Create: `hf-hub/src/files/listing.rs`
- Create: `hf-hub/src/files/download.rs`
- Create: `hf-hub/src/files/upload.rs`

**Content sources:**

1. **`src/files/mod.rs`** — types, params, sub-module declarations:
   - Sub-module declarations:
     ```rust
     mod download;
     mod listing;
     mod upload;
     ```
     (No `pub use` from the sub-modules — they only add `impl HFRepository` blocks, which are picked up automatically.)
   - **Response types** from `src/types/repo.rs`:
     - `BlobLfsInfo`, `LastCommitInfo`, `RepoTreeEntry`, `FileMetadataInfo`.
     - `RepoSibling` does **not** move — it stays in `repo.rs` because `ModelInfo`/`DatasetInfo`/`SpaceInfo` reference it.
     - Include the `test_repo_tree_entry_deserialize_file` and `test_repo_tree_entry_deserialize_directory` tests (currently at the bottom of `src/types/repo.rs`).
   - **Commit-creation types** from `src/types/commit.rs`:
     - `CommitInfo`, `CommitOperation`, `AddSource`.
   - **File-operation params** from `src/types/repo_params.rs`:
     - `RepoListFilesParams`, `RepoListTreeParams`.
     - `RepoGetFileMetadataParams`, `RepoGetPathsInfoParams`.
     - `RepoDownloadFileParams`, `RepoDownloadFileStreamParams`.
     - `RepoDownloadFileToBytesParams` and `RepoDownloadFileToBytesParamsBuilder` type aliases (lines 144-145 of current `repo_params.rs`).
     - `RepoSnapshotDownloadParams`.
     - `RepoUploadFileParams`, `RepoUploadFolderParams`.
     - `RepoDeleteFileParams`, `RepoDeleteFolderParams`.
     - `RepoCreateCommitParams`.

2. **`src/files/listing.rs`** — from `src/api/files.rs`:
   - The following `pub` methods on `HFRepository`, wrapped in a single `impl HFRepository { … }` block:
     - `list_files` (line 27).
     - `list_tree` (line 51).
     - `get_paths_info` (line 69).
     - `get_file_metadata` (line 109).
   - Any private helpers used only by these methods. If a helper is shared with download or upload, move it to `files/mod.rs` as `pub(super) fn …`.
   - The `sync_api!` entries for these four methods (from lines 1691–1707 of `src/api/files.rs`).

3. **`src/files/download.rs`** — from `src/api/files.rs`:
   - The following methods, wrapped in an `impl HFRepository { … }` block:
     - `download_file`, `download_file_inner` (private helper).
     - `download_file_stream`, `download_file_to_bytes`.
     - `download_file_to_local_dir`, `resolve_from_cache_only`, `find_cached_etag`, `download_file_to_cache`, `download_file_to_cache_network`, `resolve_commit_hash`, `list_filtered_files` (private helpers used only in download).
     - `snapshot_download`.
   - Free functions used only in download: `extract_etag`, `extract_commit_hash`, `build_download_params`, `download_concurrently`, `stream_response_to_file_with_progress`, `mark_no_exist_and_return_error`, `finalize_cached_file`.
   - `pub(crate)` free functions that may be shared: `extract_file_size`, `extract_xet_hash` — if used only by download, keep here; if upload references them, move to `files/mod.rs`. Check with grep during execution.
   - The `sync_api!` entries for `download_file`, `download_file_to_bytes`, `snapshot_download`.

4. **`src/files/upload.rs`** — from `src/api/files.rs`:
   - The following methods, wrapped in an `impl HFRepository { … }` block:
     - `create_commit`, `inline_base64_entry` (private helper).
     - `upload_file`, `upload_folder`.
     - `delete_file`, `delete_folder`.
     - `preupload_and_upload_lfs_files`, `fetch_upload_modes`, `upload_lfs_files_via_xet`, `post_lfs_batch_info` (private helpers).
   - Free functions used only in upload: `hex_encode`, `sha256_of_source`, `read_size_and_sample`, `collect_files_recursive`, `matches_any_glob`.
   - The `sync_api!` entries for `create_commit`, `upload_file`, `upload_folder`, `delete_file`, `delete_folder`.

**Shared helpers discovery:** Before splitting, grep inside `src/api/files.rs` for each free function to confirm where it is used. Any function used by ≥2 of {listing, download, upload} goes into `files/mod.rs` as `pub(super)`.

**Steps:**

- [ ] **Step 1: Enumerate private helpers in `src/api/files.rs`.** Run:
  ```bash
  grep -n "^fn\|^async fn\|^pub(crate) fn" hf-hub/src/api/files.rs
  ```
  For each helper, grep its name within the same file to determine which methods call it. Assign each helper to listing / download / upload / shared accordingly.
- [ ] **Step 2: Create `hf-hub/src/files/mod.rs`** — types, params, sub-module declarations, shared helpers.
- [ ] **Step 3: Create `hf-hub/src/files/listing.rs`** — listing methods + listing-only helpers + sync_api!.
- [ ] **Step 4: Create `hf-hub/src/files/download.rs`** — download/snapshot methods + download-only helpers + sync_api!.
- [ ] **Step 5: Create `hf-hub/src/files/upload.rs`** — upload/commit/delete methods + upload-only helpers + sync_api!.
- [ ] **Step 6: Validate compilation:** `cargo build -p hf-hub --all-features` (orphan files — unchanged).
- [ ] **Step 7: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 8: Commit:**
  ```bash
  git add hf-hub/src/files/
  git commit -m "refactor(hf-hub): add files/ component module (not yet wired)"
  ```

### Task 1D: Author `src/buckets/` (mod.rs + sync.rs)

**Files:**
- Create: `hf-hub/src/buckets/mod.rs`
- Create: `hf-hub/src/buckets/sync.rs`

**Content sources:**

1. **`src/buckets/mod.rs`**:
   - **HFBucket handle** from `src/bucket.rs` (83 LOC) — the full contents: struct, `Debug`, construction, URL helpers.
   - **Bucket types and params** from `src/types/buckets.rs`:
     - `CreateBucketParams`, `ListBucketTreeParams`, `BatchBucketFilesParams`, `BucketAddFile`, `BucketCopyFile`, `BucketDownloadFilesParams`.
     - `BucketInfo`, `BucketUrl`, `BucketTreeEntry`, `BucketFileMetadata`.
     - **Do NOT move to mod.rs:** `BucketSyncDirection`, `BucketSyncParams`, `BucketSyncAction`, `BucketSyncOperation`, `BucketSyncPlan` and their tests — those go to `buckets/sync.rs`.
   - **API impls** from `src/api/buckets/mod.rs`:
     - The `pub mod sync;` sub-module declaration.
     - Constants: `BUCKET_BATCH_CHUNK_SIZE`, `BUCKET_PATHS_INFO_BATCH_SIZE`.
     - `impl HFClient { create_bucket, list_buckets, delete_bucket, move_bucket, … }` — the full block.
     - `impl HFBucket { info, list_tree, get_paths_info, batch, download_files, get_file_metadata, … }` — the full block.
     - The `sync_api!` blocks verbatim.

2. **`src/buckets/sync.rs`**:
   - **Sync types** from `src/types/buckets.rs`:
     - `BucketSyncDirection`, `BucketSyncParams`, `BucketSyncAction`, `BucketSyncOperation`, `BucketSyncPlan` and the `impl BucketSyncPlan { uploads, downloads, deletes, skips, transfer_bytes }` block.
     - The `#[cfg(test)] mod tests` block at the bottom of `types/buckets.rs` (plan counts / transfer bytes tests).
   - **Sync API impl** — full contents of `src/api/buckets/sync.rs`:
     - `impl HFBucket { sync }` and all private helpers in that file.
     - The `sync_api!` block for `sync`.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/buckets/sync.rs`** — sync types + sync API impls.
- [ ] **Step 2: Create `hf-hub/src/buckets/mod.rs`** — handle + non-sync types + non-sync API impls + `pub mod sync;`.
- [ ] **Step 3: Validate compilation:** `cargo build -p hf-hub --all-features` (orphan files).
- [ ] **Step 4: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 5: Commit:**
  ```bash
  git add hf-hub/src/buckets/
  git commit -m "refactor(hf-hub): add buckets/ component module (not yet wired)"
  ```

### Task 1E: Author `src/spaces.rs`

**Files:**
- Create: `hf-hub/src/spaces.rs`

**Content sources:**

1. **HFSpace handle** from `src/repository.rs`:
   - `HFSpace` struct, `impl fmt::Debug for HFSpace`, `impl HFClient { pub fn space }`, `impl HFSpace { pub fn new, pub fn repo }`, `impl TryFrom<HFRepository> for HFSpace`, `impl From<HFSpace> for Arc<HFRepository>`, `impl Deref for HFSpace`.
   - `#[cfg(test)] mod tests` — only `test_hfspace_constructor_and_deref` and `test_hfspace_try_from_repo` (leave `test_repo_path_and_accessors` in `repo.rs`).

2. **Response types** from `src/types/spaces.rs`:
   - `SpaceRuntime`, `SpaceVariable`, and their `#[cfg(test)] mod tests`.

3. **Params** from `src/types/params.rs`:
   - `DuplicateSpaceParams`.

4. **Space-scoped params** from `src/types/repo_params.rs`:
   - `SpaceHardwareRequestParams`, `SpaceSleepTimeParams`.
   - `SpaceSecretParams`, `SpaceSecretDeleteParams`.
   - `SpaceVariableParams`, `SpaceVariableDeleteParams`.

5. **API impls** — full contents of `src/api/spaces.rs`:
   - `impl HFSpace { runtime, request_hardware, set_sleep_time, add_secret, delete_secret, add_variable, delete_variable, duplicate, pause, restart, … }`.
   - The `sync_api!` block for `HFSpace -> HFSpaceSync` verbatim.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/spaces.rs`** per the sources above.
- [ ] **Step 2: Validate compilation:** `cargo build -p hf-hub --all-features`.
- [ ] **Step 3: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 4: Commit:**
  ```bash
  git add hf-hub/src/spaces.rs
  git commit -m "refactor(hf-hub): add spaces.rs component module (not yet wired)"
  ```

### Task 1F: Author `src/users.rs`

**Files:**
- Create: `hf-hub/src/users.rs`

**Content sources:**

1. **Types** from `src/types/user.rs`:
   - `User`, `OrgMembership`, `Organization`.

2. **API impls** — full contents of `src/api/users.rs`:
   - `impl HFClient { whoami, auth_check, get_user_overview, get_organization, list_followers, list_following, … }`.
   - The `sync_api!` blocks verbatim.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/users.rs`** per the sources above.
- [ ] **Step 2: Validate compilation:** `cargo build -p hf-hub --all-features`.
- [ ] **Step 3: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 4: Commit:**
  ```bash
  git add hf-hub/src/users.rs
  git commit -m "refactor(hf-hub): add users.rs component module (not yet wired)"
  ```

### Task 1G: Author `src/progress.rs`

**Files:**
- Create: `hf-hub/src/progress.rs`

**Content source:** The full contents of `src/types/progress.rs` (653 LOC), verbatim. No item is split out. The only change is rewriting internal imports if the file references `crate::types::*` anywhere (it should not — progress is self-contained).

**Steps:**

- [ ] **Step 1: Copy `src/types/progress.rs` to `src/progress.rs`** (`cp`). Scan for any `crate::types::` references and rewrite if present.
- [ ] **Step 2: Validate compilation:** `cargo build -p hf-hub --all-features`.
- [ ] **Step 3: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 4: Commit:**
  ```bash
  git add hf-hub/src/progress.rs
  git commit -m "refactor(hf-hub): add progress.rs component module (not yet wired)"
  ```

### Task 1H: Author xet staging file `src/_new_xet.rs`

**Background:** `src/xet.rs` already exists; we cannot place new content at the final path without clobbering the active module. Stage to `src/_new_xet.rs` instead; Phase 2 replaces `src/xet.rs` atomically.

**Files:**
- Create: `hf-hub/src/_new_xet.rs` (staging; removed in Phase 2).

**Content sources:**

1. **Existing `src/xet.rs`** (817 LOC) verbatim — keep `XetConnectionInfo`, `XetError`, the URL helpers, transfer poller, download progress, upload plumbing, and every `pub(crate)` item. Rewrite its `use` statements per the translation table (most of `use crate::types::{…}` changes to `crate::repo::RepoType`, `crate::files::AddSource`, `crate::progress::*`).

2. **Public params** from `src/types/params.rs`:
   - `XetTokenType` (enum + `impl XetTokenType { as_str }`).
   - `GetXetTokenParams`.

3. **Public API** from `src/api/repo.rs`:
   - The `pub async fn get_xet_token(&self, params: &GetXetTokenParams) -> HFResult<XetConnectionInfo>` method on `HFClient`, and any helpers in `api/repo.rs` used only by it. Find the method via `grep -n "fn get_xet_token" hf-hub/src/api/repo.rs`.
   - The corresponding `sync_api!` entry that mentions `get_xet_token` (search `grep -n "get_xet_token" hf-hub/src/api/repo.rs`).

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/_new_xet.rs`** with the merged content. Place `XetTokenType` / `GetXetTokenParams` near the top (public surface), then `XetConnectionInfo` (already pub), then the existing plumbing, then the new `impl HFClient { get_xet_token }` block, then the `sync_api!` wrapper for `get_xet_token`.
- [ ] **Step 2: Validate compilation:** `cargo build -p hf-hub --all-features` (staging file is not declared — orphan, ignored).
- [ ] **Step 3: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 4: Commit:**
  ```bash
  git add hf-hub/src/_new_xet.rs
  git commit -m "refactor(hf-hub): stage new xet.rs content (not yet wired)"
  ```

### Task 1I: Author cache staging files `src/_new_cache_mod.rs` and `src/_new_cache_storage.rs`

**Background:** `src/cache.rs` already exists and collides with the future `src/cache/` directory. Stage the new files; Phase 2 does the directory swap atomically.

**Files:**
- Create: `hf-hub/src/_new_cache_mod.rs` (staging → `src/cache/mod.rs` in Phase 2).
- Create: `hf-hub/src/_new_cache_storage.rs` (staging → `src/cache/storage.rs` in Phase 2).

**Content sources:**

1. **`_new_cache_storage.rs`** — the full contents of `src/cache.rs` (613 LOC) verbatim. Rewrite internal `use` statements per the translation table (notably `crate::types::RepoType` → `crate::repo::RepoType` and `crate::types::cache::{…}` → `crate::cache::{…}`). Change visibility of items exposed to other modules to `pub(crate)` if they are not already. The file will be referenced as `mod storage;` from `cache/mod.rs`, so items read `crate::cache::storage::foo`.

2. **`_new_cache_mod.rs`**:
   - `pub(crate) mod storage;` sub-module declaration.
   - **Types** from `src/types/cache.rs`:
     - `CachedFileInfo`, `CachedRevisionInfo`, `CachedRepoInfo`, `HFCacheInfo`.
     - `DeleteCacheRevision` (if present — search `grep -n "DeleteCacheRevision" hf-hub/src/types/cache.rs hf-hub/src/cache.rs`).
   - **API impl** — full contents of `src/api/cache.rs`:
     - `impl HFClient { scan_cache, delete_cache_revisions }` (whichever methods exist — verify by reading the file).
     - The `sync_api!` block.

**Steps:**

- [ ] **Step 1: Create `hf-hub/src/_new_cache_storage.rs`** (copy of current `src/cache.rs`, with imports and visibilities adjusted).
- [ ] **Step 2: Create `hf-hub/src/_new_cache_mod.rs`** (types + scan API + `pub(crate) mod storage;`).
- [ ] **Step 3: Validate compilation:** `cargo build -p hf-hub --all-features` (both staging files are orphans).
- [ ] **Step 4: Format and lint:** `cargo +nightly fmt -p hf-hub && cargo clippy -p hf-hub --all-features -- -D warnings`.
- [ ] **Step 5: Commit:**
  ```bash
  git add hf-hub/src/_new_cache_mod.rs hf-hub/src/_new_cache_storage.rs
  git commit -m "refactor(hf-hub): stage new cache/ module content (not yet wired)"
  ```

---

## Phase 2 — Atomic cutover (SERIAL)

This phase is a single task. It flips `lib.rs` to reference the new modules, removes the old `types/` and `api/` directories, and finishes the in-place renames for `cache` and `xet`. After this task, the crate uses the new layout; `cargo build` and `cargo test` must pass.

### Task 2A: Wire new modules and delete old layout

**Files:**
- Modify: `hf-hub/src/lib.rs`
- Delete: `hf-hub/src/types/` (entire directory)
- Delete: `hf-hub/src/api/` (entire directory)
- Delete: `hf-hub/src/repository.rs`
- Delete: `hf-hub/src/bucket.rs`
- Delete: `hf-hub/src/diff.rs`
- Delete: `hf-hub/src/cache.rs`
- Delete: `hf-hub/src/xet.rs`
- Rename/move: `hf-hub/src/_new_xet.rs` → `hf-hub/src/xet.rs`
- Rename/move: `hf-hub/src/_new_cache_mod.rs` → `hf-hub/src/cache/mod.rs`
- Rename/move: `hf-hub/src/_new_cache_storage.rs` → `hf-hub/src/cache/storage.rs`

- [ ] **Step 1: Perform the file operations.**
  ```bash
  cd hf-hub/src

  # Remove old directories and monoliths
  git rm -r types
  git rm -r api
  git rm repository.rs
  git rm bucket.rs
  git rm diff.rs
  git rm cache.rs
  git rm xet.rs

  # Put xet in place
  git mv _new_xet.rs xet.rs

  # Put cache in place
  mkdir cache
  git mv _new_cache_mod.rs cache/mod.rs
  git mv _new_cache_storage.rs cache/storage.rs

  cd -
  ```

- [ ] **Step 2: Rewrite `hf-hub/src/lib.rs`.** Replace the current module declarations and re-exports so the file ends up exactly matching the structure below (preserve the existing crate-level doc comment at the top — only the `mod`/`pub use`/`#![…]` block changes).

  ```rust
  #![cfg_attr(docsrs, feature(doc_cfg))]

  #[macro_use]
  mod macros;

  // Private infrastructure
  mod client;
  mod constants;
  mod error;
  mod pagination;
  mod retry;

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

  // Crate-root re-exports — the minimal ergonomic set
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

  Update doctests in the crate-level doc comment that reference `use hf_hub::types::…` to use the new paths (e.g. `use hf_hub::repo::{RepoType, RepoInfoParams};`).

- [ ] **Step 3: Update internal imports inside `src/client.rs`, `src/blocking/mod.rs`, and any other file that still references `crate::types::`, `crate::api::`, `crate::repository::`, `crate::bucket::`, or `crate::diff::`.**
  ```bash
  grep -rln "crate::types::\|crate::api::\|crate::repository::\|crate::bucket::\|crate::diff::" hf-hub/src/
  ```
  For each match, rewrite the import path per the translation table at the top of Phase 1.

- [ ] **Step 4: Build the crate.** First pass catches any missed imports.
  ```bash
  cargo build -p hf-hub --all-features
  ```
  Expected: PASS. If any error references a missing type, locate where it now lives and fix the import.

- [ ] **Step 5: Run the unit tests.**
  ```bash
  cargo test -p hf-hub --all-features --lib
  ```
  Expected: PASS. Same test count as before (no tests added or removed; they moved with their types).

- [ ] **Step 6: Format and lint.**
  ```bash
  cargo +nightly fmt -p hf-hub
  cargo clippy -p hf-hub --all-features -- -D warnings
  ```
  Expected: both pass. Clippy may surface unused-import warnings if the translation missed any — clean up until green.

- [ ] **Step 7: Commit.**
  ```bash
  git add -A hf-hub/src/
  git commit -m "refactor(hf-hub): cut over to per-component modules, drop types/ and api/"
  ```

---

## Phase 3 — Update in-tree consumers (PARALLEL)

All three tasks in this phase run concurrently.

### Task 3A: Update `examples/`

**Files:** every `.rs` file under `examples/` that imports from `hf_hub::types` or other removed paths.

- [ ] **Step 1: Enumerate the files that need changes.**
  ```bash
  grep -rln "hf_hub::types\|hf_hub::diff\|hf_hub::repository\|hf_hub::bucket" examples/
  ```
  Expected list (from the spec): `repo.rs`, `repo_handles.rs`, `files.rs`, `commits.rs`, `diff.rs`, `spaces.rs`, `buckets.rs`, `download_upload.rs`, `progress.rs`, `progress_logger.rs`, `blocking_read.rs`, `blocking_write.rs`, `blocking_repo_handles.rs`, `blocking_spaces.rs`.

- [ ] **Step 2: Rewrite imports in each file.** Apply the translation table from Phase 1, adapted for the external crate path:
  | Old | New |
  |---|---|
  | `hf_hub::types::RepoType` | `hf_hub::RepoType` |
  | `hf_hub::types::{RepoInfo, RepoInfoParams, …}` (repo items) | `hf_hub::repo::{…}` |
  | `hf_hub::types::{RepoDownloadFileParams, RepoUploadFileParams, CommitOperation, AddSource, …}` | `hf_hub::files::{…}` |
  | `hf_hub::types::{GitCommitInfo, GitRefs, RepoGetRawDiffParams, RepoCreateBranchParams, …}` | `hf_hub::commits::{…}` |
  | `hf_hub::types::{BucketInfo, CreateBucketParams, ListBucketTreeParams, BucketDownloadFilesParams, …}` | `hf_hub::buckets::{…}` |
  | `hf_hub::types::{SpaceRuntime, SpaceSecretParams, …}` | `hf_hub::spaces::{…}` |
  | `hf_hub::types::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler, UploadEvent, FileProgress, …}` | `hf_hub::progress::{…}` — except `ProgressHandler` which is also at `hf_hub::ProgressHandler` |
  | `hf_hub::types::Progress` | `hf_hub::progress::Progress` |

- [ ] **Step 3: Build the examples crate.**
  ```bash
  cargo build -p examples
  ```
  Expected: PASS. If any example errors on a renamed path, locate the type and update.

- [ ] **Step 4: Format and lint.**
  ```bash
  cargo +nightly fmt -p examples
  cargo clippy -p examples --all-features -- -D warnings
  ```
  Expected: both pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add examples/
  git commit -m "refactor(examples): update imports for hf-hub module reorg"
  ```

### Task 3B: Update `hfrs/`

**Files:** every `.rs` file under `hfrs/src/` and `hfrs/tests/` that imports from the removed paths.

- [ ] **Step 1: Enumerate files:**
  ```bash
  grep -rln "hf_hub::types\|hf_hub::diff\|hf_hub::repository\|hf_hub::bucket" hfrs/
  ```
  Known files from the spec: `hfrs/src/cli.rs`, `hfrs/src/progress.rs`, `hfrs/src/util/mod.rs`, `hfrs/src/commands/upload.rs`, `hfrs/src/commands/buckets/cp.rs`, `hfrs/src/commands/repos/branch.rs`, and any others surfaced by the grep.

- [ ] **Step 2: Rewrite imports** per the translation table in Task 3A.

- [ ] **Step 3: Build the CLI crate.**
  ```bash
  cargo build -p hfrs --all-features
  ```
  Expected: PASS.

- [ ] **Step 4: Run hfrs tests.**
  ```bash
  cargo test -p hfrs
  ```
  Expected: PASS.

- [ ] **Step 5: Format and lint.**
  ```bash
  cargo +nightly fmt -p hfrs
  cargo clippy -p hfrs --all-features -- -D warnings
  ```
  Expected: both pass.

- [ ] **Step 6: Commit.**
  ```bash
  git add hfrs/
  git commit -m "refactor(hfrs): update imports for hf-hub module reorg"
  ```

### Task 3C: Update `hf-hub/tests/integration_test.rs`

**Files:** `hf-hub/tests/integration_test.rs`.

- [ ] **Step 1: Rewrite imports** per the translation table in Task 3A.

- [ ] **Step 2: Build the integration tests (without running them — they need `HF_TOKEN`).**
  ```bash
  cargo test -p hf-hub --test integration_test --no-run
  ```
  Expected: PASS.

- [ ] **Step 3: Format and lint.**
  ```bash
  cargo +nightly fmt -p hf-hub
  cargo clippy -p hf-hub --tests --all-features -- -D warnings
  ```
  Expected: both pass.

- [ ] **Step 4: Commit.**
  ```bash
  git add hf-hub/tests/integration_test.rs
  git commit -m "refactor(hf-hub-tests): update imports for module reorg"
  ```

---

## Phase 4 — Final validation (SERIAL)

### Task 4A: Full workspace validation

- [ ] **Step 1: Format the whole workspace.**
  ```bash
  cargo +nightly fmt --all
  ```
  Expected: no diff (idempotent after prior per-crate runs). If a diff appears, commit it.

- [ ] **Step 2: Clippy on everything, all features, deny warnings.**
  ```bash
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  ```
  Expected: PASS.

- [ ] **Step 3: Build the whole workspace including all targets.**
  ```bash
  cargo build --workspace --all-targets --all-features
  ```
  Expected: PASS.

- [ ] **Step 4: Run the full unit test suite.**
  ```bash
  cargo test -p hf-hub --all-features
  cargo test -p hfrs --all-features
  ```
  Expected: PASS, with the same test count as before the reorg.

- [ ] **Step 5: (If `HF_TOKEN` is available) Run the read-only integration tests.**
  ```bash
  HF_TOKEN="$HF_TOKEN" cargo test -p hf-hub --test integration_test
  ```
  Expected: PASS. Skips gracefully if `HF_TOKEN` is unset.

- [ ] **Step 6: Verify no stray references to the old paths remain.**
  ```bash
  grep -rn "crate::types::\|crate::api::\|crate::repository::\|crate::bucket::\|crate::diff::" hf-hub/ examples/ hfrs/ 2>&1 | grep -v "^Binary file" || echo "CLEAN"
  grep -rn "hf_hub::types::\|hf_hub::diff::\|hf_hub::repository::\|hf_hub::bucket::" hf-hub/ examples/ hfrs/ 2>&1 | grep -v "^Binary file" || echo "CLEAN"
  ```
  Expected: both print `CLEAN`.

- [ ] **Step 7: Check nothing extra was committed.** List the final tree:
  ```bash
  find hf-hub/src -type f -name "*.rs" | sort
  ```
  Expected output:
  ```
  hf-hub/src/blocking/mod.rs
  hf-hub/src/buckets/mod.rs
  hf-hub/src/buckets/sync.rs
  hf-hub/src/cache/mod.rs
  hf-hub/src/cache/storage.rs
  hf-hub/src/client.rs
  hf-hub/src/commits/diff.rs
  hf-hub/src/commits/mod.rs
  hf-hub/src/constants.rs
  hf-hub/src/error.rs
  hf-hub/src/files/download.rs
  hf-hub/src/files/listing.rs
  hf-hub/src/files/mod.rs
  hf-hub/src/files/upload.rs
  hf-hub/src/lib.rs
  hf-hub/src/macros.rs
  hf-hub/src/pagination.rs
  hf-hub/src/progress.rs
  hf-hub/src/repo.rs
  hf-hub/src/retry.rs
  hf-hub/src/spaces.rs
  hf-hub/src/test_utils.rs
  hf-hub/src/users.rs
  hf-hub/src/xet.rs
  ```

- [ ] **Step 8: Commit any format or clippy fixes surfaced by the workspace-wide passes.** If everything was already clean, skip this commit.

  ```bash
  git status
  # If there are changes:
  git add -A
  git commit -m "chore: final fmt/clippy pass after module reorg"
  ```

---

## Parallel execution map

For subagent-driven execution, the recommended dispatch pattern is:

1. **Phase 1 parallel batch** — dispatch Tasks 1A, 1B, 1C, 1D, 1E, 1F, 1G, 1H, 1I simultaneously. Each produces a single commit. (Heaviest task is 1C, files.)
2. **Phase 2 serial** — single agent runs Task 2A after all Phase 1 tasks report complete.
3. **Phase 3 parallel batch** — dispatch Tasks 3A, 3B, 3C simultaneously.
4. **Phase 4 serial** — single agent runs Task 4A.

Review checkpoints: after Phase 1 (optional — each task is independent and reviewable on its own), after Phase 2 (MANDATORY — the atomic cutover is where behavior risk lives; ensure test count is preserved), after Phase 4 (final green signal).
