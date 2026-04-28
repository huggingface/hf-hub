# Releasing hf-hub

This document covers the full release process for the `hf-hub` crate. If anything here is unclear or out of date, please open a PR.

## What gets released

A single tag push releases one artifact:

- **`hf-hub` Rust crate** on [crates.io](https://crates.io/crates/hf-hub), via `.github/workflows/rust-release.yml`.

The workflow triggers on tags matching `v*` (e.g. `v1.0.0`, `v1.0.0-rc.0`). Tags containing `rc` are skipped by the publish step, so they can be used for pre-release validation without producing a crates.io upload.

There are no Python components in this repo. The other workspace members are not published:

- `hfrs/` — CLI binary, distributed via `cargo install --git`.
- `examples/`, `benches/`, `integration-tests/` — internal-only, version `0.0.0`, never published.

## Pre-release checklist

1. **CI is green on `main`.** The `Rust` workflow must be passing on every platform in the matrix (Ubuntu, Windows, macOS) with both feature configurations (`""` and `--all-features`).
2. **Review the diff since the last release.**
   ```bash
   git log --oneline v0.5.0..main
   git diff v0.5.0..main --stat -- hf-hub/
   ```
   Pay particular attention to changes under `hf-hub/src/` — those are the only changes that actually ship to crates.io.
3. **Identify breaking changes.** Anything that changes the public Rust API (types, function signatures, removed re-exports, builder fields) needs to be reflected in the version bump per [semver](https://semver.org) and called out in the release notes.
4. **Run the full pre-release test sweep** (see next section).

## Pre-release test sweep

Run all of these from the repo root before tagging. They mirror what CI runs, plus a publish dry-run that CI does not currently do.

### Format and lint

```bash
cargo +nightly fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

### Unit tests (`hf-hub`)

```bash
cargo test -p hf-hub
cargo test -p hf-hub --features blocking
```

### Integration tests (`integration-tests`)

These hit the live Hub API. They skip gracefully if `HF_TOKEN` is unset, so always pass it explicitly when validating a release. For a release, run the full suite including writes:

```bash
HF_TOKEN=hf_xxx HF_TEST_WRITE=1 cargo test -p integration-tests
```

The covered surfaces are:

- `integration_test.rs` — read-only API coverage
- `blocking_test.rs` — `blocking` feature parity
- `download_test.rs` — file download paths
- `xet_transfer_test.rs` — Xet upload/download
- `cache_test.rs` — local cache scanning and locking
- `bucket_sync_test.rs` — bucket sync planning and execution
- `bucket_xet_transfer_test.rs` — bucket Xet transfers

### CLI (`hfrs`)

The CLI is not published, but it consumes the same crate we're shipping, so its tests are a good consistency check:

```bash
cargo test -p hfrs
```

For a release with user-facing CLI changes, also do a manual smoke run:

```bash
cargo run -p hfrs -- version
cargo run -p hfrs -- env
HF_TOKEN=hf_xxx cargo run -p hfrs -- models info openai-community/gpt2
```

### Examples

Examples are part of the public-facing surface (they live next to the README and are linked from docs). They must compile cleanly and use the current public API:

```bash
cargo build -p examples --all-features
```

If you changed any public types or builders, spot-check that the most affected examples still read correctly — examples are documentation.

### Docs

`docs.rs` will rebuild docs on publish. Catch breakage locally first:

```bash
RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +nightly doc --workspace --all-features --no-deps
```

### Benches (perf-sensitive releases only)

If the release includes performance-sensitive work in `hf-hub`, run the suite and compare against the previous release:

```bash
cargo bench -p hf-hub-benches
```

### Publish dry-run

This is the canonical "will the workflow succeed?" check. It runs cargo's full publish pipeline including the verify build, but stops short of uploading. It catches packaging issues (missing files, broken symlinks, README path problems, version conflicts) that unit tests do not.

```bash
cd hf-hub && cargo publish --dry-run
```

If the working tree has uncommitted changes you want to ignore for the dry-run only, add `--allow-dirty`. Do not pass `--allow-dirty` once you've created the release commit — a clean tree at tag time is the contract.

## Cutting the release

### 1. Set the release version

Update `hf-hub/Cargo.toml`:

```toml
version = "1.0.0"
```

Run `cargo check` to refresh `Cargo.lock`. Commit:

```
Set version to 1.0.0
```

Open a PR, get CI green, merge to `main`.

### 2. Tag and push

```bash
git checkout main && git pull
git tag v1.0.0
git push origin v1.0.0
```

The tag triggers `.github/workflows/rust-release.yml`, which checks out the tagged commit, runs `cargo publish` from `./hf-hub`, and uploads to crates.io.

### 3. Monitor the workflow

Check the [Actions tab](https://github.com/huggingface/hf-hub/actions) for the `Rust Release` run. It typically completes in a few minutes.

### 4. Create the GitHub release

Once the crate is on crates.io, draft a release on the [Releases page](https://github.com/huggingface/hf-hub/releases) targeting the tag you just pushed. Generate release notes from the previous tag and add highlights.

Suggested structure:

```markdown
## Breaking changes
- ...

## New features
- ...

## Bug fixes
- ...

## Internal / CI
- ...
```

### 5. Verify

```bash
cargo search hf-hub
```

Then in a scratch directory:

```bash
cargo new --bin verify-hf-hub && cd verify-hf-hub
cargo add hf-hub@1.0.0 tokio --features tokio/macros,tokio/rt-multi-thread
cargo build
```

Also confirm:

- [crates.io/crates/hf-hub](https://crates.io/crates/hf-hub) shows the new version
- [docs.rs/hf-hub](https://docs.rs/hf-hub) finishes the docs build (may take several minutes after publish)

## Release candidates

For releases with significant changes, cut an RC first.

- Set the version to `1.0.0-rc.0` in `hf-hub/Cargo.toml`.
- Tag as `v1.0.0-rc.0` and push.
- The workflow's `if: ${{ !contains(github.ref, 'rc') }}` guard skips the publish step, so the tag exists for record-keeping but no crates.io upload happens.
- If you want the RC actually published to crates.io for downstream testing, drop or relax that guard temporarily — but generally prefer a separate `git_v*` branch with a regular tag.

## Hotfixing a release

`cargo publish` cannot replace an already-published version. If a released version is broken:

1. Fix the bug on `main` (or on a release branch if `main` has moved on).
2. Bump the patch version (`1.0.0` → `1.0.1`).
3. Tag and push as usual.

The broken version stays on crates.io but can be marked yanked via `cargo yank --vers 1.0.0 hf-hub` (yanking does not delete; it prevents new resolution from picking the version).

## Secrets

The release workflow uses one repository secret:

- `CRATES_TOKEN` — crates.io API token with publish rights for the `hf-hub` crate.

If publish fails with an auth error, the token has likely expired or lost ownership. Rotate it on [crates.io](https://crates.io/me) and update the repo secret in **Settings → Secrets and variables → Actions**.

## Testing release CI changes

If you're modifying `.github/workflows/rust-release.yml`:

1. Replace `cargo publish` with `cargo publish --dry-run --allow-dirty` in your branch.
2. Temporarily change the trigger to `push` on your branch (or `workflow_dispatch`).
3. Iterate until the pipeline runs cleanly.
4. Revert both changes before merging.

## Troubleshooting

- **`cargo publish` says "crate version is already uploaded".** You cannot re-publish the same version. Bump to the next patch or pre-release and tag again.
- **403 from crates.io.** The token in `CRATES_TOKEN` is missing publish rights for `hf-hub`. Ask the current owner to add the token's user via `cargo owner --add <user> hf-hub`.
- **`failed to open for archiving: README.md` / "Too many levels of symbolic links".** The crate-level `README.md` symlink target is wrong. It must point at the workspace-root README via `../README.md` (not just `README.md`, which would resolve to itself). Fix with `ln -sfn ../README.md hf-hub/README.md`.
- **`manifest has no documentation, homepage or repository`.** Non-blocking warning, but those fields should stay populated in `hf-hub/Cargo.toml` so crates.io and docs.rs render the proper links.
- **Workflow didn't trigger.** Verify the tag was pushed (`git push origin <tag>`) and that the name matches `v*`. Annotated and lightweight tags both work.
- **Docs.rs build failed after publish.** The `[package.metadata.docs.rs]` block in `hf-hub/Cargo.toml` controls the docs.rs build. If it broke, reproduce locally with `RUSTDOCFLAGS="-D warnings --cfg docsrs" cargo +nightly doc -p hf-hub --features blocking --no-deps` before bumping the patch version.
