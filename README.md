# hf-hub

> **Note:** This library is experimental. APIs may change without notice between versions.

Async Rust client for the [Hugging Face Hub API](https://huggingface.co/docs/hub/api).

`hf-hub` provides a typed, ergonomic interface for interacting with the Hugging Face Hub from Rust. It is the Rust equivalent of the Python [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library.

## Features

- **Repository operations** — query model, dataset, and space metadata; create, delete, update, and move repositories
- **File operations** — upload files and folders, download files, list repository trees, check file existence
- **Commit operations** — create commits with multiple file operations, list commit history, view diffs between revisions
- **Branch and tag management** — create and delete branches and tags, list refs
- **User and organization info** — whoami, user profiles, organization details, followers
- **Streaming pagination** — list endpoints return `impl Stream<Item = Result<T>>` for lazy, memory-efficient iteration
- **Bucket operations** — create, delete, list, and move buckets; upload, download, and delete files within buckets
- **Xet high-performance transfers** — support for Hugging Face's Xet storage backend

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hf-hub = { git = "https://github.com/huggingface/hf-hub.git" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## CLI Installation

The `hfrs` command-line tool provides a terminal interface to the Hub. Install it with:

```sh
cargo install --git https://github.com/huggingface/hf-hub.git hfrs
```

This builds in release mode by default. Once installed, run `hfrs --help` to see available commands.

## Quick Start

```rust,no_run
use hf_hub::HFClient;
use hf_hub::types::{RepoInfo, RepoInfoParams};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    // Get model info
    let RepoInfo::Model(info) = client
        .model("openai-community", "gpt2")
        .info(&RepoInfoParams::default())
        .await?
    else {
        unreachable!("handle type guarantees the Model variant");
    };
    println!("Model: {} (downloads: {:?})", info.id, info.downloads);

    Ok(())
}
```

## Usage Examples

### List models by author

```rust,no_run
use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::types::ListModelsParams;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let params = ListModelsParams::builder()
        .author("meta-llama")
        .limit(5_usize)
        .build();

    let stream = client.list_models(&params)?;
    futures::pin_mut!(stream);

    while let Some(model) = stream.next().await {
        let model = model?;
        println!("{}", model.id);
    }

    Ok(())
}
```

### Work with a repository handle

```rust,no_run
use hf_hub::HFClient;
use hf_hub::types::{RepoFileExistsParams, RepoInfo, RepoInfoParams};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("openai-community", "gpt2");

    let RepoInfo::Model(model_info) = repo.info(&RepoInfoParams::default()).await? else {
        println!("error, not a model");
        return Ok(());
    };
    println!("Model: {}", model_info.id);

    let exists = repo
        .file_exists(
            &RepoFileExistsParams::builder()
                .filename("config.json")
                .build(),
        )
        .await?;

    println!("config.json exists: {exists}");
    Ok(())
}
```

### Download a file

```rust,no_run
use std::path::PathBuf;
use hf_hub::HFClient;
use hf_hub::types::RepoDownloadFileParams;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("openai-community", "gpt2");

    let path = repo.download_file(
        &RepoDownloadFileParams::builder()
            .filename("config.json")
            .local_dir(PathBuf::from("/tmp/hf-downloads"))
            .build()
    ).await?;

    println!("Downloaded to: {}", path.display());
    Ok(())
}
```

### Upload a file

```rust,no_run
use hf_hub::HFClient;
use hf_hub::types::{AddSource, RepoUploadFileParams};

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;
    let repo = client.model("your-username", "your-repo");

    let commit = repo.upload_file(
        &RepoUploadFileParams::builder()
            .source(AddSource::Bytes(b"Hello, world!".to_vec()))
            .path_in_repo("greeting.txt")
            .commit_message("Add greeting file")
            .build()
    ).await?;

    println!("Committed: {:?}", commit.oid);
    Ok(())
}
```

### Create a repository

```rust,no_run
use hf_hub::HFClient;
use hf_hub::types::CreateRepoParams;

#[tokio::main]
async fn main() -> hf_hub::HFResult<()> {
    let client = HFClient::new()?;

    let url = client.create_repo(
        &CreateRepoParams::builder()
            .repo_id("your-username/new-model")
            .private(true)
            .exist_ok(true)
            .build()
    ).await?;

    println!("Repository URL: {}", url.url);
    Ok(())
}
```

## Authentication

The client resolves authentication tokens in this order:

1. Explicit token via `HFClientBuilder::token()`
2. `HF_TOKEN` environment variable
3. Token file at path specified by `HF_TOKEN_PATH`
4. Default token file at `~/.cache/huggingface/token`

Set `HF_HUB_DISABLE_IMPLICIT_TOKEN` to any non-empty value to disable automatic token resolution.

## Configuration

| Environment Variable | Description |
|---|---|
| `HF_ENDPOINT` | Hub API endpoint (default: `https://huggingface.co`) |
| `HF_TOKEN` | Authentication token |
| `HF_TOKEN_PATH` | Path to token file |
| `HF_HOME` | Cache directory root (default: `~/.cache/huggingface`) |
| `HF_HUB_DISABLE_IMPLICIT_TOKEN` | Disable automatic token loading |
| `HF_HUB_USER_AGENT_ORIGIN` | Custom User-Agent origin string |

## Error Handling

All fallible operations return `Result<T, HFError>`. The `HFError` enum provides structured variants for common failure modes:

- `HFError::AuthRequired` — 401 response, token is missing or invalid
- `HFError::RepoNotFound` — repository does not exist or is inaccessible
- `HFError::BucketNotFound` — bucket does not exist or is inaccessible
- `HFError::EntryNotFound` — file or path does not exist in the repository or bucket
- `HFError::RevisionNotFound` — branch, tag, or commit does not exist
- `HFError::Forbidden` — 403 response, insufficient permissions
- `HFError::Conflict` — 409 response, resource already exists or conflicts
- `HFError::RateLimited` — 429 response, too many requests
- `HFError::Http` — other HTTP errors with status code, URL, and response body

## License

Apache-2.0
