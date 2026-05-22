//! wasm32-unknown-unknown integration tests for `hf-hub`.
//!
//! These run against the live Hub via a headless browser (configured by
//! `run_tests.sh` / the CI `wasm-test` job). Tests are wasm-bindgen-test
//! futures and target stable public resources so no auth is required.

#![cfg(target_family = "wasm")]

use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use hf_hub::HFClientBuilder;
use hf_hub::buckets::BucketTreeEntry;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::repository::AddSource;
use hf_hub::{HFClient, RepoTypeModel};
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(feature = "browser-tests")]
use wasm_bindgen_test::wasm_bindgen_test_configure;

// `wasm-bindgen-test`'s default runner is Node.js; compiling in
// `wasm_bindgen_test_configure!(run_in_browser)` switches it to the headless
// browser runner. We want both, so the directive is feature-gated: the
// `browser-tests` feature flips it on (the `wasm-test` CI job + local
// `RUN_IN_BROWSER=1 ./run_tests.sh`), absent it the tests run under Node
// (default `./run_tests.sh` and the `wasm-test-node` CI job). `hf-xet`'s
// threaded wasm needs `SharedArrayBuffer`, which Node 20+ provides natively
// and the `wasm-bindgen-test-runner` browser harness exposes via the
// COOP/COEP headers it sets automatically.
#[cfg(feature = "browser-tests")]
wasm_bindgen_test_configure!(run_in_browser);

const ENDPOINT: &str = "https://huggingface.co";
const TEST_MODEL_OWNER: &str = "openai-community";
const TEST_MODEL_NAME: &str = "gpt2";

fn client() -> hf_hub::HFClient {
    HFClientBuilder::new()
        .endpoint(ENDPOINT.to_string())
        .build()
        .expect("build HFClient")
}

#[wasm_bindgen_test]
async fn download_file_to_bytes_works() {
    let bytes = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .download_file_to_bytes()
        .filename("config.json")
        .send()
        .await
        .expect("download config.json");

    assert!(!bytes.is_empty(), "config.json was empty");
    let s = std::str::from_utf8(&bytes).expect("config.json is utf-8");
    assert!(s.trim_start().starts_with('{'), "config.json doesn't look like JSON: {s:.80}");
    assert!(s.contains("\"model_type\""), "config.json missing model_type field");
}

#[wasm_bindgen_test]
async fn model_info_works() {
    let info = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .info()
        .send()
        .await
        .expect("model info");

    assert_eq!(info.id, format!("{TEST_MODEL_OWNER}/{TEST_MODEL_NAME}"));
    assert!(info.pipeline_tag.is_some(), "expected gpt2 to have a pipeline_tag, got None",);
}

#[wasm_bindgen_test]
async fn raw_diff_works() {
    // Two old, immutable gpt2 commit SHAs — the older commit added the ONNX
    // file, the newer one added tokenizer_config.json. The diff between them
    // is small, stable, and non-empty.
    let diff = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .get_raw_diff()
        .compare("11c5a3d5811f..607a30d783df")
        .send()
        .await
        .expect("get_raw_diff");

    assert!(!diff.is_empty(), "raw diff between known gpt2 commits was empty");
    assert!(
        diff.contains("tokenizer_config.json"),
        "expected the diff to mention tokenizer_config.json, got: {diff:.200}",
    );
}

/// Small xet-backed file in a stable public test repo
/// (`hf-internal-testing/tiny-random-bert/pytorch_model.bin`, ~528 KiB).
/// Exercising this drives the wasm download dispatch through the
/// `paths-info`-detected xet path in `repository/download.rs`, which
/// goes through `hf-xet`'s threaded-wasm runtime.
const XET_REPO_OWNER: &str = "hf-internal-testing";
const XET_REPO_NAME: &str = "tiny-random-bert";
const XET_FILE_NAME: &str = "pytorch_model.bin";
const XET_FILE_SIZE: usize = 540_217;

#[wasm_bindgen_test]
async fn download_xet_file_works() {
    let bytes = client()
        .model(XET_REPO_OWNER, XET_REPO_NAME)
        .download_file_to_bytes()
        .filename(XET_FILE_NAME)
        .send()
        .await
        .expect("download xet-backed pytorch_model.bin");

    assert_eq!(
        bytes.len(),
        XET_FILE_SIZE,
        "xet-backed file size mismatch: got {} bytes, want {XET_FILE_SIZE}",
        bytes.len(),
    );
    // PyTorch pickle archives are zip files, so the first two bytes are the
    // ZIP local file header magic. A sanity check that we got the actual
    // file content, not an LFS pointer or an error body.
    assert_eq!(&bytes[..2], b"PK", "pytorch_model.bin did not start with the ZIP magic");
}

#[wasm_bindgen_test]
async fn list_models_pagination_works() {
    // Passing `limit >= 1000` short-circuits the `?limit=` query
    // parameter (see `HFClient::list_models`), so the server returns its
    // default page size (1000) and the pagination loop has to follow the
    // `Link: rel="next"` header to satisfy the requested count.
    const REQUESTED: usize = 1100;

    let client = client();
    let stream = client.list_models().limit(REQUESTED).send().expect("list_models builder");
    futures_util::pin_mut!(stream);

    let mut count = 0usize;
    while let Some(item) = stream.next().await {
        let _model = item.expect("model item");
        count += 1;
    }

    assert_eq!(count, REQUESTED, "pagination short-changed us: got {count} of {REQUESTED} requested models");
}

#[wasm_bindgen_test]
async fn list_spaces_search_works() {
    let client = client();
    let stream = client
        .list_spaces()
        .search("stable-diffusion".to_string())
        .limit(3usize)
        .send()
        .expect("list_spaces builder");
    futures_util::pin_mut!(stream);

    let mut count = 0usize;
    while let Some(item) = stream.next().await {
        let space = item.expect("space item");
        assert!(!space.id.is_empty(), "space id was empty");
        count += 1;
    }

    assert!(count >= 1, "expected at least one space result, got {count}");
}

/// `ProgressHandler` that records every event it sees. Behind an `Arc<Mutex<_>>`
/// so it's `Send + Sync` (required by the handler trait) and the test can read
/// the recording back after the download.
#[derive(Default)]
struct EventRecorder {
    events: Mutex<Vec<ProgressEvent>>,
}

impl ProgressHandler for EventRecorder {
    fn on_progress(&self, event: &ProgressEvent) {
        self.events.lock().expect("recorder lock").push(event.clone());
    }
}

impl EventRecorder {
    fn snapshot(&self) -> Vec<ProgressEvent> {
        self.events.lock().expect("recorder lock").clone()
    }
}

#[wasm_bindgen_test]
async fn download_with_progress_handler_emits_lifecycle() {
    // Plain HTTP path (config.json is a small non-LFS file).
    let recorder = Arc::new(EventRecorder::default());
    let bytes = client()
        .model(TEST_MODEL_OWNER, TEST_MODEL_NAME)
        .download_file_to_bytes()
        .filename("config.json")
        .progress(recorder.clone())
        .send()
        .await
        .expect("download config.json with progress");

    let events = recorder.snapshot();
    assert!(!events.is_empty(), "no progress events recorded");

    let start_total = events.iter().find_map(|e| match e {
        ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => Some(*total_bytes),
        _ => None,
    });
    assert!(start_total.is_some(), "expected Download::Start event, got {events:#?}");
    assert_eq!(start_total.unwrap() as usize, bytes.len(), "Start.total_bytes did not match downloaded length",);

    assert!(
        events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Complete))),
        "expected Download::Complete event, got {events:#?}",
    );

    assert!(
        events.iter().all(|e| matches!(e, ProgressEvent::Download(_))),
        "every progress event from a download must be ProgressEvent::Download, got {events:#?}",
    );
}

#[wasm_bindgen_test]
async fn download_xet_with_progress_handler_reports_bytes() {
    // Xet path — drives `hf-xet`'s threaded wasm runtime. `download_file_to_bytes`
    // streams the file and the progress handler must see at least Start, some
    // byte-level progress (Progress or AggregateProgress), and Complete.
    let recorder = Arc::new(EventRecorder::default());
    let bytes = client()
        .model(XET_REPO_OWNER, XET_REPO_NAME)
        .download_file_to_bytes()
        .filename(XET_FILE_NAME)
        .progress(recorder.clone())
        .send()
        .await
        .expect("download xet pytorch_model.bin with progress");
    assert_eq!(bytes.len(), XET_FILE_SIZE);

    let events = recorder.snapshot();
    let start_total = events.iter().find_map(|e| match e {
        ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => Some(*total_bytes),
        _ => None,
    });
    assert_eq!(start_total.unwrap_or(0) as usize, XET_FILE_SIZE, "xet Start.total_bytes mismatch in {events:#?}",);

    let has_byte_progress = events.iter().any(|e| {
        matches!(
            e,
            ProgressEvent::Download(DownloadEvent::Progress { .. })
                | ProgressEvent::Download(DownloadEvent::AggregateProgress { .. })
        )
    });
    assert!(has_byte_progress, "expected at least one Progress or AggregateProgress event, got {events:#?}",);

    assert!(
        events
            .iter()
            .any(|e| matches!(e, ProgressEvent::Download(DownloadEvent::Complete))),
        "expected Download::Complete event, got {events:#?}",
    );
}

// `option_env!` reads at compile time, so the token + write-flag must be set
// when `run_tests.sh` invokes cargo (e.g.
// `HF_TOKEN=hf_xxx HF_TEST_WRITE=1 ./wasm/tests/run_tests.sh`). Missing
// either short-circuits the test to an early return — wasm-bindgen-test has
// no built-in skip, so this is the convention used here.
fn write_token() -> Option<&'static str> {
    if option_env!("HF_TEST_WRITE") != Some("1") {
        return None;
    }
    let token = option_env!("HF_TOKEN")?;
    if token.is_empty() { None } else { Some(token) }
}

fn authed_client(token: &str) -> HFClient {
    HFClientBuilder::new()
        .endpoint(ENDPOINT.to_string())
        .token(token.to_string())
        .build()
        .expect("build authed HFClient")
}

#[wasm_bindgen_test]
async fn upload_file_bytes_roundtrip() {
    let Some(token) = write_token() else {
        // HF_TOKEN / HF_TEST_WRITE not baked in at compile time — skip.
        return;
    };

    let client = authed_client(token);
    let username = client.whoami().send().await.expect("whoami").username;
    let name = format!("hf-hub-wasm-upload-{}", js_sys::Date::now() as u64);
    let repo_id = format!("{username}/{name}");

    client
        .create_repository()
        .repo_id(&repo_id)
        .repo_type(RepoTypeModel)
        .private(true)
        .exist_ok(true)
        .send()
        .await
        .expect("create test repo");

    let repo = client.model(&username, &name);
    let payload: &[u8] = b"hello from the wasm upload test\n";

    let commit = repo
        .upload_file()
        .source(AddSource::bytes(payload.to_vec()))
        .path_in_repo("wasm-upload-test.txt")
        .commit_message("wasm upload roundtrip")
        .send()
        .await
        .expect("upload_file");
    assert!(commit.commit_oid.is_some(), "expected commit_oid, got {commit:#?}");

    let downloaded = repo
        .download_file_to_bytes()
        .filename("wasm-upload-test.txt")
        .send()
        .await
        .expect("roundtrip download");
    assert_eq!(downloaded.as_ref(), payload, "uploaded and downloaded bytes differ");

    client
        .delete_repository()
        .repo_id(&repo_id)
        .repo_type(RepoTypeModel)
        .missing_ok(true)
        .send()
        .await
        .expect("delete test repo");
}

#[wasm_bindgen_test]
async fn upload_bucket_files_roundtrip() {
    let Some(token) = write_token() else {
        // HF_TOKEN / HF_TEST_WRITE not baked in at compile time — skip.
        return;
    };

    let client = authed_client(token);
    let username = client.whoami().send().await.expect("whoami").username;
    let bucket_name = format!("hf-hub-wasm-bucket-{}", js_sys::Date::now() as u64);
    let bucket_id = format!("{username}/{bucket_name}");

    client
        .create_bucket()
        .namespace(&username)
        .name(&bucket_name)
        .private(true)
        .exist_ok(true)
        .send()
        .await
        .expect("create test bucket");

    let bucket = client.bucket(&username, &bucket_name);
    let payload: &[u8] = b"hello from the wasm bucket upload test\n";
    let remote_path = "wasm-bucket-upload-test.txt".to_string();

    bucket
        .upload_source_files()
        .files(vec![(remote_path.clone(), AddSource::bytes(payload.to_vec()))])
        .send()
        .await
        .expect("bucket upload_source_files");

    let entries = bucket
        .get_paths_info()
        .paths(vec![remote_path.clone()])
        .send()
        .await
        .expect("get_paths_info after bucket upload");
    let entry = entries
        .into_iter()
        .find(|e| matches!(e, BucketTreeEntry::File { path, .. } if path == &remote_path))
        .expect("uploaded file not visible in bucket tree");
    let BucketTreeEntry::File { size, .. } = entry else {
        panic!("expected File entry, got {entry:?}");
    };
    assert_eq!(
        size,
        payload.len() as u64,
        "uploaded file size mismatch in bucket: got {size}, want {}",
        payload.len(),
    );

    client
        .delete_bucket()
        .bucket_id(&bucket_id)
        .missing_ok(true)
        .send()
        .await
        .expect("delete test bucket");
}
